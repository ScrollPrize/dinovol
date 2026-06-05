from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from dinovol_2.config import load_config
from dinovol_2.pretrain import DinoIBOTPretrainer


def _memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "current_allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
        "current_reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
    }


def _sdpa_backend_stats(device: torch.device) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "torch_version": torch.__version__,
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    cudnn_sdp_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
    if callable(cudnn_sdp_enabled):
        stats["cudnn_sdp_enabled"] = bool(cudnn_sdp_enabled())
    if device.type == "cuda":
        stats["cuda_device_name"] = torch.cuda.get_device_name(device)
        stats["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return stats


def _batch_stats(batch: dict[str, Any]) -> dict[str, Any]:
    global_crops = batch["collated_global_crops"]
    local_crops = batch["collated_local_crops"]
    masks = batch["collated_masks"]
    return {
        "global_crop_shape": list(global_crops.shape),
        "local_crop_shape": list(local_crops.shape),
        "global_tokens": int(masks.shape[1]),
        "n_masked_patches": int(batch["n_masked_patches"].item()),
        "n_global_views": int(batch["n_global_views"]),
        "n_local_views": int(batch["n_local_views"]),
        "batch_size": int(batch["batch_size"]),
    }


def _gather_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_available() or not dist.is_initialized():
        return [report]
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, report)
    return [item for item in gathered if item is not None]


def run_profile(config_path: Path, *, steps: int, train: bool, no_resume: bool) -> dict[str, Any] | None:
    config = load_config(config_path)
    if no_resume:
        config["resume"] = False
        config["auto_resume"] = False
        config.pop("resume_from", None)
    config.setdefault("wandb_project", None)
    config.setdefault("val_every_n", 0)
    config.setdefault("save_every_n", 0)

    trainer = DinoIBOTPretrainer(config)
    dataloader = trainer.build_dataloader()
    reports: list[dict[str, Any]] = []
    try:
        iterator = iter(dataloader)
        for step in range(int(steps)):
            accumulation_steps = trainer.gradient_accumulation_steps if train else 1
            if accumulation_steps > 1:
                batch = [next(iterator) for _ in range(accumulation_steps)]
                batch_for_stats = batch[0]
            else:
                batch = next(iterator)
                batch_for_stats = batch
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(trainer.device)
                torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()
            if train:
                metrics = trainer.train_step(batch, step)
            else:
                metrics = trainer.validate(batch, step)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            elapsed = time.perf_counter() - start
            reports.append(
                {
                    "rank": trainer.rank,
                    "local_rank": trainer.local_rank,
                    "data_parallel_rank": trainer.data_parallel_rank,
                    "tensor_parallel_rank": trainer.tensor_parallel_rank,
                    "context_parallel_rank": trainer.context_parallel_rank,
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "batch": {
                        **_batch_stats(batch_for_stats),
                        "gradient_accumulation_steps": int(accumulation_steps),
                    },
                    "metrics": {key: float(value) for key, value in metrics.items()},
                    "memory": _memory_stats(trainer.device),
                }
            )
    finally:
        trainer._close_dataloader(dataloader)
        trainer._close_auxiliary_datasets()
        trainer._finish_wandb()
        if trainer.is_distributed and dist.is_initialized():
            dist.barrier()

    gathered = _gather_reports({"rank": trainer.rank, "reports": reports})
    if trainer.is_main_process:
        return {
            "config_path": str(config_path),
            "steps": int(steps),
            "train": bool(train),
            "world_size": trainer.world_size,
            "data_parallel_world_size": trainer.data_parallel_world_size,
            "tensor_parallel_size": trainer.tensor_parallel_size,
            "context_parallel_size": trainer.context_parallel_size,
            "parallelism_sharding": trainer.parallelism_sharding,
            "attention_mode": str(trainer.model_config.get("attention_mode", "dense")),
            "sdpa_backend": _sdpa_backend_stats(trainer.device),
            "rank_reports": gathered,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one or more DINO pretraining steps.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--validate", action="store_true", help="Run validation forward/loss only instead of train steps.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_profile(args.config, steps=args.steps, train=not args.validate, no_resume=args.no_resume)
    if report is not None:
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
