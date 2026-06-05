from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _env_snapshot() -> dict[str, str]:
    prefixes = ("MASTER_", "WORLD_", "RANK", "LOCAL_", "NCCL_", "GLOO_", "TORCH_NCCL_")
    return {key: value for key, value in sorted(os.environ.items()) if key.startswith(prefixes)}


def run_probe(*, min_bytes: int, max_bytes: int, steps: int) -> dict[str, Any] | None:
    device = _device()
    backend = "nccl" if device.type == "cuda" else "gloo"
    init_kwargs: dict[str, Any] = {"backend": backend, "init_method": "env://"}
    if backend == "nccl":
        init_kwargs["device_id"] = device
    dist.init_process_group(**init_kwargs)
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        results = []
        size = int(min_bytes)
        while size <= int(max_bytes):
            numel = max(1, size // 4)
            tensor = torch.ones(numel, device=device, dtype=torch.float32)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dist.barrier()
            start = time.perf_counter()
            for _ in range(int(steps)):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            expected = float(world ** steps)
            ok = bool(torch.isclose(tensor[0].detach().cpu(), torch.tensor(expected)).item())
            results.append(
                {
                    "bytes": int(numel * 4),
                    "steps": int(steps),
                    "elapsed_seconds": float(elapsed),
                    "avg_seconds": float(elapsed / max(1, int(steps))),
                    "ok": ok,
                }
            )
            size *= 2
        report = {
            "rank": int(rank),
            "world_size": int(world),
            "backend": backend,
            "device": str(device),
            "env": _env_snapshot(),
            "results": results,
        }
        gathered: list[dict[str, Any] | None] = [None for _ in range(world)]
        dist.all_gather_object(gathered, report)
        if rank == 0:
            return {"rank_reports": [item for item in gathered if item is not None]}
        return None
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small torch.distributed all-reduce fabric probe.")
    parser.add_argument("--min-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-bytes", type=int, default=1 << 26)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_probe(min_bytes=args.min_bytes, max_bytes=args.max_bytes, steps=args.steps)
    if report is None:
        return
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
