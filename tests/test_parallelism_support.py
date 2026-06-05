from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dinovol_2.config import load_config
from dinovol_2.loss import KoLeoLoss, iBOTPatchLoss
from dinovol_2.model.dinov2_eva import EvaAttention
from dinovol_2.ops.collate import build_dino_ibot_collate_fn
from dinovol_2.ops.distributed_utils import resolve_distributed_config


def _fake_sample(global_size: int = 16, local_size: int = 8) -> dict:
    return {
        "global_views": [
            torch.zeros((1, global_size, global_size, global_size)),
            torch.ones((1, global_size, global_size, global_size)),
        ],
        "local_views": [torch.zeros((1, local_size, local_size, local_size))],
    }


def _distributed_koleo_worker(rank: int, world_size: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        x = torch.tensor([[float(rank + 1), float(rank + 2)]], requires_grad=True)
        loss = KoLeoLoss(distributed=True)(x)
        loss.backward()
        queue.put((rank, tuple(x.grad.shape), bool(torch.isfinite(x.grad).all().item()), float(loss.detach())))
    finally:
        dist.destroy_process_group()


def _tensor_parallel_attention_worker(rank: int, world_size: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(1234)
        full = EvaAttention(dim=16, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
        tp = EvaAttention(dim=16, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
        tp.load_state_dict(full.state_dict())
        group = dist.new_group(ranks=list(range(world_size)))
        tp.set_tensor_parallel(
            process_group=group,
            ranks=tuple(range(world_size)),
            rank=rank,
            world_size=world_size,
        )

        x = torch.randn(2, 7, 16, requires_grad=True)
        full_x = x.detach().clone().requires_grad_(True)
        full_y = full(full_x)
        tp_y = tp(x)
        grad = torch.randn_like(full_y)
        full_y.backward(grad)
        tp_y.backward(grad)
        queue.put(
            (
                rank,
                bool(torch.allclose(tp_y, full_y, atol=1e-5, rtol=1e-5)),
                bool(torch.allclose(x.grad, full_x.grad, atol=1e-5, rtol=1e-5)),
                bool(torch.isfinite(tp_y).all().item()),
                bool(torch.isfinite(x.grad).all().item()),
            )
        )
    finally:
        dist.destroy_process_group()


class ParallelismSupportTests(unittest.TestCase):
    def test_config_expands_env_and_datasets_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets_path = root / "datasets.json"
            datasets_path.write_text(
                json.dumps({"datasets": [{"volume_path": "${VOLUME_PATH}", "volume_scale": 0}]}),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_dir": "${OUTPUT_DIR}",
                        "dataset": {
                            "global_crop_size": [16, 16, 16],
                            "datasets_file": "datasets.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            old_env = dict(os.environ)
            try:
                os.environ["OUTPUT_DIR"] = str(root / "out")
                os.environ["VOLUME_PATH"] = str(root / "volume.zarr")
                config = load_config(config_path)
            finally:
                os.environ.clear()
                os.environ.update(old_env)

        self.assertEqual(config["output_dir"], str(root / "out"))
        self.assertEqual(config["dataset"]["datasets"][0]["volume_path"], str(root / "volume.zarr"))

    def test_config_fails_on_unresolved_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps({"dataset": {"datasets": [{"volume_path": "${MISSING_ENV}", "volume_scale": 0}]}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "MISSING_ENV"):
                load_config(config_path)

    def test_collate_caps_masked_patches(self) -> None:
        collate = build_dino_ibot_collate_fn(
            {
                "global_crop_size": [16, 16, 16],
                "patch_size": [4, 4, 4],
                "mask_ratio_min_max": (1.0, 1.0),
                "mask_sample_probability": 1.0,
                "max_masked_patches": 7,
            }
        )
        batch = collate([_fake_sample(), _fake_sample()])
        self.assertEqual(int(batch["n_masked_patches"].item()), 7)
        self.assertEqual(int(batch["collated_masks"].sum().item()), 7)
        self.assertEqual(batch["mask_indices_list"].numel(), 7)
        self.assertEqual(batch["masks_weight"].numel(), 7)

    def test_chunked_ibot_loss_matches_full_loss(self) -> None:
        torch.manual_seed(0)
        student = torch.randn(17, 11)
        teacher = torch.softmax(torch.randn(17, 11), dim=-1)
        masks = torch.ones((1, 17), dtype=torch.bool)
        weights = torch.ones(17) / 17
        full = iBOTPatchLoss(11, masked_loss_chunk_size=None)
        chunked = iBOTPatchLoss(11, masked_loss_chunk_size=4)
        full_loss = full.forward_masked(student, teacher, masks, n_masked_patches=17, masks_weight=weights)
        chunked_loss = chunked.forward_masked(student, teacher, masks, n_masked_patches=17, masks_weight=weights)
        self.assertTrue(torch.allclose(full_loss, chunked_loss, atol=1e-6, rtol=1e-6))

    def test_distributed_koleo_gathers_and_backprops(self) -> None:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "dist_init")
            queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(target=_distributed_koleo_worker, args=(rank, 2, init_file, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        self.assertEqual({rank for rank, _, _, _ in results}, {0, 1})
        for _, grad_shape, finite_grad, loss_value in results:
            self.assertEqual(grad_shape, (1, 2))
            self.assertTrue(finite_grad)
            self.assertTrue(torch.isfinite(torch.tensor(loss_value)).item())

    def test_tensor_parallel_attention_matches_full_attention(self) -> None:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "tp_dist_init")
            queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(target=_tensor_parallel_attention_worker, args=(rank, 2, init_file, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        self.assertEqual({rank for rank, *_ in results}, {0, 1})
        for _, output_close, grad_close, finite_output, finite_grad in results:
            self.assertTrue(output_close)
            self.assertTrue(grad_close)
            self.assertTrue(finite_output)
            self.assertTrue(finite_grad)

    def test_resolve_tp_ddp_topology_from_env(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ.update(
                {
                    "WORLD_SIZE": "16",
                    "LOCAL_WORLD_SIZE": "8",
                    "RANK": "5",
                    "LOCAL_RANK": "5",
                }
            )
            config = resolve_distributed_config(
                {"parallelism": {"strategy": "tp_ddp", "tensor_parallel_size": 4}}
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        self.assertTrue(config["use_ddp"])
        self.assertEqual(config["tensor_parallel_size"], 4)
        self.assertEqual(config["tensor_parallel_rank"], 1)
        self.assertEqual(config["data_parallel_world_size"], 4)
        self.assertEqual(config["data_parallel_rank"], 1)


if __name__ == "__main__":
    unittest.main()
