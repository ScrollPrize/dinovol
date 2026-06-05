from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from dinovol_2.config import load_config
from dinovol_2.loss import KoLeoLoss, iBOTPatchLoss
from dinovol_2.model.dinov2_eva import CompileStableDropPath, EvaAttention, EvaBlock
from dinovol_2.model.model import DinoVitStudentTeacher
from dinovol_2.model.rope import MixedRopePositionEmbedding
from dinovol_2.ops.collate import build_dino_ibot_collate_fn
from dinovol_2.ops.distributed_utils import resolve_distributed_config
from dinovol_2.ops.weighted_loader import WeightedCombinedLoader
from dinovol_2.pretrain import DinoIBOTPretrainer


def _fake_sample(global_size: int = 16, local_size: int = 8) -> dict:
    return {
        "global_views": [
            torch.zeros((1, global_size, global_size, global_size)),
            torch.ones((1, global_size, global_size, global_size)),
        ],
        "local_views": [torch.zeros((1, local_size, local_size, local_size))],
    }


def _tiny_compile_model_config() -> dict:
    return {
        "model_type": "v2",
        "input_channels": 1,
        "global_crops_size": [8, 8, 8],
        "local_crops_size": [4, 4, 4],
        "patch_size": [4, 4, 4],
        "embed_dim": 24,
        "depth": 2,
        "num_heads": 4,
        "num_reg_tokens": 1,
        "dino_out_dim": 16,
        "ibot_out_dim": 16,
        "dino_head_hidden_dim": 32,
        "dino_head_bottleneck_dim": 16,
        "ibot_head_hidden_dim": 32,
        "ibot_head_bottleneck_dim": 16,
        "use_abs_pos_emb": False,
        "use_rot_pos_emb": False,
        "drop_path_rate": 0.0,
        "block_chunks": 1,
        "masked_projection_chunk_size": 2,
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


def _context_parallel_attention_worker(rank: int, world_size: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(4321)
        full = EvaAttention(
            dim=16,
            num_heads=4,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            num_prefix_tokens=1,
            attention_mode="window_global_3d",
            window_size_patches=(2, 2, 2),
            shift_size_patches=(0, 0, 0),
        ).eval()
        cp = EvaAttention(
            dim=16,
            num_heads=4,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            num_prefix_tokens=1,
            attention_mode="window_global_3d",
            window_size_patches=(2, 2, 2),
            shift_size_patches=(0, 0, 0),
        ).eval()
        cp.load_state_dict(full.state_dict())
        group = dist.new_group(ranks=list(range(world_size)))
        cp.set_context_parallel(
            process_group=group,
            ranks=tuple(range(world_size)),
            rank=rank,
            world_size=world_size,
        )

        x = torch.randn(1, 9, 16)
        start = rank * 4
        end = start + 4
        x_local = torch.cat((x[:, :1], x[:, 1 + start:1 + end]), dim=1)
        full_y = full(x, rope_shape=(2, 2, 2))
        cp_y = cp(x_local, rope_shape=(2, 2, 2))
        expected = torch.cat((full_y[:, :1], full_y[:, 1 + start:1 + end]), dim=1)
        queue.put((rank, tuple(cp_y.shape), bool(torch.allclose(cp_y, expected, atol=1e-5, rtol=1e-5))))
    finally:
        dist.destroy_process_group()


def _context_parallel_masked_gather_worker(rank: int, world_size: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)))
        local_positions = torch.tensor([rank, rank + world_size], dtype=torch.long)
        local_projections = torch.tensor(
            [[float(rank), 1.0], [float(rank + world_size), 1.0]],
            requires_grad=True,
        )
        expected_positions = torch.arange(world_size * 2, dtype=torch.long)
        gathered = DinoVitStudentTeacher._gather_context_parallel_masked_projections(
            local_projections,
            local_positions,
            expected_positions,
            process_group=group,
            world_size=world_size,
        )
        gathered.sum().backward()
        queue.put(
            (
                rank,
                tuple(gathered.shape),
                bool(torch.equal(gathered[:, 0].detach().long(), expected_positions)),
                bool(torch.allclose(local_projections.grad, torch.ones_like(local_projections))),
            )
        )
    finally:
        dist.destroy_process_group()


class ParallelismSupportTests(unittest.TestCase):
    def test_compile_config_validation(self) -> None:
        default = DinoIBOTPretrainer._resolve_compile_config(None)
        self.assertFalse(default["enabled"])
        self.assertEqual(default["scope"], "blocks")
        self.assertEqual(default["backend"], "inductor")
        self.assertEqual(default["mode"], "default")
        self.assertFalse(default["fullgraph"])
        self.assertFalse(default["dynamic"])

        configured = DinoIBOTPretrainer._resolve_compile_config(
            {
                "enabled": True,
                "scope": "blocks_and_heads",
                "backend": "inductor",
                "mode": "max-autotune",
                "fullgraph": True,
                "dynamic": "auto",
            }
        )
        self.assertTrue(configured["enabled"])
        self.assertEqual(configured["scope"], "blocks_and_heads")
        self.assertEqual(configured["mode"], "max-autotune")
        self.assertTrue(configured["fullgraph"])
        self.assertIsNone(configured["dynamic"])

        diagnostic = DinoIBOTPretrainer._resolve_compile_config({"enabled": True, "mode": "reduce-overhead"})
        self.assertEqual(diagnostic["mode"], "reduce-overhead")

        with self.assertRaisesRegex(ValueError, "compile.scope"):
            DinoIBOTPretrainer._resolve_compile_config({"scope": "everything"})
        with self.assertRaisesRegex(ValueError, "compile.mode"):
            DinoIBOTPretrainer._resolve_compile_config({"mode": "fastest"})

    def test_cluster_gpu_metrics_config_and_formatting(self) -> None:
        trainer = object.__new__(DinoIBOTPretrainer)
        trainer.log_every = 20

        default = trainer._resolve_cluster_metrics_config(None)
        self.assertFalse(default["enabled"])
        self.assertEqual(default["every_n"], 20)
        self.assertTrue(default["query_nvidia_smi"])
        self.assertTrue(default["log_per_rank"])

        enabled = trainer._resolve_cluster_metrics_config(
            {
                "enabled": True,
                "every_n": 5,
                "query_nvidia_smi": False,
                "log_per_rank": False,
            }
        )
        self.assertTrue(enabled["enabled"])
        self.assertEqual(enabled["every_n"], 5)
        self.assertFalse(enabled["query_nvidia_smi"])
        self.assertFalse(enabled["log_per_rank"])

        with self.assertRaisesRegex(ValueError, "cluster_metrics.every_n"):
            trainer._resolve_cluster_metrics_config({"enabled": True, "every_n": 0})

        rows = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1.0, 2.0, 3.0, 4.0, 10.0, 80.0, 50.0, 700.0, 41.0],
                [1, 1, 0, 1, 0, 1, 2.0, 4.0, 5.0, 6.0, 20.0, 80.0, 75.0, 650.0, 43.0],
            ],
            dtype=torch.float64,
        )
        payload = DinoIBOTPretrainer._format_cluster_gpu_metrics(rows, log_per_rank=True)
        self.assertEqual(payload["cluster/gpu_utilization_pct/max"], 75.0)
        self.assertEqual(payload["cluster/gpu_power_w/sum"], 1350.0)
        self.assertEqual(payload["cluster/torch_memory_reserved_gib/mean"], 3.0)
        self.assertEqual(payload["cluster/rank_001/local_rank"], 1.0)
        self.assertEqual(payload["cluster/rank_001/context_parallel_rank"], 1.0)

    def test_compile_blocks_and_heads_preserves_state_dict_keys(self) -> None:
        model = DinoVitStudentTeacher(_tiny_compile_model_config()).eval()
        trainer = object.__new__(DinoIBOTPretrainer)
        trainer.model = model
        trainer.rank = 1
        trainer.compiled_module_names = ()
        trainer.compile_config = DinoIBOTPretrainer._resolve_compile_config(
            {
                "enabled": True,
                "scope": "blocks_and_heads",
                "backend": "eager",
                "mode": "default",
            }
        )

        before_keys = tuple(model.state_dict().keys())
        trainer._apply_compile_if_configured()
        after_keys = tuple(model.state_dict().keys())

        self.assertEqual(after_keys, before_keys)
        self.assertEqual(len(trainer.compiled_module_names), 8)
        self.assertIn("student.backbone.blocks.0.0", trainer.compiled_module_names)
        self.assertIn("teacher.ibot_head", trainer.compiled_module_names)

    def test_compile_stable_drop_path_preserves_basic_semantics(self) -> None:
        x = torch.ones(32, 4, 3)
        zero = CompileStableDropPath(0.0).train()
        self.assertTrue(torch.equal(zero(x), x))

        drop = CompileStableDropPath(0.5).train()
        torch.manual_seed(123)
        y = drop(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(set(torch.unique(y).tolist()).issubset({0.0, 2.0}))

        drop.eval()
        self.assertTrue(torch.equal(drop(x), x))

    def test_compile_blocks_and_heads_matches_eager_forward(self) -> None:
        torch.manual_seed(11)
        eager = DinoVitStudentTeacher(_tiny_compile_model_config()).eval()
        compiled = DinoVitStudentTeacher(_tiny_compile_model_config()).eval()
        compiled.load_state_dict(eager.state_dict())

        trainer = object.__new__(DinoIBOTPretrainer)
        trainer.model = compiled
        trainer.rank = 1
        trainer.compiled_module_names = ()
        trainer.compile_config = DinoIBOTPretrainer._resolve_compile_config(
            {
                "enabled": True,
                "scope": "blocks_and_heads",
                "backend": "eager",
                "mode": "default",
            }
        )
        trainer._apply_compile_if_configured()

        student_input = torch.randn(2, 1, 8, 8, 8)
        local_input = torch.randn(2, 1, 4, 4, 4)
        masks = torch.zeros(2, 8, dtype=torch.bool)
        mask_indices = torch.tensor([0, 9], dtype=torch.long)

        def assert_nested_close(left: object, right: object) -> None:
            if isinstance(left, torch.Tensor):
                self.assertIsInstance(right, torch.Tensor)
                self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))
                return
            self.assertIsInstance(left, dict)
            self.assertIsInstance(right, dict)
            self.assertEqual(set(left.keys()), set(right.keys()))
            for key in left:
                assert_nested_close(left[key], right[key])

        with torch.no_grad():
            eager_outputs = eager(
                student_input,
                student_masks=masks,
                local_student_input=local_input,
                mask_indices_list=mask_indices,
                n_masked_patches=2,
                return_teacher=True,
            )
            compiled_outputs = compiled(
                student_input,
                student_masks=masks,
                local_student_input=local_input,
                mask_indices_list=mask_indices,
                n_masked_patches=2,
                return_teacher=True,
            )

        assert_nested_close(eager_outputs, compiled_outputs)

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

    def test_weighted_loader_rejects_zero_batch_loader(self) -> None:
        empty_loader = DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=1, drop_last=True)
        loader = WeightedCombinedLoader([empty_loader], weights=(1.0,))
        with self.assertRaisesRegex(RuntimeError, "zero batches"):
            iter(loader)

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

    def test_context_parallel_window_attention_matches_full_local_slice(self) -> None:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "cp_dist_init")
            queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(target=_context_parallel_attention_worker, args=(rank, 2, init_file, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        self.assertEqual({rank for rank, *_ in results}, {0, 1})
        for _, output_shape, output_close in results:
            self.assertEqual(output_shape, (1, 5, 16))
            self.assertTrue(output_close)

    def test_context_parallel_masked_projection_gather_preserves_order_and_grad(self) -> None:
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "cp_mask_dist_init")
            queue: mp.Queue = ctx.Queue()
            processes = [
                ctx.Process(target=_context_parallel_masked_gather_worker, args=(rank, 2, init_file, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

        self.assertEqual({rank for rank, *_ in results}, {0, 1})
        for _, gathered_shape, order_ok, grad_ok in results:
            self.assertEqual(gathered_shape, (4, 2))
            self.assertTrue(order_ok)
            self.assertTrue(grad_ok)

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

    def test_resolve_mesh_topology_from_env(self) -> None:
        old_env = dict(os.environ)
        try:
            os.environ.update(
                {
                    "WORLD_SIZE": "40",
                    "LOCAL_WORLD_SIZE": "8",
                    "RANK": "23",
                    "LOCAL_RANK": "7",
                }
            )
            config = resolve_distributed_config(
                {
                    "parallelism": {
                        "strategy": "mesh",
                        "tensor_parallel_size": 4,
                        "context_parallel_size": 2,
                        "sharding": "zero1",
                    }
                }
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        self.assertTrue(config["use_ddp"])
        self.assertEqual(config["tensor_parallel_size"], 4)
        self.assertEqual(config["context_parallel_size"], 2)
        self.assertEqual(config["model_parallel_size"], 8)
        self.assertEqual(config["tensor_parallel_rank"], 3)
        self.assertEqual(config["context_parallel_rank"], 1)
        self.assertEqual(config["data_parallel_world_size"], 5)
        self.assertEqual(config["data_parallel_rank"], 2)
        self.assertEqual(config["sharding"], "zero1")

    def test_window_global_attention_matches_dense_for_full_mixed_rope_window(self) -> None:
        torch.manual_seed(7)
        kwargs = {
            "dim": 24,
            "num_heads": 4,
            "qkv_bias": True,
            "qkv_fused": True,
            "num_prefix_tokens": 2,
            "rope_impl": MixedRopePositionEmbedding,
            "rope_kwargs": {"base": 100.0},
            "ndim": 3,
        }
        dense = EvaBlock(**kwargs).eval()
        windowed = EvaBlock(
            **kwargs,
            attention_mode="window_global_3d",
            window_size_patches=(2, 2, 2),
            shift_size_patches=(0, 0, 0),
        ).eval()
        windowed.load_state_dict(dense.state_dict())
        x = torch.randn(2, 10, 24)

        dense_output = dense(x, rope_shape=(2, 2, 2))
        windowed_output = windowed(x, rope_shape=(2, 2, 2))

        self.assertTrue(torch.allclose(windowed_output, dense_output, atol=1e-5, rtol=1e-5))

    def test_shifted_window_global_attention_is_finite_and_shape_stable(self) -> None:
        torch.manual_seed(8)
        block = EvaBlock(
            dim=24,
            num_heads=4,
            qkv_bias=True,
            qkv_fused=True,
            num_prefix_tokens=2,
            rope_impl=MixedRopePositionEmbedding,
            rope_kwargs={"base": 100.0},
            ndim=3,
            attention_mode="window_global_3d",
            window_size_patches=(2, 2, 2),
            shift_size_patches=(1, 1, 1),
        ).eval()
        x = torch.randn(2, 66, 24)

        output = block(x, rope_shape=(4, 4, 4))

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_mlp_token_chunking_matches_unchunked_block(self) -> None:
        torch.manual_seed(9)
        kwargs = {
            "dim": 24,
            "num_heads": 4,
            "qkv_bias": True,
            "qkv_fused": True,
            "num_prefix_tokens": 2,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "drop_path": 0.0,
            "norm_layer": torch.nn.LayerNorm,
        }
        full = EvaBlock(**kwargs).eval()
        chunked = EvaBlock(**kwargs, mlp_token_chunk_size=5).eval()
        chunked.load_state_dict(full.state_dict())
        x = torch.randn(2, 17, 24)

        self.assertTrue(torch.allclose(chunked(x), full(x), atol=1e-6, rtol=1e-6))

    def test_view_chunking_matches_full_forward_and_can_drop_local_patch_tokens(self) -> None:
        torch.manual_seed(10)
        config = {
            "model_type": "v2",
            "input_channels": 1,
            "global_crops_size": [8, 8, 8],
            "local_crops_size": [4, 4, 4],
            "patch_size": [4, 4, 4],
            "embed_dim": 24,
            "depth": 1,
            "num_heads": 4,
            "num_reg_tokens": 1,
            "dino_out_dim": 16,
            "ibot_out_dim": 16,
            "dino_head_hidden_dim": 32,
            "dino_head_bottleneck_dim": 16,
            "ibot_head_hidden_dim": 32,
            "ibot_head_bottleneck_dim": 16,
            "use_abs_pos_emb": False,
            "use_rot_pos_emb": False,
            "drop_path_rate": 0.0,
            "masked_projection_chunk_size": 2,
        }
        full = DinoVitStudentTeacher(config).eval()
        chunked = DinoVitStudentTeacher({**config, "view_chunk_size": 1}).eval()
        chunked.load_state_dict(full.state_dict())

        student_input = torch.randn(2, 1, 8, 8, 8)
        local_input = torch.randn(3, 1, 4, 4, 4)
        masks = torch.zeros(2, 8, dtype=torch.bool)
        mask_indices = torch.tensor([0, 9], dtype=torch.long)
        with torch.no_grad():
            full_outputs = full(
                student_input,
                student_masks=masks,
                local_student_input=local_input,
                mask_indices_list=mask_indices,
                n_masked_patches=2,
                return_teacher=False,
            )["student"]
            chunked_outputs = chunked(
                student_input,
                student_masks=masks,
                local_student_input=local_input,
                mask_indices_list=mask_indices,
                n_masked_patches=2,
                return_teacher=False,
            )["student"]

        for key in ("cls_tokens", "patch_tokens"):
            self.assertTrue(torch.allclose(chunked_outputs["global"][key], full_outputs["global"][key], atol=1e-6, rtol=1e-6))
        for key in ("global_cls_projections", "global_masked_patch_projections"):
            self.assertTrue(torch.allclose(chunked_outputs[key], full_outputs[key], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(chunked_outputs["local"]["cls_projections"], full_outputs["local"]["cls_projections"], atol=1e-6, rtol=1e-6))

        drop_local = DinoVitStudentTeacher({**config, "view_chunk_size": 1, "discard_local_patch_tokens": True}).eval()
        drop_local.load_state_dict(full.state_dict())
        with torch.no_grad():
            drop_outputs = drop_local(
                student_input,
                student_masks=masks,
                local_student_input=local_input,
                mask_indices_list=mask_indices,
                n_masked_patches=2,
                return_teacher=False,
            )["student"]
        self.assertNotIn("patch_tokens", drop_outputs["local"])
        self.assertTrue(torch.allclose(drop_outputs["local"]["cls_projections"], full_outputs["local"]["cls_projections"], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
