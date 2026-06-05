from collections import OrderedDict
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, use_fused_attn, SwiGLU, GluMlp, Mlp
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from dinovol_2.model.patch_encode_decode import PatchEmbed, PatchEmbedDeeper
from dinovol_2.model.rope import (
    MixedRopePositionEmbedding,
    RopeCoords,
    RopeEmbedding,
    RopePositionEmbedding,
    apply_rotary_embedding,
)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group):
        ctx.process_group = process_group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.process_group is not None and dist.is_available() and dist.is_initialized():
            grad_output = grad_output.contiguous()
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, process_group):
        ctx.process_group = process_group
        if process_group is not None and dist.is_available() and dist.is_initialized():
            x = x.contiguous()
            dist.all_reduce(x, group=process_group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def _copy_to_tensor_parallel_region(x: torch.Tensor, process_group):
    return _CopyToTensorParallelRegion.apply(x, process_group)


def _reduce_from_tensor_parallel_region(x: torch.Tensor, process_group):
    return _ReduceFromTensorParallelRegion.apply(x, process_group)


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_checkpoint_wrapped_module", module)


class CompileStableDropPath(nn.Module):
    """Stochastic depth without Python-value guards on per-block drop rates."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)
        self.register_buffer(
            "_drop_prob_tensor",
            torch.tensor(self.drop_prob, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        keep_prob = (1.0 - self._drop_prob_tensor).to(device=x.device)
        keep_prob = keep_prob.clamp_min(torch.finfo(torch.float32).tiny)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, device=x.device, dtype=torch.float32) < keep_prob
        random_tensor = random_tensor.to(dtype=x.dtype)
        if self.scale_by_keep:
            random_tensor = random_tensor / keep_prob.to(dtype=x.dtype)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]
    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
            attention_mode: str = "dense",
            window_size_patches: Optional[Tuple[int, int, int]] = None,
            shift_size_patches: Optional[Tuple[int, int, int]] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        if attn_head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()
        self.qkv_bias_separate = qkv_bias_separate
        
        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.tensor_parallel_size = 1
        self.tensor_parallel_rank = 0
        self.tensor_parallel_group = None
        self.tensor_parallel_ranks: tuple[int, ...] | None = None
        self.context_parallel_size = 1
        self.context_parallel_rank = 0
        self.context_parallel_group = None
        self.context_parallel_ranks: tuple[int, ...] | None = None
        self.attention_mode = str(attention_mode).strip().lower()
        if self.attention_mode not in {"dense", "context_parallel_dense", "window_global_3d"}:
            raise ValueError(
                "attention_mode must be one of dense, context_parallel_dense, or window_global_3d; "
                f"got {attention_mode!r}"
            )
        self.window_size_patches = self._normalize_window_tuple(
            window_size_patches,
            name="window_size_patches",
            default=(0, 0, 0),
        )
        self.shift_size_patches = self._normalize_window_tuple(
            shift_size_patches,
            name="shift_size_patches",
            default=(0, 0, 0),
        )

    @property
    def is_tensor_parallel(self) -> bool:
        return self.tensor_parallel_size > 1

    @property
    def head_dim(self) -> int:
        return self.proj.in_features // self.num_heads

    @property
    def local_num_heads(self) -> int:
        return self.num_heads // self.tensor_parallel_size

    @staticmethod
    def _normalize_window_tuple(
            value: Optional[Tuple[int, int, int]],
            *,
            name: str,
            default: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        if value is None:
            return default
        if isinstance(value, int):
            result = (int(value), int(value), int(value))
        else:
            result = tuple(int(item) for item in value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain three integers, got {result}")
        if any(item < 0 for item in result):
            raise ValueError(f"{name} must be non-negative, got {result}")
        return result

    @property
    def local_channel_slice(self) -> slice:
        start = self.tensor_parallel_rank * self.local_num_heads * self.head_dim
        stop = start + self.local_num_heads * self.head_dim
        return slice(start, stop)

    @property
    def is_context_parallel(self) -> bool:
        return self.context_parallel_size > 1

    def _context_parallel_patch_range(self, rope_shape: Optional[Tuple[int, ...]]) -> tuple[int, int, int]:
        if rope_shape is None:
            raise ValueError("context parallel attention requires rope_shape.")
        if len(rope_shape) != 3:
            raise ValueError(f"context parallel attention requires a 3D rope_shape, got {rope_shape}.")
        depth = int(rope_shape[0])
        height = int(rope_shape[1])
        width = int(rope_shape[2])
        n_patches = depth * height * width
        if n_patches % self.context_parallel_size != 0:
            raise ValueError(
                f"patch token count {n_patches} must be divisible by context_parallel_size="
                f"{self.context_parallel_size}."
            )
        local_tokens = n_patches // self.context_parallel_size
        start = self.context_parallel_rank * local_tokens
        return start, start + local_tokens, n_patches

    def _gather_context_parallel_input(
            self,
            x: torch.Tensor,
            *,
            rope_shape: Optional[Tuple[int, ...]],
    ) -> tuple[torch.Tensor, slice] | tuple[torch.Tensor, None]:
        if not self.is_context_parallel:
            return x, None
        if self.context_parallel_group is None:
            raise RuntimeError("context_parallel_group is not initialized.")
        if self.attention_mode != "window_global_3d":
            raise ValueError("context parallel token sharding currently requires window_global_3d attention.")
        start, end, n_patches = self._context_parallel_patch_range(rope_shape)
        local_patch_tokens = end - start
        if x.shape[1] != self.num_prefix_tokens + local_patch_tokens:
            raise ValueError(
                "context parallel attention expected local patch-token input: "
                f"got tokens={x.shape[1]}, prefix={self.num_prefix_tokens}, local_patches={local_patch_tokens}, "
                f"full_patches={n_patches}"
            )
        prefix = x[:, :self.num_prefix_tokens]
        local_patch = x[:, self.num_prefix_tokens:].contiguous()
        gathered = dist_nn.all_gather(local_patch, group=self.context_parallel_group)
        full_patch = torch.cat(tuple(gathered), dim=1)
        if full_patch.shape[1] != n_patches:
            raise RuntimeError(
                f"context parallel gather produced {full_patch.shape[1]} patches, expected {n_patches}."
            )
        return torch.cat((prefix, full_patch), dim=1), slice(
            self.num_prefix_tokens + start,
            self.num_prefix_tokens + end,
        )

    def _qkv_indices_for_channel_slice(self, channel_slice: slice) -> torch.Tensor:
        all_head_dim = self.proj.in_features
        starts = (
            channel_slice.start,
            all_head_dim + channel_slice.start,
            2 * all_head_dim + channel_slice.start,
        )
        pieces = [
            torch.arange(start, start + (channel_slice.stop - channel_slice.start), device=self.qkv.weight.device)
            for start in starts
        ]
        return torch.cat(pieces, dim=0)

    def set_tensor_parallel(
        self,
        *,
        process_group=None,
        ranks: tuple[int, ...] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        world_size = int(world_size)
        rank = int(rank)
        if world_size <= 0:
            raise ValueError(f"tensor parallel world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"tensor parallel rank must be in [0, {world_size}), got {rank}.")
        if self.num_heads % world_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by tensor_parallel_size={world_size}."
            )
        if not isinstance(self.norm, nn.Identity):
            raise ValueError("attention tensor parallelism does not support scale_attn_inner/norm_layer yet.")
        if self.attn_drop.p != 0.0 or self.proj_drop.p != 0.0:
            raise ValueError("attention tensor parallelism requires attn_drop_rate=proj_drop_rate=0.0.")
        self.tensor_parallel_group = process_group
        self.tensor_parallel_ranks = ranks
        self.tensor_parallel_rank = rank
        self.tensor_parallel_size = world_size

    def set_context_parallel(
        self,
        *,
        process_group=None,
        ranks: tuple[int, ...] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        world_size = int(world_size)
        rank = int(rank)
        if world_size <= 0:
            raise ValueError(f"context parallel world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"context parallel rank must be in [0, {world_size}), got {rank}.")
        self.context_parallel_group = process_group
        self.context_parallel_ranks = ranks
        self.context_parallel_rank = rank
        self.context_parallel_size = world_size

    @staticmethod
    def _slice_rope(rope: Optional[RopeEmbedding], head_slice: slice) -> Optional[RopeEmbedding]:
        if rope is None:
            return None
        sin, cos = rope
        if sin.ndim >= 3 and sin.shape[0] >= head_slice.stop:
            return sin[head_slice], cos[head_slice]
        return rope

    def _linear_qkv_tensor_parallel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        channel_slice = self.local_channel_slice
        if self.qkv is not None:
            indices = self._qkv_indices_for_channel_slice(channel_slice)
            weight = self.qkv.weight.index_select(0, indices)
            if self.q_bias is None:
                bias = None
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                bias = qkv_bias.index_select(0, indices)
            qkv = F.linear(x, weight=weight, bias=bias)
            qkv = qkv.reshape(B, N, 3, self.local_num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            return qkv.unbind(0)

        q = F.linear(
            x,
            self.q_proj.weight[channel_slice],
            self.q_proj.bias[channel_slice] if self.q_proj.bias is not None else None,
        ).reshape(B, N, self.local_num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(
            x,
            self.k_proj.weight[channel_slice],
            self.k_proj.bias[channel_slice] if self.k_proj.bias is not None else None,
        ).reshape(B, N, self.local_num_heads, self.head_dim).transpose(1, 2)
        v = F.linear(
            x,
            self.v_proj.weight[channel_slice],
            self.v_proj.bias[channel_slice] if self.v_proj.bias is not None else None,
        ).reshape(B, N, self.local_num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _partition_3d_windows(
            x: torch.Tensor,
            spatial_shape: Tuple[int, int, int],
            window_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        bsz, heads, depth, height, width, channels = x.shape
        window_d, window_h, window_w = window_shape
        return (
            x.view(
                bsz,
                heads,
                depth // window_d,
                window_d,
                height // window_h,
                window_h,
                width // window_w,
                window_w,
                channels,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7, 8)
            .reshape(-1, heads, window_d * window_h * window_w, channels)
        )

    @staticmethod
    def _reverse_3d_windows(
            windows: torch.Tensor,
            *,
            batch_size: int,
            spatial_shape: Tuple[int, int, int],
            window_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        depth, height, width = spatial_shape
        window_d, window_h, window_w = window_shape
        heads = windows.shape[1]
        channels = windows.shape[-1]
        return (
            windows.view(
                batch_size,
                depth // window_d,
                height // window_h,
                width // window_w,
                heads,
                window_d,
                window_h,
                window_w,
                channels,
            )
            .permute(0, 4, 1, 5, 2, 6, 3, 7, 8)
            .reshape(batch_size, heads, depth, height, width, channels)
        )

    def _window_global_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            rope_shape: Optional[Tuple[int, ...]],
    ) -> torch.Tensor:
        if rope_shape is None or len(rope_shape) != 3:
            raise ValueError("window_global_3d attention requires a 3D patch grid shape.")
        depth = int(rope_shape[0])
        height = int(rope_shape[1])
        width = int(rope_shape[2])
        spatial_shape = (depth, height, width)
        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"window_global_3d received invalid spatial shape: {spatial_shape}")
        patch_tokens = depth * height * width
        if q.shape[-2] != self.num_prefix_tokens + patch_tokens:
            raise ValueError(
                "window_global_3d token count mismatch: "
                f"tokens={q.shape[-2]}, prefix={self.num_prefix_tokens}, patch_grid={spatial_shape}"
            )

        configured_window_d, configured_window_h, configured_window_w = self.window_size_patches
        window_d = min(configured_window_d, depth) if configured_window_d > 0 else depth
        window_h = min(configured_window_h, height) if configured_window_h > 0 else height
        window_w = min(configured_window_w, width) if configured_window_w > 0 else width
        window_shape = (window_d, window_h, window_w)
        if depth % window_d != 0 or height % window_h != 0 or width % window_w != 0:
            raise ValueError(
                "window_global_3d requires each patch-grid dimension to be divisible by the window size; "
                f"got spatial_shape={spatial_shape}, window_size_patches={window_shape}"
            )

        configured_shift_d, configured_shift_h, configured_shift_w = self.shift_size_patches
        shift_d = min(configured_shift_d, window_d - 1) if depth > window_d else 0
        shift_h = min(configured_shift_h, window_h - 1) if height > window_h else 0
        shift_w = min(configured_shift_w, window_w - 1) if width > window_w else 0
        shift_shape = (shift_d, shift_h, shift_w)
        has_shift = shift_d != 0 or shift_h != 0 or shift_w != 0

        q_prefix = q[:, :, :self.num_prefix_tokens]
        k_prefix = k[:, :, :self.num_prefix_tokens]
        v_prefix = v[:, :, :self.num_prefix_tokens]
        q_patch = q[:, :, self.num_prefix_tokens:].reshape(*q.shape[:2], *spatial_shape, q.shape[-1])
        k_patch = k[:, :, self.num_prefix_tokens:].reshape(*k.shape[:2], *spatial_shape, k.shape[-1])
        v_patch = v[:, :, self.num_prefix_tokens:].reshape(*v.shape[:2], *spatial_shape, v.shape[-1])

        prefix_out = F.scaled_dot_product_attention(
            q_prefix,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        if has_shift:
            roll_shifts = (-shift_d, -shift_h, -shift_w)
            q_patch = torch.roll(q_patch, shifts=roll_shifts, dims=(2, 3, 4))
            k_patch = torch.roll(k_patch, shifts=roll_shifts, dims=(2, 3, 4))
            v_patch = torch.roll(v_patch, shifts=roll_shifts, dims=(2, 3, 4))

        q_windows = self._partition_3d_windows(q_patch, spatial_shape, window_shape)
        k_windows = self._partition_3d_windows(k_patch, spatial_shape, window_shape)
        v_windows = self._partition_3d_windows(v_patch, spatial_shape, window_shape)
        windows_per_sample = q_windows.shape[0] // q.shape[0]
        prefix_k_windows = (
            k_prefix[:, None]
            .expand(-1, windows_per_sample, -1, -1, -1)
            .reshape(-1, k_prefix.shape[1], k_prefix.shape[2], k_prefix.shape[3])
        )
        prefix_v_windows = (
            v_prefix[:, None]
            .expand(-1, windows_per_sample, -1, -1, -1)
            .reshape(-1, v_prefix.shape[1], v_prefix.shape[2], v_prefix.shape[3])
        )
        k_windows = torch.cat((prefix_k_windows, k_windows), dim=-2)
        v_windows = torch.cat((prefix_v_windows, v_windows), dim=-2)
        patch_windows = F.scaled_dot_product_attention(
            q_windows,
            k_windows,
            v_windows,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        patch_out = self._reverse_3d_windows(
            patch_windows,
            batch_size=q.shape[0],
            spatial_shape=spatial_shape,
            window_shape=window_shape,
        )
        if has_shift:
            patch_out = torch.roll(patch_out, shifts=shift_shape, dims=(2, 3, 4))
        patch_out = patch_out.reshape(q.shape[0], q.shape[1], patch_tokens, q.shape[-1])
        return torch.cat((prefix_out, patch_out), dim=-2)

    def _project_tensor_parallel_output(self, x: torch.Tensor) -> torch.Tensor:
        channel_slice = self.local_channel_slice
        projected = F.linear(x, self.proj.weight[:, channel_slice], bias=None)
        projected = _reduce_from_tensor_parallel_region(projected, self.tensor_parallel_group)
        if self.proj.bias is not None:
            projected = projected + self.proj.bias
        return projected

    def _broadcast_parameter_slice(self, tensor: torch.Tensor, slice_spec, source_rank: int) -> None:
        if self.tensor_parallel_group is None or self.tensor_parallel_ranks is None:
            return
        view = tensor[slice_spec].contiguous()
        dist.broadcast(view, src=source_rank, group=self.tensor_parallel_group)
        tensor[slice_spec].copy_(view)

    def _sync_optimizer_state_slice(self, optimizer_state: dict, parameter: nn.Parameter, slice_spec, source_rank: int) -> None:
        state = optimizer_state.get(parameter)
        if not state:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                self._broadcast_parameter_slice(value, slice_spec, source_rank)

    def sync_tensor_parallel_parameters(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        if not self.is_tensor_parallel or self.tensor_parallel_ranks is None:
            return
        optimizer_state = optimizer.state if optimizer is not None else {}
        all_head_dim = self.proj.in_features
        local_width = self.local_num_heads * self.head_dim
        for owner_rank, source_rank in enumerate(self.tensor_parallel_ranks):
            start = owner_rank * local_width
            stop = start + local_width
            channel_slice = slice(start, stop)
            if self.qkv is not None:
                qkv_indices = (
                    list(range(start, stop))
                    + list(range(all_head_dim + start, all_head_dim + stop))
                    + list(range(2 * all_head_dim + start, 2 * all_head_dim + stop))
                )
                qkv_slice = torch.tensor(qkv_indices, device=self.qkv.weight.device, dtype=torch.long)
                weight_view = self.qkv.weight.index_select(0, qkv_slice).contiguous()
                dist.broadcast(weight_view, src=source_rank, group=self.tensor_parallel_group)
                self.qkv.weight.data.index_copy_(0, qkv_slice, weight_view)
                state = optimizer_state.get(self.qkv.weight)
                if state:
                    for value in state.values():
                        if torch.is_tensor(value) and value.shape == self.qkv.weight.shape:
                            state_view = value.index_select(0, qkv_slice).contiguous()
                            dist.broadcast(state_view, src=source_rank, group=self.tensor_parallel_group)
                            value.index_copy_(0, qkv_slice, state_view)
                for bias_parameter in (self.q_bias, self.v_bias):
                    if bias_parameter is not None:
                        self._broadcast_parameter_slice(bias_parameter.data, channel_slice, source_rank)
                        self._sync_optimizer_state_slice(optimizer_state, bias_parameter, channel_slice, source_rank)
            else:
                for projection in (self.q_proj, self.k_proj, self.v_proj):
                    self._broadcast_parameter_slice(projection.weight.data, (channel_slice, slice(None)), source_rank)
                    self._sync_optimizer_state_slice(
                        optimizer_state,
                        projection.weight,
                        (channel_slice, slice(None)),
                        source_rank,
                    )
                    if projection.bias is not None:
                        self._broadcast_parameter_slice(projection.bias.data, channel_slice, source_rank)
                        self._sync_optimizer_state_slice(optimizer_state, projection.bias, channel_slice, source_rank)

            self._broadcast_parameter_slice(self.proj.weight.data, (slice(None), channel_slice), source_rank)
            self._sync_optimizer_state_slice(
                optimizer_state,
                self.proj.weight,
                (slice(None), channel_slice),
                source_rank,
            )
    
    def forward(
            self,
            x,
            rope: Optional[RopeEmbedding] = None,
            attn_mask: Optional[torch.Tensor] = None,
            rope_shape: Optional[Tuple[int, ...]] = None,
    ):
        context_output_slice: slice | None
        x, context_output_slice = self._gather_context_parallel_input(x, rope_shape=rope_shape)
        B, N, C = x.shape
        
        if self.is_tensor_parallel:
            x = _copy_to_tensor_parallel_region(x, self.tensor_parallel_group)
            q, k, v = self._linear_qkv_tensor_parallel(x)
            local_head_slice = slice(
                self.tensor_parallel_rank * self.local_num_heads,
                (self.tensor_parallel_rank + 1) * self.local_num_heads,
            )
            rope = self._slice_rope(rope, local_head_slice)
        elif self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        
        if rope is not None:
            q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
            k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
        
        if self.attention_mode == "window_global_3d":
            if attn_mask is not None:
                raise ValueError("attn_mask is not supported with window_global_3d attention.")
            x = self._window_global_attention(q, k, v, rope_shape=rope_shape)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            raise RuntimeError("Fused attention should be used.")
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)
            
            attn = self.attn_drop(attn)
            x = attn @ v
        
        output_width = self.local_num_heads * self.head_dim if self.is_tensor_parallel else C
        x = x.transpose(1, 2).reshape(B, N, output_width)
        x = self.norm(x)
        if self.is_tensor_parallel:
            x = self._project_tensor_parallel_output(x)
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        if context_output_slice is not None:
            x = torch.cat((x[:, :self.num_prefix_tokens], x[:, context_output_slice]), dim=1)
        return x


class EvaBlock(nn.Module):
    
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
            drop_path_scale: bool = True,
            rope_impl=None,
            rope_kwargs=None,
            ndim: Optional[int] = None,
            attention_mode: str = "dense",
            window_size_patches: Optional[Tuple[int, int, int]] = None,
            shift_size_patches: Optional[Tuple[int, int, int]] = None,
            mlp_token_chunk_size: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
            attention_mode=attention_mode,
            window_size_patches=window_size_patches,
            shift_size_patches=shift_size_patches,
        )
        self.rope_embed = None
        if rope_impl is not None:
            if attn_head_dim is None and dim % num_heads != 0:
                raise ValueError(f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}")
            head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
            rope_kwargs_local = dict(rope_kwargs)
            self.rope_embed = rope_impl(
                head_dim,
                ndim=ndim,
                num_heads=num_heads,
                **rope_kwargs_local,
            )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = CompileStableDropPath(drop_path, drop_path_scale)
        
        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = CompileStableDropPath(drop_path, drop_path_scale)
        self.mlp_token_chunk_size = self._normalize_optional_positive_int(
            mlp_token_chunk_size,
            name="mlp_token_chunk_size",
        )

    @staticmethod
    def _normalize_optional_positive_int(value: Optional[int], *, name: str) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")
        return value

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = self.mlp_token_chunk_size
        if chunk_size is None or x.shape[1] <= chunk_size:
            return self.mlp(x)
        return torch.cat([self.mlp(chunk) for chunk in x.split(chunk_size, dim=1)], dim=1)
    
    def forward(
            self,
            x,
            rope: Optional[RopeEmbedding] = None,
            attn_mask: Optional[torch.Tensor] = None,
            rope_shape: Optional[Tuple[int, ...]] = None,
            rope_coords: Optional[RopeCoords] = None,
    ):
        if rope is None and self.rope_embed is not None:
            if rope_coords is not None:
                rope = self.rope_embed.get_embed_from_coords(rope_coords)
            else:
                if rope_shape is None:
                    raise ValueError("rope_shape must be provided when using per-block RoPE")
                rope = self.rope_embed.get_embed(rope_shape)
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, rope_shape=rope_shape))
            x = x + self.drop_path2(self._forward_mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, rope_shape=rope_shape))
            x = x + self.drop_path2(self.gamma_2 * self._forward_mlp(self.norm2(x)))
        return x

    def set_tensor_parallel(
        self,
        *,
        process_group=None,
        ranks: tuple[int, ...] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.attn.set_tensor_parallel(
            process_group=process_group,
            ranks=ranks,
            rank=rank,
            world_size=world_size,
        )

    def set_context_parallel(
        self,
        *,
        process_group=None,
        ranks: tuple[int, ...] | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.attn.set_context_parallel(
            process_group=process_group,
            ranks=ranks,
            rank=rank,
            world_size=world_size,
        )

    def sync_tensor_parallel_parameters(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        self.attn.sync_tensor_parallel_parameters(optimizer=optimizer)


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)


    """
    
    @staticmethod
    def _assert_patch_aligned(
            spatial_shape: Tuple[int, ...],
            patch_size: Tuple[int, ...],
            *,
            context: str,
    ) -> None:
        remainders = [int(size) % int(patch) for size, patch in zip(spatial_shape, patch_size)]
        if any(remainders):
            raise AssertionError(
                f"{context} must be divisible by patch_size for PatchEmbedDeeper, "
                f"got spatial_shape={tuple(spatial_shape)} and patch_size={tuple(patch_size)}"
            )

    @staticmethod
    def _normalize_optional_chunk_shape(
            value: Optional[int | Tuple[int, ...]],
            *,
            ndim: int,
            name: str,
    ) -> Optional[Tuple[int, ...]]:
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            return tuple([int(value)] * ndim)
        result = tuple(int(v) for v in value)
        if len(result) != ndim:
            raise ValueError(f"{name} must provide {ndim} values, got {result}")
        if any(v <= 0 for v in result):
            raise ValueError(f"{name} must be positive, got {result}")
        return result

    @staticmethod
    def _normalize_optional_nonnegative_shape(
            value: Optional[int | Tuple[int, ...]],
            *,
            ndim: int,
            name: str,
    ) -> Optional[Tuple[int, ...]]:
        if value is None:
            return None
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            return tuple([int(value)] * ndim)
        result = tuple(int(v) for v in value)
        if len(result) != ndim:
            raise ValueError(f"{name} must provide {ndim} values, got {result}")
        if any(v < 0 for v in result):
            raise ValueError(f"{name} must be non-negative, got {result}")
        return result

    @staticmethod
    def _normalize_optional_positive_int(
            value: Optional[int],
            *,
            name: str,
    ) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def __init__(
            self,
            input_channels: int = 1,
            global_crops_size: Tuple[int, ...] = None,
            local_crops_size: Tuple[int, ...] = None,
            embed_dim: int = 864,
            patch_size: Tuple[int, ...] = (8, 8, 8),
            embedding_type: str = "default",
            depth: int = 24,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = False,
            mlp_ratio: float = 4 * 2 / 3,
            swiglu_mlp: bool = True,
            scale_mlp: bool = True,
            scale_attn_inner: bool = False,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            # drops out things related to the projection. That is in the MLP and at the end of EVA attention
            attn_drop_rate: float = 0.,
            # drops attention, meaning connections between patches may bebroken up at random
            drop_path_rate: float = 0.,
            # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
            drop_path_uniform: bool = False,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = False,
            use_rot_pos_emb: bool = True,
            dynamic_img_size: bool = False,
            num_reg_tokens: int = 0,
            drop_path_scale: bool = True,
            rope_impl=RopePositionEmbedding,
            rope_kwargs=None,
            grad_checkpointing=False,
            deeper_embed_patch_chunk_size: Optional[int | Tuple[int, ...]] = None,
            deeper_embed_batch_chunk_size: Optional[int] = None,
            attention_mode: str = "dense",
            window_size_patches: Optional[Tuple[int, int, int]] = None,
            shift_size_patches: Optional[Tuple[int, int, int]] = None,
            alternate_window_shift: bool = True,
            mlp_token_chunk_size: Optional[int] = None,
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.patch_size = [patch_size] * 3 if isinstance(patch_size, int) else patch_size
        self.ndim = len(self.patch_size)
        self.embedding_type = str(embedding_type).lower()
        self.global_crops_size = [global_crops_size] * 3 if isinstance(global_crops_size, int) else global_crops_size
        self.local_crops_size = [local_crops_size] * 3 if isinstance(local_crops_size, int) else local_crops_size
        self.global_input_size = tuple(int(size) for size in self.global_crops_size)
        self.local_input_size = tuple(int(size) for size in self.local_crops_size)
        self.deeper_embed_patch_halo = tuple(0 for _ in range(self.ndim))
        self.deeper_embed_patch_chunk_size = self._normalize_optional_chunk_shape(
            deeper_embed_patch_chunk_size,
            ndim=self.ndim,
            name="deeper_embed_patch_chunk_size",
        )
        self.deeper_embed_batch_chunk_size = self._normalize_optional_positive_int(
            deeper_embed_batch_chunk_size,
            name="deeper_embed_batch_chunk_size",
        )
        self.mlp_token_chunk_size = self._normalize_optional_positive_int(
            mlp_token_chunk_size,
            name="mlp_token_chunk_size",
        )

        if self.embedding_type == "deeper":
            self._assert_patch_aligned(
                tuple(self.global_crops_size),
                tuple(self.patch_size),
                context="global_crops_size",
            )
            self._assert_patch_aligned(
                tuple(self.local_crops_size),
                tuple(self.patch_size),
                context="local_crops_size",
            )

        self.global_ref_feat_shape = tuple([i // ds for i, ds in zip(self.global_crops_size, self.patch_size)])
        self.local_ref_feat_shape = tuple([i // ds for i, ds in zip(self.local_crops_size, self.patch_size)])

        if self.embedding_type == "default":
            self.down_projection = PatchEmbed(
                patch_size=tuple(self.patch_size),
                input_channels=input_channels,
                embed_dim=embed_dim,
            )
        elif self.embedding_type == "deeper":
            self.down_projection = PatchEmbedDeeper(
                patch_size=tuple(self.patch_size),
                input_channels=input_channels,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(
                f"unsupported embedding_type={embedding_type!r}; expected 'default' or 'deeper'"
            )
        
        if rope_kwargs is None:
            rope_kwargs = {}
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = grad_checkpointing
        self.attention_mode = str(attention_mode).strip().lower()
        if self.attention_mode not in {"dense", "context_parallel_dense", "window_global_3d"}:
            raise ValueError(
                "attention_mode must be one of dense, context_parallel_dense, or window_global_3d; "
                f"got {attention_mode!r}"
            )
        self.window_size_patches = self._normalize_optional_chunk_shape(
            window_size_patches,
            ndim=self.ndim,
            name="window_size_patches",
        )
        self.shift_size_patches = self._normalize_optional_nonnegative_shape(
            shift_size_patches,
            ndim=self.ndim,
            name="shift_size_patches",
        )
        self.alternate_window_shift = bool(alternate_window_shift)
        if self.attention_mode == "window_global_3d" and self.window_size_patches is None:
            raise ValueError("window_global_3d attention requires window_size_patches.")
        if self.shift_size_patches is None:
            self.shift_size_patches = tuple(0 for _ in range(self.ndim))
        
        self.num_reg_tokens = num_reg_tokens
        self.num_class_tokens = (1 if class_token else 0)
        self.num_prefix_tokens = self.num_class_tokens + self.num_reg_tokens
        
        num_patches = np.prod(self.global_ref_feat_shape)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_class_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.use_per_block_rope = bool(use_rot_pos_emb and rope_impl is MixedRopePositionEmbedding)
        if use_rot_pos_emb and embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got embed_dim={embed_dim}, num_heads={num_heads}"
            )
        head_dim = embed_dim // num_heads
        if use_rot_pos_emb and head_dim % (2 * self.ndim) != 0:
            raise ValueError(
                f"RoPE requires head_dim divisible by 2 * ndim, got head_dim={head_dim}, ndim={self.ndim}"
            )
        if use_rot_pos_emb and not self.use_per_block_rope:
            self.rope_embed = rope_impl(
                head_dim,
                ndim=self.ndim,
                num_heads=num_heads,
                **rope_kwargs,
            )
        else:
            self.rope_embed = None
        
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        block_fn = EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                num_prefix_tokens=self.num_prefix_tokens,
                drop_path_scale=drop_path_scale,
                rope_impl=rope_impl if self.use_per_block_rope else None,
                rope_kwargs=rope_kwargs if self.use_per_block_rope else None,
                ndim=self.ndim,
                attention_mode=self.attention_mode,
                window_size_patches=self.window_size_patches,
                shift_size_patches=(
                    self.shift_size_patches
                    if self.attention_mode == "window_global_3d" and (not self.alternate_window_shift or i % 2 == 1)
                    else tuple(0 for _ in range(self.ndim))
                ),
                mlp_token_chunk_size=self.mlp_token_chunk_size,
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.context_parallel_size = 1
        self.context_parallel_rank = 0
        self.context_parallel_group = None
        self.context_parallel_ranks: tuple[int, ...] | None = None
        self._last_context_parallel_patch_start = 0
        self._last_context_parallel_patch_end = math.prod(self.global_ref_feat_shape)
        self._last_context_parallel_full_patch_tokens = math.prod(self.global_ref_feat_shape)
        
        self._init_weights()
    
    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(init_fn)
        self.down_projection.apply(InitWeights_He(1e-2))
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)
        if self.mask_token is not None:
            nn.init.zeros_(self.mask_token)
        
        # Inline fix_init_weight
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            if hasattr(layer.attn.proj, 'weight'):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp.fc2, 'weight'):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
    
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher
    
    def _pos_embed(self, x, *spatial) -> Tuple[torch.Tensor, Optional[RopeEmbedding]]:
        """
        Computes positional embeddings with interpolation if needed.

        Args:
            x (torch.Tensor): Input tensor after patch embedding, shape (B, N, C).
            spatial: Spatial dimensions of the original image before patch embedding.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Positionally encoded input.
        """
        pos_embed = self.pos_embed
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        source_size = tuple(self.global_ref_feat_shape)
        target_size = tuple(dim // patch for dim, patch in zip(spatial, self.patch_size))
        
        # If needed, interpolate only patch embeddings
        if source_size != target_size:
            if pos_embed is not None:
                pos_embed = self.interpolate_pos_encoding_nd(
                    pos_embed,
                    source_size=source_size,
                    target_size=target_size,
                    num_prefix_tokens=self.num_class_tokens
                )
        rot_pos_embed = self._get_rot_pos_embed(target_size)
        
        # Add interpolated positional embeddings
        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
            if self.cls_token is not None:
                x = torch.cat((x[:, :1], reg_tokens, x[:, 1:]), dim=1)
            else:
                x = torch.cat((reg_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        
        return x, rot_pos_embed

    def _get_rot_pos_embed(self, target_size: Tuple[int, ...]) -> Optional[RopeEmbedding]:
        if self.rope_embed is None:
            return None
        return self.rope_embed.get_embed(target_size)

    def _get_shared_per_block_rope_coords(self, target_size: Tuple[int, ...]) -> Optional[RopeCoords]:
        if not self.use_per_block_rope:
            return None
        for block in self.blocks:
            if hasattr(block, "rope_embed") and block.rope_embed is not None:
                return block.rope_embed.get_coords(target_size)
            if isinstance(block, nn.ModuleList):
                for inner_block in block:
                    if hasattr(inner_block, "rope_embed") and inner_block.rope_embed is not None:
                        return inner_block.rope_embed.get_coords(target_size)
        return None
    
    def interpolate_pos_encoding_nd(
            self,
            pos_embed: torch.Tensor,
            source_size: tuple,
            target_size: tuple,
            num_prefix_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Interpolates positional embeddings to match a new spatial size.

        Args:
            pos_embed (torch.Tensor): Positional embeddings (1, N, D).
            source_size: Original source grid size.
            target_size: New target grid size.
            num_prefix_tokens (int): Number of special tokens (e.g., CLS, registers).

        Returns:
            torch.Tensor: Rescaled positional embeddings (1, N_new, D).
        """
        _, N, C = pos_embed.shape
        N = N - num_prefix_tokens  # Remove prefix tokens

        previous_dtype = pos_embed.dtype
        pos_embed = pos_embed.float()

        if num_prefix_tokens > 0:
            pos_prefix, pos_embed = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
        else:
            pos_prefix = None

        ndim = len(source_size)
        if ndim not in (2, 3):
            raise ValueError(f"Only 2D and 3D positional interpolation are supported, got ndim={ndim}")

        interpolation_mode = "bilinear" if ndim == 2 else "trilinear"

        # Reshape from (1, N, C) -> (1, C, *source_size)
        pos_embed = pos_embed.reshape(1, *source_size, C)
        permute_order = (0, ndim + 1, *range(1, ndim + 1))
        pos_embed = pos_embed.permute(*permute_order)

        # Interpolate to the new spatial grid.
        pos_embed = F.interpolate(pos_embed, size=target_size, mode=interpolation_mode, align_corners=False)

        # Reshape back to (1, N, C)
        inverse_permute = (0, *range(2, ndim + 2), 1)
        pos_embed = pos_embed.permute(*inverse_permute).reshape(1, -1, C)

        # Reattach prefix tokens
        if pos_prefix is not None:
            pos_embed = torch.cat([pos_prefix, pos_embed], dim=1)

        pos_embed = pos_embed.to(previous_dtype)

        return pos_embed

    def resolve_output_spatial_shape(self, spatial: tuple[int, ...], *, view_kind: str) -> tuple[int, ...]:
        spatial = tuple(int(dim) for dim in spatial)
        if self.embedding_type != "deeper":
            return spatial
        if view_kind == "global":
            target = tuple(int(dim) for dim in self.global_crops_size)
            input_size = tuple(int(dim) for dim in self.global_input_size)
        elif view_kind == "local":
            target = tuple(int(dim) for dim in self.local_crops_size)
            input_size = tuple(int(dim) for dim in self.local_input_size)
        else:
            raise ValueError(f"unknown view_kind={view_kind!r}")

        if spatial == input_size or spatial == target:
            return target
        if self.embedding_type == "deeper":
            halo_voxels = tuple(int(tokens) * int(size) for tokens, size in zip(self.deeper_embed_patch_halo, self.patch_size))
            dynamic_target = tuple(int(dim) - 2 * int(halo) for dim, halo in zip(spatial, halo_voxels))
            if (
                any(halo_voxels)
                and all(dim > 0 for dim in dynamic_target)
                and all(dim % patch == 0 for dim, patch in zip(dynamic_target, self.patch_size))
            ):
                return dynamic_target
        raise ValueError(
            f"unexpected input shape {spatial} for embedding_type={self.embedding_type!r} and view_kind={view_kind!r}; "
            f"expected {input_size} or {target}"
        )

    def _resolve_target_spatial_shape(self, spatial: tuple[int, ...], *, view_kind: str) -> tuple[int, ...]:
        return self.resolve_output_spatial_shape(spatial, view_kind=view_kind)

    def _crop_embedded_grid(self, x: torch.Tensor, target_spatial: tuple[int, ...]) -> torch.Tensor:
        target_patch_shape = tuple(int(size) // int(patch) for size, patch in zip(target_spatial, self.patch_size))
        current_patch_shape = tuple(int(dim) for dim in x.shape[2:])
        if current_patch_shape == target_patch_shape:
            return x

        starts = []
        for current, target in zip(current_patch_shape, target_patch_shape):
            delta = current - target
            if delta < 0 or delta % 2 != 0:
                raise ValueError(
                    f"cannot center-crop embedded grid from {current_patch_shape} to {target_patch_shape}"
                )
            starts.append(delta // 2)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, target_patch_shape))
        return x[(slice(None), slice(None), *slices)]
    
    def prepare_tokens_with_masks(self, x, masks=None, *, view_kind: str = "global"):
        spatial = tuple(x.shape[2:])
        self._assert_patch_aligned(spatial, tuple(self.patch_size), context="input shape")
        if self.embedding_type == "deeper":
            target_spatial = self._resolve_target_spatial_shape(spatial, view_kind=view_kind)
            target_patch_shape = tuple(int(size) // int(patch) for size, patch in zip(target_spatial, self.patch_size))
            if (
                    self.deeper_embed_patch_chunk_size is not None
                    or self.deeper_embed_batch_chunk_size is not None
            ):
                x = self.down_projection.forward_tiled(
                    x,
                    target_patch_shape=target_patch_shape,
                    patch_chunk_size=self.deeper_embed_patch_chunk_size,
                    batch_chunk_size=self.deeper_embed_batch_chunk_size,
                )
            else:
                x = self.down_projection(x)
                x = self._crop_embedded_grid(x, target_spatial)
        else:
            x = self.down_projection(x)
            target_spatial = spatial
        if self.ndim == 2:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        else:
            x = rearrange(x, 'b c d h w -> b (d h w) c').contiguous()
        
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        
        target_patch_shape = tuple(dim // patch for dim, patch in zip(target_spatial, self.patch_size))
        x, rot_pos_embed = self._pos_embed(x, *target_spatial)
        
        return x, rot_pos_embed, target_patch_shape
    
    def forward_features_list(self, x_list, masks_list, *, view_kind: str = "global"):
        if not isinstance(x_list, list):
            return self.forward_features(x_list, masks_list, view_kind=view_kind)
        output = []
        for x, masks in zip(x_list, masks_list):
            x_out = self.forward_features(x, masks, view_kind=view_kind)
            output.append(x_out)
        return output
    
    def forward_features(self, x, masks=None, *, view_kind: str = "global"):
        x, rot_pos_embed, rope_shape = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind)
        if self.context_parallel_size > 1:
            start, end, n_patches = self.context_parallel_patch_range(rope_shape)
            self._last_context_parallel_patch_start = start
            self._last_context_parallel_patch_end = end
            self._last_context_parallel_full_patch_tokens = n_patches
            x = torch.cat((x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens + start:self.num_prefix_tokens + end]), dim=1)
        rope_coords = self._get_shared_per_block_rope_coords(rope_shape)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(
                    blk,
                    x,
                    rope=rot_pos_embed,
                    rope_shape=rope_shape,
                    rope_coords=rope_coords,
                    use_reentrant=False,
                )
            else:
                x = blk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)
        x = self.norm(x)
        outputs = {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
        return outputs
    
    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global"):
        return self.forward_features_list(x, masks, view_kind=view_kind)

    def set_tensor_parallel(
            self,
            *,
            process_group=None,
            ranks: tuple[int, ...] | None = None,
            rank: int = 0,
            world_size: int = 1,
    ) -> None:
        for block in self.blocks:
            set_tensor_parallel = getattr(_unwrap_checkpoint_module(block), "set_tensor_parallel", None)
            if callable(set_tensor_parallel):
                set_tensor_parallel(
                    process_group=process_group,
                    ranks=ranks,
                    rank=rank,
                    world_size=world_size,
                )

    def context_parallel_patch_range(self, target_size: Tuple[int, ...]) -> tuple[int, int, int]:
        n_patches = math.prod(int(dim) for dim in target_size)
        if self.context_parallel_size <= 1:
            return 0, n_patches, n_patches
        if n_patches % self.context_parallel_size != 0:
            raise ValueError(
                f"patch token count {n_patches} must be divisible by context_parallel_size="
                f"{self.context_parallel_size}."
            )
        local_tokens = n_patches // self.context_parallel_size
        start = self.context_parallel_rank * local_tokens
        return start, start + local_tokens, n_patches

    def set_context_parallel(
            self,
            *,
            process_group=None,
            ranks: tuple[int, ...] | None = None,
            rank: int = 0,
            world_size: int = 1,
    ) -> None:
        world_size = int(world_size)
        rank = int(rank)
        if world_size <= 0:
            raise ValueError(f"context parallel world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"context parallel rank must be in [0, {world_size}), got {rank}.")
        if world_size > 1 and self.attention_mode != "window_global_3d":
            raise ValueError("context parallel token sharding currently requires window_global_3d attention.")
        self.context_parallel_group = process_group
        self.context_parallel_ranks = ranks
        self.context_parallel_rank = rank
        self.context_parallel_size = world_size
        for block in self.blocks:
            set_context_parallel = getattr(_unwrap_checkpoint_module(block), "set_context_parallel", None)
            if callable(set_context_parallel):
                set_context_parallel(
                    process_group=process_group,
                    ranks=ranks,
                    rank=rank,
                    world_size=world_size,
                )

    def sync_tensor_parallel_parameters(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        for block in self.blocks:
            sync_tensor_parallel_parameters = getattr(
                _unwrap_checkpoint_module(block),
                "sync_tensor_parallel_parameters",
                None,
            )
            if callable(sync_tensor_parallel_parameters):
                sync_tensor_parallel_parameters(optimizer=optimizer)
    
    def load_pretrained_weights(self, state_dict, backbone_only=False, unchunk=False):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu", weights_only=False)['teacher']
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("backbone."):
                    continue
                new_key = k.replace("backbone.", "", 1)
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if unchunk:
            state_dict = self.unchunk_state_dict(state_dict)
        state_dict = dict(state_dict)
        if self.rope_embed is None:
            for key in [key for key in state_dict if key.startswith("rope_embed.")]:
                state_dict.pop(key)
        load_result = self.load_state_dict(state_dict, strict=False)

        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        allowed_missing_keys = set()

        if self.rope_embed is not None and hasattr(self.rope_embed, "reset_mixed_frequencies_to_axial"):
            allowed_missing_keys.add("rope_embed.mix_frequencies")
            if "rope_embed.mix_frequencies" in missing_keys:
                self.rope_embed.reset_mixed_frequencies_to_axial()

        for module_name, module in self.named_modules():
            if not module_name or not hasattr(module, "rope_embed") or module.rope_embed is None:
                continue
            prefix = f"{module_name}.rope_embed"
            if hasattr(module.rope_embed, "reset_mixed_frequencies_to_axial"):
                allowed_missing_keys.add(f"{prefix}.mix_frequencies")
                if f"{prefix}.mix_frequencies" in missing_keys:
                    module.rope_embed.reset_mixed_frequencies_to_axial()
            allowed_missing_keys.add(f"{prefix}.periods")

        disallowed_missing = [key for key in missing_keys if key not in allowed_missing_keys]
        if disallowed_missing or unexpected_keys:
            details = []
            if disallowed_missing:
                details.append(f"missing keys: {disallowed_missing}")
            if unexpected_keys:
                details.append(f"unexpected keys: {unexpected_keys}")
            raise RuntimeError("failed to load pretrained backbone weights cleanly: " + "; ".join(details))

        return load_result
    
    def unchunk_state_dict(self, state_dict):
        """
        Convert a state_dict from EvaWithChunking (nested blocks)
        to Eva (flat blocks).
        """
        if not any([key.startswith("blocks.0.0") for key in state_dict.keys()]):
            return state_dict
        
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith("blocks."):
                parts = key.split(".")
                # e.g. "blocks.0.1.attn.qkv.weight"
                # parts[1] = chunk idx, parts[2] = inner idx
                if parts[2].isdigit():
                    chunk_idx = int(parts[1])
                    inner_idx = int(parts[2])
                    # compute new flat index
                    flat_idx = chunk_idx * 9999 + inner_idx  # temporary large stride
                    # rewrite key
                    new_key = ".".join(["blocks", str(flat_idx)] + parts[3:])
                    new_state_dict[new_key] = val
                else:
                    # already a normal block key (no extra index)
                    new_state_dict[key] = val
            else:
                new_state_dict[key] = val
        
        # Fix flat indices back to consecutive 0..N
        # because above we used a stride
        mapping = {old: new for new, old in enumerate(sorted(set(
            int(k.split(".")[1]) for k in new_state_dict if k.startswith("blocks.")
        )))}
        final_state_dict = OrderedDict()
        for key, val in new_state_dict.items():
            if key.startswith("blocks."):
                parts = key.split(".")
                parts[1] = str(mapping[int(parts[1])])
                final_state_dict[".".join(parts)] = val
            else:
                final_state_dict[key] = val
        
        return final_state_dict


class BlockChunk(nn.ModuleList):
    def forward(self, x, rope=None, attn_mask=None, rope_shape=None, rope_coords=None):
        for blk in self:
            x = blk(x, rope=rope, attn_mask=attn_mask, rope_shape=rope_shape, rope_coords=rope_coords)
        return x

    def set_tensor_parallel(
            self,
            *,
            process_group=None,
            ranks: tuple[int, ...] | None = None,
            rank: int = 0,
            world_size: int = 1,
    ) -> None:
        for block in self:
            _unwrap_checkpoint_module(block).set_tensor_parallel(
                process_group=process_group,
                ranks=ranks,
                rank=rank,
                world_size=world_size,
            )

    def sync_tensor_parallel_parameters(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        for block in self:
            _unwrap_checkpoint_module(block).sync_tensor_parallel_parameters(optimizer=optimizer)

    def set_context_parallel(
            self,
            *,
            process_group=None,
            ranks: tuple[int, ...] | None = None,
            rank: int = 0,
            world_size: int = 1,
    ) -> None:
        for block in self:
            _unwrap_checkpoint_module(block).set_context_parallel(
                process_group=process_group,
                ranks=ranks,
                rank=rank,
                world_size=world_size,
            )


class EvaWithChunking(Eva):
    def __init__(self, *args, block_chunks: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.block_chunks = block_chunks
        self.chunked_blocks = block_chunks > 0 and block_chunks < len(self.blocks)
        
        if self.chunked_blocks:
            self._apply_block_chunking()
    
    def _apply_block_chunking(self):
        depth = len(self.blocks)
        chunksize = depth // self.block_chunks
        chunks = []
        for i in range(0, depth, chunksize):
            block_chunk = BlockChunk(self.blocks[i: i + chunksize])
            chunks.append(block_chunk)
        self.blocks = nn.ModuleList(chunks)
    
    def forward_features(self, x, masks=None, *, view_kind: str = "global"):
        x, rot_pos_embed, rope_shape = self.prepare_tokens_with_masks(x, masks, view_kind=view_kind)
        if self.context_parallel_size > 1:
            start, end, n_patches = self.context_parallel_patch_range(rope_shape)
            self._last_context_parallel_patch_start = start
            self._last_context_parallel_patch_end = end
            self._last_context_parallel_full_patch_tokens = n_patches
            x = torch.cat((x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens + start:self.num_prefix_tokens + end]), dim=1)
        rope_coords = self._get_shared_per_block_rope_coords(rope_shape)
        
        if self.chunked_blocks:
            for chunk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(
                        chunk,
                        x,
                        rope=rot_pos_embed,
                        rope_shape=rope_shape,
                        rope_coords=rope_coords,
                        use_reentrant=False,
                    )
                else:
                    x = chunk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)
        else:
            for blk in self.blocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(
                        blk,
                        x,
                        rope=rot_pos_embed,
                        rope_shape=rope_shape,
                        rope_coords=rope_coords,
                        use_reentrant=False,
                    )
                else:
                    x = blk(x, rope=rot_pos_embed, rope_shape=rope_shape, rope_coords=rope_coords)
        
        x = self.norm(x)
        outputs = {
            "x_norm_clstoken": x[:, 0] if self.num_class_tokens > 0 else None,
            "x_norm_regtokens": x[:, self.num_class_tokens:self.num_prefix_tokens],
            "x_norm_patchtokens": x[:, self.num_prefix_tokens:],
            "x_prenorm": x,
            "masks": masks,
        }
        return outputs
    
    def forward(self, x, masks=None, is_training=True, *, view_kind: str = "global"):
        return self.forward_features_list(x, masks, view_kind=view_kind)


class Dinov2PrimusEncL(Eva):
    def __init__(self,
                 input_channels,
                 input_shape):
        super().__init__(
            input_channels=input_channels,
            global_crops_size=96,
            local_crops_size=input_shape,
            embed_dim=864,
            patch_size=(8, 8, 8),
            depth=24,
            num_heads=16,
            mlp_ratio=2.66666666,
            attn_drop_rate=0.2,
            drop_path_rate=0.2
        )
