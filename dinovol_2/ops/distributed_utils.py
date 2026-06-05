from __future__ import annotations

import os
from typing import Any, Mapping

import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def _parallelism_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    parallelism = config.get("parallelism")
    if parallelism is None:
        return {}
    if not isinstance(parallelism, Mapping):
        raise ValueError("parallelism must be a mapping when provided.")
    return parallelism


def resolve_distributed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = bool(config.get("use_ddp", False)) or env_world_size > 1
    parallelism = _parallelism_config(config)
    strategy = str(parallelism.get("strategy", "ddp")).strip().lower()
    if strategy not in {"ddp", "tp_ddp"}:
        raise ValueError(f"unsupported parallelism.strategy={strategy!r}; expected 'ddp' or 'tp_ddp'.")

    tensor_parallel_size = int(parallelism.get("tensor_parallel_size", 1))
    if strategy == "ddp" and tensor_parallel_size != 1:
        raise ValueError("parallelism.tensor_parallel_size must be 1 when strategy is 'ddp'.")
    if tensor_parallel_size <= 0:
        raise ValueError(f"tensor_parallel_size must be positive, got {tensor_parallel_size}.")
    if not use_ddp and tensor_parallel_size != 1:
        raise ValueError("tensor_parallel_size > 1 requires torchrun/DDP.")
    if use_ddp and env_world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"WORLD_SIZE={env_world_size} must be divisible by tensor_parallel_size={tensor_parallel_size}."
        )

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if use_ddp and tensor_parallel_size > 1:
        if tensor_parallel_size > local_world_size:
            raise ValueError(
                f"tensor_parallel_size={tensor_parallel_size} must not exceed LOCAL_WORLD_SIZE={local_world_size}; "
                "this implementation keeps tensor-parallel groups node-local."
            )
        if local_world_size % tensor_parallel_size != 0:
            raise ValueError(
                f"LOCAL_WORLD_SIZE={local_world_size} must be divisible by tensor_parallel_size={tensor_parallel_size}."
            )

    rank = int(os.environ.get("RANK", "0")) if use_ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) if use_ddp else 0
    data_parallel_world_size = env_world_size // tensor_parallel_size if use_ddp else 1
    tensor_parallel_rank = rank % tensor_parallel_size if use_ddp else 0
    tensor_parallel_group_index = rank // tensor_parallel_size if use_ddp else 0
    return {
        "use_ddp": use_ddp,
        "world_size": env_world_size if use_ddp else 1,
        "rank": rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size if use_ddp else 1,
        "strategy": strategy,
        "tensor_parallel_size": tensor_parallel_size,
        "tensor_parallel_rank": tensor_parallel_rank,
        "tensor_parallel_group_index": tensor_parallel_group_index,
        "data_parallel_world_size": data_parallel_world_size,
        "data_parallel_rank": tensor_parallel_group_index,
    }


def build_parallel_process_groups(
    *,
    is_distributed: bool,
    world_size: int,
    rank: int,
    tensor_parallel_size: int,
) -> dict[str, Any]:
    if not is_distributed or tensor_parallel_size == 1:
        return {
            "tensor_parallel_group": None,
            "tensor_parallel_ranks": (rank,),
            "data_parallel_group": None,
            "data_parallel_ranks": tuple(range(world_size)) if is_distributed else (0,),
        }

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before building process groups.")
    if world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tensor_parallel_size={tensor_parallel_size}."
        )

    tensor_parallel_group = None
    tensor_parallel_ranks: tuple[int, ...] | None = None
    data_parallel_group = None
    data_parallel_ranks: tuple[int, ...] | None = None
    n_tensor_groups = world_size // tensor_parallel_size

    for group_index in range(n_tensor_groups):
        ranks = tuple(range(group_index * tensor_parallel_size, (group_index + 1) * tensor_parallel_size))
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            tensor_parallel_group = group
            tensor_parallel_ranks = ranks

    for tensor_rank in range(tensor_parallel_size):
        ranks = tuple(tensor_rank + group_index * tensor_parallel_size for group_index in range(n_tensor_groups))
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            data_parallel_group = group
            data_parallel_ranks = ranks

    if tensor_parallel_group is None or tensor_parallel_ranks is None:
        raise RuntimeError(f"rank {rank} was not assigned to a tensor-parallel group.")
    if data_parallel_group is None or data_parallel_ranks is None:
        raise RuntimeError(f"rank {rank} was not assigned to a data-parallel group.")

    return {
        "tensor_parallel_group": tensor_parallel_group,
        "tensor_parallel_ranks": tensor_parallel_ranks,
        "data_parallel_group": data_parallel_group,
        "data_parallel_ranks": data_parallel_ranks,
    }


def build_distributed_sampler(
    dataset: Dataset[Any],
    *,
    is_distributed: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> DistributedSampler[Any] | None:
    if not is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )
