#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES to the number of nodes.}"
: "${NODE_RANK:?Set NODE_RANK to this node zero-based rank.}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous host or IP.}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-dinovol-pretrain}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_CONF="${RDZV_CONF:-}"
MODULE="${MODULE:-dinovol_2.pretrain}"
CONFIG_ARGS=()
if [[ -n "${CONFIG:-}" ]]; then
  CONFIG_ARGS=("${CONFIG}")
elif [[ "${CONFIG_REQUIRED:-1}" == "1" ]]; then
  echo "Set CONFIG to a JSON config path, or set CONFIG_REQUIRED=0 for configless modules." >&2
  exit 2
fi
LOCAL_ADDR_ARGS=()
if [[ -n "${LOCAL_ADDR:-}" ]]; then
  LOCAL_ADDR_ARGS=(--local-addr="${LOCAL_ADDR}")
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was not found on PATH or at ~/.local/bin/uv; set UV_BIN." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_IB_ADDR_FAMILY="${NCCL_IB_ADDR_FAMILY:-AF_INET}"
export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
if [[ -n "${TORCH_COMPILE_CACHE_DIR:-}" ]]; then
  export TORCHINDUCTOR_CACHE_DIR="${TORCH_COMPILE_CACHE_DIR}"
fi
if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
  export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
  export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
fi
if [[ -n "${TORCH_COMPILE_LOGS:-}" ]]; then
  export TORCH_LOGS="${TORCH_COMPILE_LOGS}"
fi
if [[ -n "${TORCH_COMPILE_TRACE_DIR:-}" ]]; then
  export TORCH_TRACE="${TORCH_COMPILE_TRACE_DIR}"
fi

if [[ "${REQUIRE_NCCL_RDMA_ENV:-0}" == "1" && "${NCCL_IB_DISABLE:-0}" != "1" ]]; then
  : "${NCCL_IB_HCA:?Set NCCL_IB_HCA for RDMA runs or set NCCL_IB_DISABLE=1 for TCP fallback.}"
  : "${NCCL_IB_GID_INDEX:?Set NCCL_IB_GID_INDEX for RDMA/RoCE runs.}"
fi

if [[ -z "${LOCAL_ADDR:-}" ]]; then
  echo "warning: LOCAL_ADDR is not set; torchrun may advertise a hostname that peer nodes cannot route." >&2
fi

TORCHRUN_ARGS=(
  --nnodes="${NNODES}"
  --nproc-per-node="${NPROC_PER_NODE}"
  --node-rank="${NODE_RANK}"
)
if [[ "${RDZV_BACKEND}" == "static" ]]; then
  TORCHRUN_ARGS+=(--master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}")
else
  TORCHRUN_ARGS+=(--rdzv-backend="${RDZV_BACKEND}" --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" --rdzv-id="${RDZV_ID}")
  if [[ -n "${RDZV_CONF}" ]]; then
    TORCHRUN_ARGS+=(--rdzv-conf="${RDZV_CONF}")
  fi
fi
if [[ -n "${TORCHRUN_NUMA_BINDING:-}" ]]; then
  TORCHRUN_ARGS+=(--numa-binding="${TORCHRUN_NUMA_BINDING}")
fi

"${UV_BIN}" run python -m torch.distributed.run \
  "${TORCHRUN_ARGS[@]}" \
  "${LOCAL_ADDR_ARGS[@]}" \
  -m "${MODULE}" "${CONFIG_ARGS[@]}" "$@"
