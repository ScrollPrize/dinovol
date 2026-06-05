#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?Set CONFIG to a JSON config path.}"
: "${NNODES:?Set NNODES to the number of nodes.}"
: "${NODE_RANK:?Set NODE_RANK to this node zero-based rank.}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous host or IP.}"

MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-dinovol-pretrain}"
MODULE="${MODULE:-dinovol_2.pretrain}"
LOCAL_ADDR_ARGS=()
if [[ -n "${LOCAL_ADDR:-}" ]]; then
  LOCAL_ADDR_ARGS=(--local-addr="${LOCAL_ADDR}")
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

uv run python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${RDZV_ID}" \
  "${LOCAL_ADDR_ARGS[@]}" \
  -m "${MODULE}" "${CONFIG}" "$@"
