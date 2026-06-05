#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname

echo "== key environment =="
env | sort | grep -E '^(MASTER_|WORLD_|RANK|LOCAL_|NCCL_|GLOO_|TORCH_NCCL_)' || true

echo "== interfaces =="
ip -br addr || true

echo "== routes =="
ip route || true
if [[ -n "${FABRIC_ROUTE_TARGETS:-}" ]]; then
  echo "== route probes =="
  for target in ${FABRIC_ROUTE_TARGETS}; do
    ip route get "${target}" || true
  done
fi

echo "== rdma links =="
rdma link show 2>/dev/null || true

echo "== hcas =="
ibv_devinfo -l 2>/dev/null || true

echo "== gids =="
show_gids 2>/dev/null | head -200 || true

echo "== hca port states =="
for hca in /sys/class/infiniband/*; do
  [[ -e "${hca}" ]] || continue
  printf 'HCA %s\n' "$(basename "${hca}")"
  for port in "${hca}"/ports/*; do
    [[ -e "${port}" ]] || continue
    printf '  port %s state=' "$(basename "${port}")"
    cat "${port}/state" 2>/dev/null | tr -d '\n' || true
    printf ' phys='
    cat "${port}/phys_state" 2>/dev/null | tr -d '\n' || true
    printf ' rate='
    cat "${port}/rate" 2>/dev/null | tr -d '\n' || true
    printf '\n'
  done
done

echo "== gpu topology =="
nvidia-smi topo -m 2>/dev/null || true

echo "== torch/nccl =="
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -n "${UV_BIN}" ]]; then
  "${UV_BIN}" run python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("nccl", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
PY
else
  echo "uv not found; skipping torch/NCCL Python probe"
fi

echo "== nccl-tests =="
if command -v all_reduce_perf >/dev/null 2>&1; then
  all_reduce_perf -h >/dev/null 2>&1 && echo "all_reduce_perf usable" || echo "all_reduce_perf present but not usable"
else
  echo "all_reduce_perf not found"
fi
