#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${ROOT}/experiments/uavflow_remote_ablation/compact_cells.sh"

INDEX="${1:?Usage: run_compact_cell.sh ZERO_BASED_CELL_INDEX}"
if ! [[ "${INDEX}" =~ ^[0-9]+$ ]] || (( INDEX >= ${#COMPACT_RUN_IDS[@]} )); then
  echo "Invalid compact cell index ${INDEX}; expected 0..$((${#COMPACT_RUN_IDS[@]} - 1))" >&2
  exit 2
fi

export RUN_IDS="${COMPACT_RUN_IDS[INDEX]}"
export NPROC="${NPROC:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

echo "compact_cell=${INDEX} run_id=${RUN_IDS} devices=${CUDA_DEVICES} nproc=${NPROC} batch=${BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_server.sh"

