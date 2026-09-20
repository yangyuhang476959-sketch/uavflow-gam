#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${ROOT}/experiments/uavflow_remote_ablation/compact_cells.sh"

# Semicolon separates independent four-GPU workers. Examples:
#   GPU_GROUPS='0,1,2,3'                         -> one sequential worker
#   GPU_GROUPS='0,1,2,3;4,5,6,7'               -> two parallel workers
#   GPU_GROUPS='0,1,2,3;4,5,6,7;8,9,10,11;...' -> as many as available
IFS=';' read -r -a GROUPS <<< "${GPU_GROUPS:-0,1,2,3}"
WORKERS="${#GROUPS[@]}"
mkdir -p "${OUTPUT_ROOT:-${ROOT}/results/remote_ablation}/launcher_logs"

echo "compact_cells=${#COMPACT_RUN_IDS[@]} gpu_workers=${WORKERS}"
for ((worker=0; worker<WORKERS; worker++)); do
  (
    for ((cell=worker; cell<${#COMPACT_RUN_IDS[@]}; cell+=WORKERS)); do
      run_id="${COMPACT_RUN_IDS[cell]}"
      log="${OUTPUT_ROOT:-${ROOT}/results/remote_ablation}/launcher_logs/${run_id}.log"
      echo "[$(date -Is)] worker=${worker} devices=${GROUPS[worker]} cell=${cell} id=${run_id}" | tee -a "${log}"
      CUDA_DEVICES="${GROUPS[worker]}" \
        bash "${ROOT}/experiments/uavflow_remote_ablation/run_compact_cell.sh" "${cell}" \
        2>&1 | tee -a "${log}"
    done
  ) &
done
wait

