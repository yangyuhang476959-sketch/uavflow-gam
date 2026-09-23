#!/usr/bin/env bash
set -Eeuo pipefail

# One cloud allocation = one experiment = eight DDP ranks.
# Keep the designed global batch at 24, so moving from four to eight GPUs does
# not alter the optimization protocol: each rank receives batch size three.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_ID="${1:?Usage: run_one_8gpu.sh RUN_ID}"

export PROJECT_ROOT="${ROOT}"
export NPROC=8
export GLOBAL_BATCH_SIZE=24
export BATCH_SIZE=3
export CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export RUN_IDS="${RUN_ID}"
export RUN_STAGE2=1

echo "cloud_job run_id=${RUN_ID} nproc=${NPROC} per_gpu_batch=${BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_server.sh"
