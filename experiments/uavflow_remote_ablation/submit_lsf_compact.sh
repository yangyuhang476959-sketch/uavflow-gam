#!/usr/bin/env bash
#BSUB -J "uavflow-gam[1-21]"
#BSUB -n 32
#BSUB -gpu "num=4"
#BSUB -oo "uavflow-gam-%J_%I.out"
#BSUB -eo "uavflow-gam-%J_%I.err"
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_compact_cell.sh" \
  "$((LSB_JOBINDEX - 1))"

