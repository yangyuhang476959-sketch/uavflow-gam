#!/usr/bin/env bash
#SBATCH --job-name=uavflow-gam
#SBATCH --array=0-20%21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=uavflow-gam-%A_%a.out
#SBATCH --error=uavflow-gam-%A_%a.err
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_compact_cell.sh" \
  "${SLURM_ARRAY_TASK_ID}"

