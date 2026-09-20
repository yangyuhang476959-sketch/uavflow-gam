#!/usr/bin/env bash
#PBS -N uavflow-gam
#PBS -J 0-20
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_compact_cell.sh" \
  "${PBS_ARRAY_INDEX}"

