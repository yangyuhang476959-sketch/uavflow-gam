#!/usr/bin/env bash
set -euo pipefail

# One-command server entry point. Read the same optional server.env used by the
# handoff script and otherwise use repository-relative defaults.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${UAVFLOW_ENV_FILE:-${ROOT}/server.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

export PROJECT_ROOT="${ROOT}"
export DATA_ROOT="${DATA_ROOT:-${ROOT}/data_remote}"
export MODEL_ROOT="${MODEL_ROOT:-${ROOT}/checkpoints}"
export UAVFLOW_SIM_ROOT="${UAVFLOW_SIM_ROOT:-${DATA_ROOT}/UAV-Flow-Sim}"
export UAVFLOW_DEPTH_ROOT="${UAVFLOW_DEPTH_ROOT:-${DATA_ROOT}/UAV-Flow-Sim-Depth}"
export DA3_CHECKPOINT="${DA3_CHECKPOINT:-${ROOT}/checkpoints/track4world_da3.pth}"
export QWEN_MODEL="${QWEN_MODEL:-${MODEL_ROOT}/qwen3.5-2b}"
export T5_MODEL="${T5_MODEL:-${MODEL_ROOT}/t5-base}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export NPROC="${NPROC:-4}"
# Use a stable compromise between OpenVLA-UAV (global 32) and GAM post-train
# (global 12): this is also GAM's released pre-training batch. Four ranks keep
# enough work per 64-GiB GPU while allowing all 21 compact cells to occupy only
# 84 GPUs in one wave.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
if [[ -n "${BATCH_SIZE:-}" ]]; then
  if (( BATCH_SIZE * NPROC != GLOBAL_BATCH_SIZE )); then
    echo "BATCH_SIZE(${BATCH_SIZE}) * NPROC(${NPROC}) != GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE})" >&2
    exit 2
  fi
else
  if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
    echo "GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) must be divisible by NPROC(${NPROC})" >&2
    exit 2
  fi
  export BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC))
fi
export NUM_WORKERS="${NUM_WORKERS:-4}"
export STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
export RUN_STAGE2="${RUN_STAGE2:-1}"
export RUN_IDS="${RUN_IDS:-B0,S1COS,P1,L1,D1,D2,D1LOG,W3,W5,W10,H0,HB,F3,F7,F10,M1,C1_PL,C2_D2HB,C3_W3HB,C4_F3W3,C5_F10HB}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/remote_ablation}"

# Stage 1 defaults to the earlier UAV action runs' constant schedule. Stage 2
# is always cosine by default, matching the earlier CLIP Stop runs. Set
# USE_GAM_SCHEDULE=1 to switch Stage 1 to released GAM's schedule.
if [[ "${USE_GAM_SCHEDULE:-0}" == "1" ]]; then
  export STAGE1_LR_SCHEDULE=cosine
  export STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-500}"
  export STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.01}"
else
  export STAGE1_LR_SCHEDULE="${STAGE1_LR_SCHEDULE:-constant}"
  export STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-0}"
  export STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-0.05}"
fi
export STAGE2_LR_SCHEDULE="${STAGE2_LR_SCHEDULE:-cosine}"
export STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
export STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
export BASE_LR="${BASE_LR:-5.0e-5}"

mkdir -p "${OUTPUT_ROOT}"
exec bash "${ROOT}/experiments/uavflow_remote_ablation/run_remote.sh"
