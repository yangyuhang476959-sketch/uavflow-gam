#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT="${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_env.sh"

export NPROC=1
export GLOBAL_BATCH_SIZE=4
export DEVICE_IDS="${DEVICE_IDS:-0}"
export STAGE1_EPOCHS=0
export UAVFLOW_BENCH_START_STEP=11
export UAVFLOW_BENCH_END_STEP=50
export UAVFLOW_SKIP_FINAL_CHECKPOINT=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/bench_r1_ascend}"

python -m experiments.uavflow_remote_ablation.run_experiment R1 \
  --stage stage1 \
  --set model.gradient_checkpointing=false \
  --set training.amp=true \
  --set training.amp_dtype=bf16 \
  --set training.eval_every=0 \
  --set training.save_every=0 \
  --set training.save_latest_every=0 \
  --set training.save_every_epochs=0
