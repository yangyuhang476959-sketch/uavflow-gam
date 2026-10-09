#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT="${ROOT}"
# Use the engineer's already activated Python/CANN environment (NPU_SETUP.md).
# Do not activate the historical .ascend/env or change the selected CANN stack.
cd "${ROOT}"
export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
export TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${ROOT}/.ascend/triton}"
export FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${ROOT}/.ascend/flash-linear-attention}"
export PYTHONPATH="${TRITON_ASCEND_TARGET}:${FLA_ASCEND_DIR}:${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# A benchmark must not inherit profiling or experimental fusion flags from an
# interactive shell. Keep only the validated FLA full-sequence path.
unset UAVFLOW_PROFILE_NPU UAVFLOW_PROFILE_START_STEP UAVFLOW_PROFILE_STEPS UAVFLOW_PROFILE_DIR
unset UAVFLOW_QWEN_NPU_RMSNORM UAVFLOW_QWEN_FUSED_GDN_PROJ UAVFLOW_NPU_FUSED_OPTIMIZER ASCEND_UB_CAPACITY_BITS
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_QWEN_BENCHMARK_MODE=normal
export UAVFLOW_GRAD_SCALER=auto

export NPROC=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export DEVICE_IDS="${DEVICE_IDS:-0}"
export STAGE1_EPOCHS=0
export UAVFLOW_BENCH_START_STEP=11
export UAVFLOW_BENCH_END_STEP=50
export UAVFLOW_SKIP_FINAL_CHECKPOINT=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/bench_r1_ascend}"

python -m experiments.uavflow_remote_ablation.run_experiment R1 \
  --stage stage1 \
  --max-trajectories "${MAX_TRAJECTORIES:-20}" \
  --set model.gradient_checkpointing=false \
  --set training.amp=true \
  --set training.amp_dtype=bf16 \
  --set training.eval_every=0 \
  --set training.save_every=0 \
  --set training.save_latest_every=0 \
  --set training.save_every_epochs=0
