#!/usr/bin/env bash
# Original local evaluation runner; no policy startup or process-name killing.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UAVFLOW_EVAL_ROOT:?Set UAVFLOW_EVAL_ROOT to UAV-Flow/UAV-Flow-Eval}"
: "${OUT:?Set OUT to an evaluation output directory}"
TASK_ROOT="${TASK_ROOT:-${UAVFLOW_EVAL_ROOT}/test_jsons}"
SIM_PYTHON="${SIM_PYTHON:-python}"
test -f "${UAVFLOW_EVAL_ROOT}/relative.py"
test -d "${TASK_ROOT}"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
export PYTHONPATH="${UAVFLOW_EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
# Vulkan adapter numbering is independent of CUDA_VISIBLE_DEVICES.
# Supply the physical Vulkan adapter index explicitly; do not mask CUDA here.
cd "${OUT}"
exec "${SIM_PYTHON}" -u "${ROOT}/simulation/batch_run_act_all_debug.py" \
  --json_folder "${TASK_ROOT}" \
  --server_port "${PORT:-5007}" \
  --instruction_type instruction_unified \
  --use_agent_camera --offscreen --gpu_id "${UE_ADAPTER:-0}" \
  --sleep_time "${UE_START_WAIT:-90}" --max_steps 100 \
  --reset_drone_motion_per_task --verify_target_per_task \
  --target_placement_retries 5 \
  --images_dir "${OUT}/results" --debug_obs_dir "${OUT}/obs" \
  --debug_obs_every 1 --request_timeout 120 --log_level INFO "$@"
