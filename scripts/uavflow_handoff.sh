#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${UAVFLOW_ENV_FILE:-${ROOT}/server.env}"
ACTION="${1:-help}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}" >&2
  echo "Run: cp server.env.example server.env, then edit server.env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

: "${CONDA_ENV:=uav-gam}"
: "${DATA_ROOT:?Set DATA_ROOT in server.env}"
: "${MODEL_ROOT:?Set MODEL_ROOT in server.env}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT in server.env}"
: "${SCHEDULER:=slurm}"

export PROJECT_ROOT="${ROOT}"
export UAVFLOW_SIM_ROOT="${DATA_ROOT}/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="${DATA_ROOT}/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="${DA3_CHECKPOINT:-${ROOT}/checkpoints/track4world_da3.pth}"
export QWEN_MODEL="${QWEN_MODEL:-${MODEL_ROOT}/qwen3.5-2b}"
export T5_MODEL="${T5_MODEL:-${MODEL_ROOT}/t5-base}"

conda_exe() {
  command -v conda >/dev/null || {
    echo "conda is not on PATH. Load Miniconda/Anaconda first." >&2
    exit 2
  }
  command -v conda
}

refresh_python_paths() {
  local base
  base="$($(conda_exe) info --base)"
  export PYTHON_BIN="${base}/envs/${CONDA_ENV}/bin/python"
  export TORCHRUN_BIN="${base}/envs/${CONDA_ENV}/bin/torchrun"
}

setup_env() {
  if ! "$(conda_exe)" env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    "$(conda_exe)" env create -n "${CONDA_ENV}" -f "${ROOT}/environment-uavflow.yml"
  else
    echo "Conda environment ${CONDA_ENV} already exists; keeping it."
  fi
  "$(conda_exe)" run -n "${CONDA_ENV}" \
    bash "${ROOT}/scripts/setup_uavflow_remote.sh"
  refresh_python_paths
}

download_assets() {
  refresh_python_paths
  "$(conda_exe)" run -n "${CONDA_ENV}" \
    bash "${ROOT}/scripts/download_uavflow_assets.sh"
}

audit_assets() {
  refresh_python_paths
  "${PYTHON_BIN}" "${ROOT}/scripts/verify_uavflow_remote.py" \
    --sim-root "${UAVFLOW_SIM_ROOT}" \
    --depth-root "${UAVFLOW_DEPTH_ROOT}" \
    --da3-checkpoint "${DA3_CHECKPOINT}" \
    --qwen-model "${QWEN_MODEL}" \
    --t5-model "${T5_MODEL}"
}

smoke_test() {
  refresh_python_paths
  CUDA_DEVICES=0,1 NPROC=2 GLOBAL_BATCH_SIZE=4 BATCH_SIZE=2 \
    MAX_TRAJECTORIES=20 STAGE1_EPOCHS=1 RUN_STAGE2=0 RUN_IDS=B0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}/_smoke" \
    bash "${ROOT}/experiments/uavflow_remote_ablation/run_remote.sh"
}

submit_matrix() {
  refresh_python_paths
  mkdir -p "${OUTPUT_ROOT}"
  case "${SCHEDULER}" in
    slurm)
      command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 2; }
      sbatch "${ROOT}/experiments/uavflow_remote_ablation/submit_slurm_compact.sh"
      ;;
    pbs)
      command -v qsub >/dev/null || { echo "qsub not found" >&2; exit 2; }
      qsub -V "${ROOT}/experiments/uavflow_remote_ablation/submit_pbs_compact.sh"
      ;;
    lsf)
      command -v bsub >/dev/null || { echo "bsub not found" >&2; exit 2; }
      bsub < "${ROOT}/experiments/uavflow_remote_ablation/submit_lsf_compact.sh"
      ;;
    local)
      export GPU_GROUPS="${GPU_GROUPS:-0,1,2,3}"
      bash "${ROOT}/experiments/uavflow_remote_ablation/run_local_gpu_pool.sh"
      ;;
    *)
      echo "Unknown SCHEDULER=${SCHEDULER}; use slurm, pbs, lsf or local" >&2
      exit 2
      ;;
  esac
}

show_status() {
  refresh_python_paths
  "${PYTHON_BIN}" "${ROOT}/experiments/uavflow_remote_ablation/status.py" \
    "${OUTPUT_ROOT}"
}

case "${ACTION}" in
  setup) setup_env ;;
  download) download_assets ;;
  audit) audit_assets ;;
  smoke) smoke_test ;;
  submit) audit_assets; submit_matrix ;;
  status) show_status ;;
  prepare) setup_env; download_assets; audit_assets; smoke_test ;;
  all) setup_env; download_assets; audit_assets; smoke_test; submit_matrix ;;
  *)
    cat <<'EOF'
Usage: bash scripts/uavflow_handoff.sh ACTION

Actions:
  setup     Create/update the environment
  download  Download official data, depth sidecars and frozen models
  audit     Verify all required files and episode counts
  smoke     Run a tiny two-GPU installation test
  prepare   setup + download + audit + smoke
  submit    audit, then submit the 21-cell production matrix
  all       prepare + submit
  status    Show experiment progress
EOF
    ;;
esac
