#!/usr/bin/env bash
# Source this file before an Ascend training/benchmark command.
set -e

_UAVFLOW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_UAVFLOW_SCRIPT_DIR}/.." && pwd)}"
export ASCEND_VENV="${ASCEND_VENV:-${REPO_ROOT}/.venv-ascend}"
export TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${REPO_ROOT}/.ascend/triton}"
export FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${REPO_ROOT}/.ascend/flash-linear-attention}"

if [[ ! -x "${ASCEND_VENV}/bin/python" ]]; then
  echo "Missing Ascend venv: ${ASCEND_VENV}; run scripts/setup_ascend_cluster.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${ASCEND_VENV}/bin/activate"

if [[ -n "${CANN_ROOT:-}" && -f "${CANN_ROOT}/set_env.sh" ]]; then
  _UAVFLOW_CANN_ENV="${CANN_ROOT}/set_env.sh"
elif [[ -n "${CANN_ROOT:-}" && -f "${CANN_ROOT}/bin/set_env.sh" ]]; then
  _UAVFLOW_CANN_ENV="${CANN_ROOT}/bin/set_env.sh"
else
  _UAVFLOW_CANN_ENV="$(find /usr/local/Ascend -maxdepth 3 -type f -name set_env.sh 2>/dev/null | sort | head -n1)"
fi
if [[ -z "${_UAVFLOW_CANN_ENV:-}" || ! -f "${_UAVFLOW_CANN_ENV}" ]]; then
  echo "CANN set_env.sh not found; export CANN_ROOT first" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${_UAVFLOW_CANN_ENV}"
echo "Using CANN environment: ${_UAVFLOW_CANN_ENV}"

export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
export PYTHONPATH="${TRITON_ASCEND_TARGET}:${FLA_ASCEND_DIR}:${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Intentionally do not enable fused AdamW/RMSNorm/projection fusion or set
# ASCEND_UB_CAPACITY_BITS: all were neutral or slower in end-to-end R1 tests.
unset ASCEND_UB_CAPACITY_BITS
