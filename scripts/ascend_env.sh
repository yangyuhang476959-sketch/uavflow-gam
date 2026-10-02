#!/usr/bin/env bash
# Source this file before an Ascend training/benchmark command.
set -e

_UAVFLOW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_UAVFLOW_SCRIPT_DIR}/.." && pwd)}"
_UAVFLOW_RUNTIME_ENV="${ASCEND_RUNTIME_ENV:-${REPO_ROOT}/.ascend/runtime.env}"
if [[ -f "${_UAVFLOW_RUNTIME_ENV}" ]]; then
  # This generated file contains paths only, never credentials.
  # shellcheck disable=SC1090
  source "${_UAVFLOW_RUNTIME_ENV}"
fi
export ASCEND_ENV_ROOT="${ASCEND_ENV_ROOT:-${ASCEND_VENV:-${REPO_ROOT}/.ascend/env}}"
export ASCEND_VENV="${ASCEND_VENV:-${ASCEND_ENV_ROOT}}"
export TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${REPO_ROOT}/.ascend/triton}"
export FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${REPO_ROOT}/.ascend/flash-linear-attention}"
export ASCEND_SEARCH_ROOT="${ASCEND_SEARCH_ROOT:-${REPO_ROOT}/.ascend/cann}"

if [[ ! -x "${ASCEND_VENV}/bin/python" ]]; then
  echo "Missing Ascend venv: ${ASCEND_VENV}; run scripts/setup_ascend_cluster.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "${ASCEND_VENV}/bin/activate"

# shellcheck disable=SC1091
source "${_UAVFLOW_SCRIPT_DIR}/ascend_cann.sh"
if [[ -n "${CANN_ENV_FILE:-}" ]]; then
  [[ -f "${CANN_ENV_FILE}" ]] || {
    echo "Persisted CANN_ENV_FILE is stale: ${CANN_ENV_FILE}" >&2
    return 1 2>/dev/null || exit 1; }
  _UAVFLOW_CANN_ENV="$(readlink -f "${CANN_ENV_FILE}")"
else
  _UAVFLOW_CANN_ENV="$(uavflow_select_cann_env)" || {
    return 1 2>/dev/null || exit 1; }
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
