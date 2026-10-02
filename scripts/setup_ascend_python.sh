#!/usr/bin/env bash
# Create the isolated Python stack. CANN selection is handled separately.
set -Eeuo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BOOTSTRAP_PY="${ROOT}/scripts/ascend_bootstrap.py"
ASCEND_STACK_MODE="${ASCEND_STACK_MODE:-reference}"
MINICONDA_ROOT="${MINICONDA_ROOT:-${ROOT}/.ascend/miniconda3}"
ASCEND_ENV_ROOT="${ASCEND_ENV_ROOT:-${ROOT}/.ascend/env}"
TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${ROOT}/.ascend/triton}"
FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${ROOT}/.ascend/flash-linear-attention}"
FLA_REPO="${FLA_REPO:-https://github.com/fla-org/flash-linear-attention.git}"
FLA_COMMIT="${FLA_COMMIT:-9f38d24980c46d46bd38614e743cdacd21906578}"
DA3_DIR="${DA3_DIR:-${ROOT}/Depth-Anything-3}"
DA3_REPO="${DA3_REPO:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
DA3_COMMIT="${DA3_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"
RUNTIME_ENV="${ASCEND_RUNTIME_ENV:-${ROOT}/.ascend/runtime.env}"
PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
TORCH_NPU_INDEX="${TORCH_NPU_INDEX:-https://pypi.org/simple}"

case "$(uname -m)" in
  aarch64|arm64) ARCH=aarch64 ;;
  x86_64|amd64) ARCH=x86_64 ;;
  *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 2 ;;
esac
[[ -f "${RUNTIME_ENV}" ]] || {
  echo "Missing ${RUNTIME_ENV}; run scripts/setup_ascend_cann.sh first." >&2; exit 2; }
# shellcheck disable=SC1090
source "${RUNTIME_ENV}"
[[ -f "${CANN_ENV_FILE:-}" ]] || { echo "Persisted CANN_ENV_FILE is stale: ${CANN_ENV_FILE:-unset}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${CANN_ENV_FILE}"

python_is_311() { [[ "$("$1" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)" == 3.11.* ]]; }

bootstrap_conda() {
  if command -v conda >/dev/null; then
    command -v conda
    return
  fi
  if [[ -x "${MINICONDA_ROOT}/bin/conda" ]]; then
    printf '%s\n' "${MINICONDA_ROOT}/bin/conda"
    return
  fi
  [[ "${ASCEND_STACK_MODE}" == reference ]] || {
    echo 'Vendor mode requires an existing conda and administrator-compatible Python environment.' >&2; return 2; }
  local expected installer url
  expected="$(python3 "${BOOTSTRAP_PY}" miniconda-installer-name --arch "${ARCH}")"
  installer="${MINICONDA_INSTALLER:-${ROOT}/.ascend/packages/${expected}}"
  if [[ ! -f "${installer}" ]]; then
    [[ -z "${MINICONDA_INSTALLER:-}" ]] || { echo "MINICONDA_INSTALLER does not exist: ${installer}" >&2; return 2; }
    command -v curl >/dev/null || { echo 'curl is required to download official Miniconda.' >&2; return 2; }
    url="$(python3 "${BOOTSTRAP_PY}" miniconda-installer-url --arch "${ARCH}")"
    mkdir -p "$(dirname "${installer}")"
    echo "Downloading official Miniconda: ${url}" >&2
    curl --fail --location --retry 3 --output "${installer}.part" "${url}"
    mv "${installer}.part" "${installer}"
  fi
  [[ "$(basename "${installer}")" == "${expected}" ]] || {
    echo "Expected ${expected}; got $(basename "${installer}")" >&2; return 2; }
  bash "${installer}" -b -p "${MINICONDA_ROOT}" >&2
  printf '%s\n' "${MINICONDA_ROOT}/bin/conda"
}

echo '== Isolated Python 3.11 environment (conda) =='
CONDA_BIN="$(bootstrap_conda)"
if [[ ! -x "${ASCEND_ENV_ROOT}/bin/python" ]]; then
  "${CONDA_BIN}" create -y -p "${ASCEND_ENV_ROOT}" python=3.11 pip
fi
PY="${ASCEND_ENV_ROOT}/bin/python"
python_is_311 "${PY}" || {
  echo "Existing ${ASCEND_ENV_ROOT} is not Python 3.11.x; choose a new ASCEND_ENV_ROOT." >&2; exit 2; }
"${PY}" -m pip install --upgrade 'pip<26' setuptools wheel

PROJECT_CONSTRAINTS="${ROOT}/constraints-ascend.txt"
REFERENCE_CONSTRAINTS="${ROOT}/constraints-ascend-reference.txt"
accelerator_snapshot() { "${PY}" - <<'PY'
from importlib.metadata import version
for distribution in ('torch','torch-npu','torchvision'): print(distribution, version(distribution))
PY
}
verify_reference_stack() { "${PY}" - <<'PY'
import platform, torch, torch_npu, torchvision
assert platform.python_version_tuple()[:2] == ('3', '11')
actual = (torch.__version__.split('+')[0], torch_npu.__version__, torchvision.__version__.split('+')[0])
assert actual == ('2.7.1', '2.7.1.post4', '0.22.1'), actual
PY
}
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  local_wheels=0
  for value in "${TORCH_WHEEL:-}" "${TORCH_NPU_WHEEL:-}" "${TORCHVISION_WHEEL:-}"; do [[ -n "${value}" ]] && ((local_wheels+=1)); done
  if (( local_wheels != 0 && local_wheels != 3 )); then
    echo 'Offline mode requires TORCH_WHEEL, TORCH_NPU_WHEEL, and TORCHVISION_WHEEL together.' >&2; exit 3
  fi
  if (( local_wheels == 3 )); then
    for wheel in "${TORCH_WHEEL}" "${TORCH_NPU_WHEEL}" "${TORCHVISION_WHEEL}"; do
      "${PY}" -c "import sys; from pathlib import Path; sys.path.insert(0,'${ROOT}/scripts'); from ascend_bootstrap import validate_local_wheel; validate_local_wheel(Path('${wheel}'), arch='${ARCH}')"
    done
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" "${TORCH_WHEEL}" "${TORCH_NPU_WHEEL}"
    "${PY}" -m pip install --no-deps "${TORCHVISION_WHEEL}"
  else
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torch==2.7.1 --index-url "${PYTORCH_CPU_INDEX}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torch-npu==2.7.1.post4 --index-url "${TORCH_NPU_INDEX}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torchvision==0.22.1 --index-url "${PYTORCH_CPU_INDEX}"
  fi
else
  "${PY}" - <<'PY'
import torch, torch_npu, torchvision
print('Using vendor core stack', torch.__version__, torch_npu.__version__, torchvision.__version__)
PY
fi

ACCELERATOR_BEFORE="$(mktemp /tmp/uavflow-accelerator-before-XXXXXX.txt)"
ACCELERATOR_AFTER="$(mktemp /tmp/uavflow-accelerator-after-XXXXXX.txt)"
accelerator_snapshot > "${ACCELERATOR_BEFORE}"

INSTALL_CONSTRAINTS=(-c "${PROJECT_CONSTRAINTS}")
VENDOR_RUNTIME_CONSTRAINTS=""
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  INSTALL_CONSTRAINTS+=(-c "${REFERENCE_CONSTRAINTS}")
else
  VENDOR_RUNTIME_CONSTRAINTS="$(mktemp /tmp/uavflow-vendor-runtime-XXXXXX.txt)"
  accelerator_snapshot | sed 's/ /==/' > "${VENDOR_RUNTIME_CONSTRAINTS}"
  INSTALL_CONSTRAINTS+=(-c "${VENDOR_RUNTIME_CONSTRAINTS}")
fi
"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" numpy==1.26.4 scipy==1.15.3 transformers==5.5.4 huggingface-hub==1.10.1
"${PY}" -m pip install --dry-run "${INSTALL_CONSTRAINTS[@]}" --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
accelerator_snapshot > "${ACCELERATOR_AFTER}"
diff -u "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"
[[ "${ASCEND_STACK_MODE}" == reference ]] && verify_reference_stack
rm -f "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"
[[ -z "${VENDOR_RUNTIME_CONSTRAINTS}" ]] || rm -f "${VENDOR_RUNTIME_CONSTRAINTS}"

mkdir -p "${TRITON_ASCEND_TARGET}" "$(dirname "${FLA_ASCEND_DIR}")"
"${PY}" -m pip install --target "${TRITON_ASCEND_TARGET}" --no-deps triton-ascend==3.2.1 pybind11
[[ -d "${FLA_ASCEND_DIR}/.git" ]] || git clone "${FLA_REPO}" "${FLA_ASCEND_DIR}"
git -C "${FLA_ASCEND_DIR}" fetch --all --tags
git -C "${FLA_ASCEND_DIR}" checkout --detach "${FLA_COMMIT}"
[[ -d "${DA3_DIR}/.git" ]] || git clone "${DA3_REPO}" "${DA3_DIR}"
git -C "${DA3_DIR}" fetch --all --tags
git -C "${DA3_DIR}" checkout --detach "${DA3_COMMIT}"
"${PY}" -m pip install --no-deps -e "${DA3_DIR}"

tmp_runtime="$(mktemp "${RUNTIME_ENV}.XXXXXX")"
grep -Ev '^export (MINICONDA_ROOT|ASCEND_ENV_ROOT|ASCEND_VENV|TRITON_ASCEND_TARGET|FLA_ASCEND_DIR|DA3_DIR)=' "${RUNTIME_ENV}" > "${tmp_runtime}" || true
{
  printf 'export MINICONDA_ROOT=%q\n' "${MINICONDA_ROOT}"
  printf 'export ASCEND_ENV_ROOT=%q\n' "${ASCEND_ENV_ROOT}"
  printf 'export ASCEND_VENV=%q\n' "${ASCEND_ENV_ROOT}"
  printf 'export TRITON_ASCEND_TARGET=%q\n' "${TRITON_ASCEND_TARGET}"
  printf 'export FLA_ASCEND_DIR=%q\n' "${FLA_ASCEND_DIR}"
  printf 'export DA3_DIR=%q\n' "${DA3_DIR}"
} >> "${tmp_runtime}"
mv "${tmp_runtime}" "${RUNTIME_ENV}"
chmod 600 "${RUNTIME_ENV}"
"${PY}" -m pip check
echo "Python environment ready: ${ASCEND_ENV_ROOT}"
