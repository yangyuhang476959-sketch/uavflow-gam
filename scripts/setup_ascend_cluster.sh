#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON311="${PYTHON311:-python3.11}"
ASCEND_VENV="${ASCEND_VENV:-${ROOT}/.venv-ascend}"
TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${ROOT}/.ascend/triton}"
FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${ROOT}/.ascend/flash-linear-attention}"
FLA_REPO="${FLA_REPO:-https://github.com/fla-org/flash-linear-attention.git}"
FLA_COMMIT="${FLA_COMMIT:-9f38d24980c46d46bd38614e743cdacd21906578}"
DA3_DIR="${DA3_DIR:-${ROOT}/Depth-Anything-3}"
DA3_REPO="${DA3_REPO:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
DA3_COMMIT="${DA3_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"
PROJECT_CONSTRAINTS="${ROOT}/constraints-ascend.txt"
REFERENCE_CONSTRAINTS="${ROOT}/constraints-ascend-reference.txt"
ASCEND_STACK_MODE="${ASCEND_STACK_MODE:-reference}"
if [[ "${ASCEND_STACK_MODE}" != reference && "${ASCEND_STACK_MODE}" != vendor ]]; then
  echo "ASCEND_STACK_MODE must be reference or vendor" >&2; exit 2
fi

echo '== Phase 1: platform inventory =='
uname -a
ARCH="$(uname -m)"
echo "architecture=${ARCH}"
"${PYTHON311}" --version
command -v python3 >/dev/null && python3 --version || true
command -v python3.11 >/dev/null && python3.11 --version || true
"${PYTHON311}" -c 'import platform; print(platform.machine()); print(platform.platform())'
ldd --version | head -n1
command -v npu-smi >/dev/null && npu-smi info || true
find /usr/local/Ascend -maxdepth 3 -type f -name set_env.sh -print 2>/dev/null || true
"${PYTHON311}" - <<'PY' || true
for name in ('torch', 'torch_npu', 'torchvision'):
    try:
        module = __import__(name)
        print('existing', name, getattr(module, '__version__', 'unknown'))
    except Exception as exc:
        print('existing', name, 'UNAVAILABLE', repr(exc))
PY
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "x86_64" ]]; then
  echo "Unsupported/untested architecture: ${ARCH}" >&2; exit 2
fi
# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_cann.sh"
CANN_ENV="$(uavflow_select_cann_env)"
# shellcheck disable=SC1090
source "${CANN_ENV}"
echo "Using CANN environment: ${CANN_ENV}"
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  [[ "${ARCH}" == aarch64 ]] || {
    echo 'Reference mode requires the validated Linux aarch64 platform; use ASCEND_STACK_MODE=vendor only when required by the cluster.' >&2; exit 2; }
  [[ "$("${PYTHON311}" -c 'import platform; print(platform.python_version())')" == 3.11.15 ]] || {
    echo 'Reference mode requires Python 3.11.15; exact reproduction is not possible. Use an exact interpreter or explicitly select vendor mode.' >&2; exit 2; }
  CANN_DETECTED_VERSION="${CANN_VERSION:-}"
  if [[ -z "${CANN_DETECTED_VERSION}" ]]; then
    CANN_DETECTED_VERSION="$(grep -RhoE '9\.0\.0' "$(dirname "${CANN_ENV}")" "$(dirname "$(dirname "${CANN_ENV}")")" 2>/dev/null | head -n1 || true)"
  fi
  [[ "${CANN_DETECTED_VERSION}" == 9.0.0 ]] || {
    echo 'Reference mode requires confirmed CANN 9.0.0. Set CANN_VERSION=9.0.0 only after checking the selected installation, or explicitly use ASCEND_STACK_MODE=vendor.' >&2; exit 2; }
else
  echo 'Vendor fallback selected explicitly; preserving administrator torch stack.'
fi

echo '== Phase 2: clean Python 3.11 venv =='
if [[ ! -x "${ASCEND_VENV}/bin/python" ]]; then
  # Vendor mode deliberately inherits the administrator runtime. Reference
  # mode is isolated unless explicitly supplied compatible local wheels.
  if [[ "${ASCEND_STACK_MODE}" == vendor ]]; then
    "${PYTHON311}" -m venv --system-site-packages "${ASCEND_VENV}"
  else
    "${PYTHON311}" -m venv "${ASCEND_VENV}"
  fi
fi
PY="${ASCEND_VENV}/bin/python"
"${PY}" -m pip install --upgrade 'pip<26' setuptools wheel

echo "== Phase 3: accelerator stack mode=${ASCEND_STACK_MODE} =="
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  if [[ -n "${TORCH_WHEEL:-}" || -n "${TORCH_NPU_WHEEL:-}" ]]; then
    [[ -f "${TORCH_WHEEL:-}" && -f "${TORCH_NPU_WHEEL:-}" ]] || {
      echo 'Set both TORCH_WHEEL and TORCH_NPU_WHEEL to local compatible wheels.' >&2; exit 3; }
    "${PY}" -m pip install \
      -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" \
      "${TORCH_WHEEL}" "${TORCH_NPU_WHEEL}"
  fi
  if ! "${PY}" - <<'PY'
import torch, torch_npu
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torch_npu.__version__ == '2.7.1.post4', torch_npu.__version__
print('reference torch', torch.__version__, 'torch_npu', torch_npu.__version__)
PY
  then
    echo 'Exact torch 2.7.1+cpu / torch_npu 2.7.1.post4 is unavailable. Supply compatible local wheels; do not let pip guess. If host policy forbids it, explicitly rerun with ASCEND_STACK_MODE=vendor.' >&2
    exit 3
  fi
else
  if [[ -n "${TORCH_WHEEL:-}${TORCH_NPU_WHEEL:-}${TORCHVISION_WHEEL:-}" ]]; then
    echo 'Do not supply core wheels in ASCEND_STACK_MODE=vendor' >&2; exit 3
  fi
  "${PY}" - <<'PY'
import torch, torch_npu
print('vendor torch', torch.__version__, 'torch_npu', torch_npu.__version__)
PY
fi

echo '== Phase 4: architecture-matched torchvision =='
if [[ -n "${TORCHVISION_WHEEL:-}" ]]; then
  [[ -f "${TORCHVISION_WHEEL}" ]] || { echo "Missing ${TORCHVISION_WHEEL}" >&2; exit 4; }
  case "$(basename "${TORCHVISION_WHEEL}")" in
    *aarch64*|*arm64*) [[ "${ARCH}" == aarch64 ]] || { echo 'ARM wheel on non-ARM host' >&2; exit 4; } ;;
    *x86_64*) [[ "${ARCH}" == x86_64 ]] || { echo 'x86_64 wheel on ARM host' >&2; exit 4; } ;;
  esac
  "${PY}" -m pip install --no-deps "${TORCHVISION_WHEEL}"
fi
"${PY}" - <<PY
import torch, torchvision
if "${ASCEND_STACK_MODE}" == "reference":
    assert torch.__version__.split('+')[0] == '2.7.1'
    assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
print('torchvision', torchvision.__version__)
PY

core_snapshot() {
  "${PY}" - <<'PY'
import platform
import torch, torch_npu, torchvision, numpy, scipy, transformers, huggingface_hub
print('python', platform.python_version())
for name, value in (
    ('torch', torch.__version__), ('torch_npu', torch_npu.__version__),
    ('torchvision', torchvision.__version__), ('numpy', numpy.__version__),
    ('scipy', scipy.__version__), ('transformers', transformers.__version__),
    ('huggingface_hub', huggingface_hub.__version__),
): print(name, value)
PY
}

accelerator_snapshot() {
  "${PY}" - <<'PY'
from importlib.metadata import version
for distribution in ('torch', 'torch-npu', 'torchvision'):
    print(distribution, version(distribution))
PY
}

verify_reference_stack() {
  "${PY}" - <<'PY'
import platform, torch, torch_npu, torchvision, numpy, scipy, transformers, huggingface_hub
expected = {
    'python': '3.11.15',
    'torch': '2.7.1',
    'torch_npu': '2.7.1.post4',
    'torchvision': '0.22.1',
    'numpy': '1.26.4',
    'scipy': '1.15.3',
    'transformers': '5.5.4',
    'huggingface_hub': '1.10.1',
}
actual = {
    'python': platform.python_version(),
    'torch': torch.__version__.split('+')[0],
    'torch_npu': torch_npu.__version__,
    'torchvision': torchvision.__version__.split('+')[0],
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'transformers': transformers.__version__,
    'huggingface_hub': huggingface_hub.__version__,
}
assert actual == expected, f'reference stack drifted: expected={expected} actual={actual}'
print('reference stack verified', actual)
PY
}

echo '== Phase 5/6: pinned scientific/HF stack and project requirements =='
# Allow normal dependency resolution while constraints prevent it from taking
# ownership of the selected accelerator runtime.
INSTALL_CONSTRAINTS=(-c "${PROJECT_CONSTRAINTS}")
VENDOR_RUNTIME_CONSTRAINTS=""
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  INSTALL_CONSTRAINTS+=(-c "${REFERENCE_CONSTRAINTS}")
else
  VENDOR_RUNTIME_CONSTRAINTS="$(mktemp /tmp/uavflow-vendor-runtime-XXXXXX.txt)"
  accelerator_snapshot > "${VENDOR_RUNTIME_CONSTRAINTS}"
  sed -i 's/ /==/' "${VENDOR_RUNTIME_CONSTRAINTS}"
  INSTALL_CONSTRAINTS+=(-c "${VENDOR_RUNTIME_CONSTRAINTS}")
  echo "Pinned vendor runtime constraints:"
  cat "${VENDOR_RUNTIME_CONSTRAINTS}"
fi

ACCELERATOR_BEFORE="$(mktemp /tmp/uavflow-accelerator-before-XXXXXX.txt)"
ACCELERATOR_AFTER="$(mktemp /tmp/uavflow-accelerator-after-XXXXXX.txt)"
accelerator_snapshot > "${ACCELERATOR_BEFORE}"

"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" \
  numpy==1.26.4 scipy==1.15.3 transformers==5.5.4 huggingface-hub==1.10.1
CORE_BEFORE="$(mktemp /tmp/uavflow-core-before-XXXXXX.txt)"
CORE_AFTER="$(mktemp /tmp/uavflow-core-after-XXXXXX.txt)"
core_snapshot > "${CORE_BEFORE}"

"${PY}" -m pip install --dry-run "${INSTALL_CONSTRAINTS[@]}" \
  --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" \
  --upgrade-strategy only-if-needed \
  -r "${ROOT}/requirements-ascend.txt"
core_snapshot > "${CORE_AFTER}"
accelerator_snapshot > "${ACCELERATOR_AFTER}"
diff -u "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then
  verify_reference_stack
else
  # The complete snapshot is printed for audit; the accelerator diff above is
  # the hard invariant for a vendor-provided runtime.
  echo 'Vendor protected runtime remained unchanged.'
fi
echo '== Protected stack before project installation =='
cat "${CORE_BEFORE}"
echo '== Protected stack after project installation =='
cat "${CORE_AFTER}"
rm -f "${CORE_BEFORE}" "${CORE_AFTER}" "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"
[[ -z "${VENDOR_RUNTIME_CONSTRAINTS}" ]] || rm -f "${VENDOR_RUNTIME_CONSTRAINTS}"

echo '== Phase 7: optional pycolmap remains isolated =='
echo 'Skipping requirements-optional-geometry.txt on the core training node.'

echo '== Phase 8: isolated Triton-Ascend =='
mkdir -p "${TRITON_ASCEND_TARGET}"
"${PY}" -m pip install --target "${TRITON_ASCEND_TARGET}" --no-deps \
  triton-ascend==3.2.1 pybind11

echo '== Phase 9: pinned FLA and DA3 sources =='
mkdir -p "$(dirname "${FLA_ASCEND_DIR}")"
[[ -d "${FLA_ASCEND_DIR}/.git" ]] || git clone "${FLA_REPO}" "${FLA_ASCEND_DIR}"
git -C "${FLA_ASCEND_DIR}" fetch --all --tags
git -C "${FLA_ASCEND_DIR}" checkout --detach "${FLA_COMMIT}"
[[ -d "${DA3_DIR}/.git" ]] || git clone "${DA3_REPO}" "${DA3_DIR}"
git -C "${DA3_DIR}" fetch --all --tags
git -C "${DA3_DIR}" checkout --detach "${DA3_COMMIT}"
"${PY}" -m pip install --no-deps -e "${DA3_DIR}"

echo '== Phase 10: runtime smoke =='
export REPO_ROOT="${ROOT}" ASCEND_VENV TRITON_ASCEND_TARGET FLA_ASCEND_DIR
# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_env.sh"
python - <<'PY'
import numpy, scipy, torch, torch_npu, torchvision, transformers, huggingface_hub
assert numpy.__version__ == '1.26.4'
assert scipy.__version__ == '1.15.3'
assert torch.npu.is_available()
x = torch.randn(256, 256, device='npu', dtype=torch.bfloat16)
y = x @ x
torch.npu.synchronize()
assert y.isfinite().all().item()
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs()
from depth_anything_3.api import DepthAnything3
import triton, fla
print('BF16 matmul OK', tuple(y.shape))
print('triton', triton.__version__, 'fla', getattr(fla, '__version__', 'source'))
PY

echo '== Phase 11: consistency and final versions =='
python -m pip check
core_snapshot
echo 'Ascend environment setup complete.'
