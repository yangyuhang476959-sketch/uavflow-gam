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
CONSTRAINTS="${ROOT}/constraints-ascend.txt"

echo '== Phase 1: platform inventory =='
uname -a
ARCH="$(uname -m)"
echo "architecture=${ARCH}"
"${PYTHON311}" --version
"${PYTHON311}" -c 'import platform; print(platform.machine()); print(platform.platform())'
ldd --version | head -n1
command -v npu-smi >/dev/null && npu-smi info || true
find /usr/local/Ascend -maxdepth 3 -type f -name set_env.sh -print 2>/dev/null || true
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "x86_64" ]]; then
  echo "Unsupported/untested architecture: ${ARCH}" >&2; exit 2
fi
if [[ -n "${CANN_ROOT:-}" && -f "${CANN_ROOT}/set_env.sh" ]]; then
  CANN_ENV="${CANN_ROOT}/set_env.sh"
elif [[ -n "${CANN_ROOT:-}" && -f "${CANN_ROOT}/bin/set_env.sh" ]]; then
  CANN_ENV="${CANN_ROOT}/bin/set_env.sh"
else
  CANN_ENV="$(find /usr/local/Ascend -maxdepth 3 -type f -name set_env.sh 2>/dev/null | sort | head -n1)"
fi
[[ -n "${CANN_ENV:-}" && -f "${CANN_ENV}" ]] || {
  echo 'CANN set_env.sh not found; set CANN_ROOT.' >&2; exit 2; }
# shellcheck disable=SC1090
source "${CANN_ENV}"
echo "Using CANN environment: ${CANN_ENV}"

echo '== Phase 2: clean Python 3.11 venv =='
if [[ ! -x "${ASCEND_VENV}/bin/python" ]]; then
  # Vendor images commonly expose torch_npu only in system site-packages.
  # Set ASCEND_INHERIT_VENDOR_PACKAGES=0 when local vendor wheels are supplied.
  if [[ "${ASCEND_INHERIT_VENDOR_PACKAGES:-1}" == 1 ]]; then
    "${PYTHON311}" -m venv --system-site-packages "${ASCEND_VENV}"
  else
    "${PYTHON311}" -m venv "${ASCEND_VENV}"
  fi
fi
PY="${ASCEND_VENV}/bin/python"
"${PY}" -m pip install --upgrade 'pip<26' setuptools wheel

echo '== Phase 3: vendor torch + torch_npu pair =='
if [[ -n "${TORCH_WHEEL:-}" || -n "${TORCH_NPU_WHEEL:-}" ]]; then
  [[ -f "${TORCH_WHEEL:-}" && -f "${TORCH_NPU_WHEEL:-}" ]] || {
    echo 'Set both TORCH_WHEEL and TORCH_NPU_WHEEL to local compatible wheels.' >&2; exit 3; }
  "${PY}" -m pip install --no-deps "${TORCH_WHEEL}" "${TORCH_NPU_WHEEL}"
fi
"${PY}" - <<'PY'
import torch, torch_npu
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torch_npu.__version__ == '2.7.1.post4', torch_npu.__version__
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)
PY

echo '== Phase 4: architecture-matched torchvision =='
if [[ -n "${TORCHVISION_WHEEL:-}" ]]; then
  [[ -f "${TORCHVISION_WHEEL}" ]] || { echo "Missing ${TORCHVISION_WHEEL}" >&2; exit 4; }
  case "$(basename "${TORCHVISION_WHEEL}")" in
    *aarch64*|*arm64*) [[ "${ARCH}" == aarch64 ]] || { echo 'ARM wheel on non-ARM host' >&2; exit 4; } ;;
    *x86_64*) [[ "${ARCH}" == x86_64 ]] || { echo 'x86_64 wheel on ARM host' >&2; exit 4; } ;;
  esac
  "${PY}" -m pip install --no-deps "${TORCHVISION_WHEEL}"
fi
"${PY}" - <<'PY'
import torch, torchvision
assert torch.__version__.split('+')[0] == '2.7.1'
assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
print('torchvision', torchvision.__version__)
PY

core_snapshot() {
  "${PY}" - <<'PY'
import torch, torch_npu, torchvision, numpy, scipy, transformers, huggingface_hub
for name, value in (
    ('torch', torch.__version__), ('torch_npu', torch_npu.__version__),
    ('torchvision', torchvision.__version__), ('numpy', numpy.__version__),
    ('scipy', scipy.__version__), ('transformers', transformers.__version__),
    ('huggingface_hub', huggingface_hub.__version__),
): print(name, value)
PY
}

echo '== Phase 5/6: pinned scientific/HF stack and project requirements =='
# Establish the protected versions first. Constraints then prevent timm or
# torchvision dependencies from taking ownership of the torch decision.
"${PY}" -m pip install -c "${CONSTRAINTS}" \
  numpy==1.26.4 scipy==1.15.3 transformers==5.5.4 huggingface-hub==1.10.1
core_snapshot > /tmp/uavflow_core_before.txt
"${PY}" -m pip install --dry-run -c "${CONSTRAINTS}" \
  --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
"${PY}" -m pip install -c "${CONSTRAINTS}" --upgrade-strategy only-if-needed \
  -r "${ROOT}/requirements-ascend.txt"
core_snapshot > /tmp/uavflow_core_after.txt
diff -u /tmp/uavflow_core_before.txt /tmp/uavflow_core_after.txt

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
