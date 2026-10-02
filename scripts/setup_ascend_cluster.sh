#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BOOTSTRAP_PY="${ROOT}/scripts/ascend_bootstrap.py"
ASCEND_STACK_MODE="${ASCEND_STACK_MODE:-reference}"
ASCEND_ENV_ROOT="${ASCEND_ENV_ROOT:-${ASCEND_VENV:-${ROOT}/.ascend/env}}"
ASCEND_VENV="${ASCEND_ENV_ROOT}"
TRITON_ASCEND_TARGET="${TRITON_ASCEND_TARGET:-${ROOT}/.ascend/triton}"
FLA_ASCEND_DIR="${FLA_ASCEND_DIR:-${ROOT}/.ascend/flash-linear-attention}"
FLA_REPO="${FLA_REPO:-https://github.com/fla-org/flash-linear-attention.git}"
FLA_COMMIT="${FLA_COMMIT:-9f38d24980c46d46bd38614e743cdacd21906578}"
DA3_DIR="${DA3_DIR:-${ROOT}/Depth-Anything-3}"
DA3_REPO="${DA3_REPO:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
DA3_COMMIT="${DA3_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"
PROJECT_CONSTRAINTS="${ROOT}/constraints-ascend.txt"
REFERENCE_CONSTRAINTS="${ROOT}/constraints-ascend-reference.txt"
CANN_USER_ROOT="${CANN_USER_ROOT:-${ROOT}/.ascend/cann}"
CANN_PACKAGE_CACHE="${CANN_PACKAGE_CACHE:-${ROOT}/.ascend/packages:${HOME}/.cache/uavflow-ascend}"
PYTORCH_CPU_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
TORCH_NPU_INDEX="${TORCH_NPU_INDEX:-https://pypi.org/simple}"

[[ "${ASCEND_STACK_MODE}" == reference || "${ASCEND_STACK_MODE}" == vendor ]] || {
  echo "ASCEND_STACK_MODE must be reference or vendor" >&2; exit 2; }
ARCH_RAW="$(uname -m)"
case "${ARCH_RAW}" in
  aarch64|arm64) ARCH=aarch64; echo 'Architecture: aarch64 (validated reference host architecture)' ;;
  x86_64|amd64) ARCH=x86_64; echo 'Architecture: x86_64 (supported alternate host architecture)' ;;
  *) echo "Unsupported CPU architecture: ${ARCH_RAW}" >&2; exit 2 ;;
esac

echo '== Phase 1: host inventory (read-only) =='
date --iso-8601=seconds || date
uname -a
command -v python3 >/dev/null && python3 --version || true
command -v python3.11 >/dev/null && python3.11 --version || true
command -v conda >/dev/null && conda --version || true
command -v micromamba >/dev/null && micromamba --version || true
command -v npu-smi >/dev/null && npu-smi info || true
for info in /usr/local/Ascend/driver/version.info /usr/local/Ascend/firmware/version.info; do
  [[ -f "${info}" ]] && { echo "-- ${info}"; cat "${info}"; }
done
echo 'The project never installs or modifies Ascend Driver/Firmware.'
command -v npu-smi >/dev/null || {
  echo 'npu-smi is unavailable. The host administrator must install compatible Ascend Driver/Firmware.' >&2; exit 2; }

python_version() { "$1" -c 'import platform; print(platform.python_version())'; }
python_is_311() { [[ "$(python_version "$1" 2>/dev/null || true)" == 3.11.* ]]; }

echo '== Phase 2: isolated Python 3.11 environment =='
if [[ -x "${ASCEND_ENV_ROOT}/bin/python" ]]; then
  python_is_311 "${ASCEND_ENV_ROOT}/bin/python" || {
    echo "Existing ${ASCEND_ENV_ROOT} is not Python 3.11.x; choose a new ASCEND_ENV_ROOT." >&2; exit 2; }
else
  PYTHON_BOOTSTRAP=""
  if [[ -n "${PYTHON311:-}" ]]; then
    PYTHON_BOOTSTRAP="$(command -v "${PYTHON311}" 2>/dev/null || true)"
    [[ -n "${PYTHON_BOOTSTRAP}" ]] || PYTHON_BOOTSTRAP="${PYTHON311}"
    python_is_311 "${PYTHON_BOOTSTRAP}" || { echo "PYTHON311=${PYTHON311} is not Python 3.11.x" >&2; exit 2; }
  elif command -v python3.11 >/dev/null && python_is_311 "$(command -v python3.11)"; then
    PYTHON_BOOTSTRAP="$(command -v python3.11)"
  fi
  if [[ -n "${PYTHON_BOOTSTRAP}" ]]; then
    if [[ "${ASCEND_STACK_MODE}" == vendor ]]; then
      "${PYTHON_BOOTSTRAP}" -m venv --system-site-packages "${ASCEND_ENV_ROOT}"
    else
      "${PYTHON_BOOTSTRAP}" -m venv "${ASCEND_ENV_ROOT}"
    fi
  else
    [[ "${ASCEND_STACK_MODE}" == reference ]] || {
      echo 'Vendor mode requires an administrator Python 3.11 environment exposing torch_npu.' >&2; exit 2; }
    ENV_MANAGER=""
    for candidate in micromamba mamba conda; do
      if command -v "${candidate}" >/dev/null; then ENV_MANAGER="$(command -v "${candidate}")"; break; fi
    done
    if [[ -z "${ENV_MANAGER}" ]]; then
      command -v curl >/dev/null && command -v tar >/dev/null || {
        echo 'Python 3.11 is absent and micromamba bootstrap requires curl and tar.' >&2; exit 2; }
      [[ "${ARCH}" == aarch64 ]] && MAMBA_PLATFORM=linux-aarch64 || MAMBA_PLATFORM=linux-64
      MICROMAMBA_HOME="${ROOT}/.ascend/micromamba"
      mkdir -p "${MICROMAMBA_HOME}"
      echo "Downloading micromamba from its official endpoint for ${MAMBA_PLATFORM}"
      curl --fail --location --silent --show-error "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" \
        | tar -xj -C "${MICROMAMBA_HOME}" bin/micromamba
      ENV_MANAGER="${MICROMAMBA_HOME}/bin/micromamba"
    fi
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${ROOT}/.ascend/mamba-root}"
    "${ENV_MANAGER}" create -y -p "${ASCEND_ENV_ROOT}" python=3.11 pip
  fi
fi
PY="${ASCEND_ENV_ROOT}/bin/python"
python_is_311 "${PY}" || { echo 'Bootstrap did not produce Python 3.11.x' >&2; exit 2; }
echo "Python environment: ${ASCEND_ENV_ROOT} ($(python_version "${PY}"))"
"${PY}" -m pip install --upgrade 'pip<26' setuptools wheel

# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_cann.sh"
installer_from_cache() {
  local expected cache
  expected="$(${PY} "${BOOTSTRAP_PY}" cann-installer-name --arch "${ARCH}")"
  if [[ -n "${CANN_INSTALLER:-}" ]]; then printf '%s\n' "${CANN_INSTALLER}"; return; fi
  IFS=':' read -r -a caches <<< "${CANN_PACKAGE_CACHE}"
  local -a matches=()
  for cache in "${caches[@]}"; do [[ -f "${cache}/${expected}" ]] && matches+=("${cache}/${expected}"); done
  (( ${#matches[@]} == 1 )) && { printf '%s\n' "${matches[0]}"; return; }
  (( ${#matches[@]} > 1 )) && { echo "Multiple CANN installers found; set CANN_INSTALLER: ${matches[*]}" >&2; return 2; }
  return 1
}
install_reference_cann() {
  local installer expected actual
  expected="$(${PY} "${BOOTSTRAP_PY}" cann-installer-name --arch "${ARCH}")"
  installer="$(installer_from_cache)" || {
    echo "Obtain official ${expected}; set CANN_INSTALLER or place it in ${CANN_PACKAGE_CACHE}." >&2
    echo 'No CANN binary URL is guessed. Explicit ASCEND_STACK_MODE=vendor remains available.' >&2; return 2; }
  "${PY}" -c "import sys; from pathlib import Path; sys.path.insert(0,'${ROOT}/scripts'); from ascend_bootstrap import validate_cann_installer; validate_cann_installer(Path('${installer}'), arch='${ARCH}')"
  [[ -n "${CANN_INSTALLER_SHA256:-}" ]] || {
    echo 'Set CANN_INSTALLER_SHA256 to the checksum supplied with the official installer.' >&2; return 2; }
  actual="$(sha256sum "${installer}" | awk '{print $1}')"
  [[ "${actual}" == "${CANN_INSTALLER_SHA256,,}" ]] || {
    echo "CANN installer SHA-256 mismatch: expected ${CANN_INSTALLER_SHA256}, got ${actual}" >&2; return 2; }
  mkdir -p "${CANN_USER_ROOT}"; chmod u+x "${installer}"
  echo "Installing ${expected} into ${CANN_USER_ROOT}; Driver/Firmware are untouched."
  "${installer}" --quiet --install --install-path="${CANN_USER_ROOT}" ${CANN_INSTALLER_ARGS:-}
  export CANN_ROOT="${CANN_USER_ROOT}"
}

echo '== Phase 3: CANN Toolkit discovery/version =='
set +e
CANN_ENV="$(uavflow_select_cann_env)"
CANN_DISCOVERY_STATUS=$?
set -e
if (( CANN_DISCOVERY_STATUS == 3 || CANN_DISCOVERY_STATUS == 4 )); then
  exit "${CANN_DISCOVERY_STATUS}"
fi
if (( CANN_DISCOVERY_STATUS != 0 )); then
  [[ "${ASCEND_STACK_MODE}" == reference ]] || exit "${CANN_DISCOVERY_STATUS}"
  install_reference_cann
  CANN_ENV="$(uavflow_select_cann_env)"
fi
CANN_METADATA_ROOT="${CANN_ROOT:-$(dirname "${CANN_ENV}")}"
CANN_JSON="$(${PY} "${BOOTSTRAP_PY}" cann-version "${CANN_METADATA_ROOT}" --arch "${ARCH}")" || {
  echo 'Selected CANN has no valid official ascend_toolkit_install.info metadata.' >&2; exit 2; }
CANN_DETECTED_VERSION="$(${PY} -c 'import json,sys; print(json.loads(sys.argv[1])["version"])' "${CANN_JSON}")"
CANN_METADATA="$(${PY} -c 'import json,sys; print(json.loads(sys.argv[1])["metadata"])' "${CANN_JSON}")"
echo "Selected CANN environment: ${CANN_ENV}"
echo "Detected CANN version: ${CANN_DETECTED_VERSION} (${CANN_METADATA})"
if [[ "${ASCEND_STACK_MODE}" == reference && "${CANN_DETECTED_VERSION}" != 9.0.0 ]]; then
  echo "Reference mode targets CANN 9.0.0; selected installation is ${CANN_DETECTED_VERSION}." >&2
  echo 'Install 9.0.0 separately in user space or explicitly choose ASCEND_STACK_MODE=vendor.' >&2; exit 2
fi
# shellcheck disable=SC1090
source "${CANN_ENV}"
export CANN_ROOT="${CANN_METADATA_ROOT}"

echo "== Phase 4: accelerator Python stack mode=${ASCEND_STACK_MODE} =="
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
    echo "Installing local reference wheels for ${ARCH}: ${TORCH_WHEEL}, ${TORCH_NPU_WHEEL}, ${TORCHVISION_WHEEL}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" "${TORCH_WHEEL}" "${TORCH_NPU_WHEEL}"
    "${PY}" -m pip install --no-deps "${TORCHVISION_WHEEL}"
  else
    echo "Online plan: torch 2.7.1+cpu and torchvision 0.22.1 from ${PYTORCH_CPU_INDEX}; torch-npu 2.7.1.post4 from ${TORCH_NPU_INDEX}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torch==2.7.1 --index-url "${PYTORCH_CPU_INDEX}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torch-npu==2.7.1.post4 --index-url "${TORCH_NPU_INDEX}"
    "${PY}" -m pip install -c "${PROJECT_CONSTRAINTS}" -c "${REFERENCE_CONSTRAINTS}" torchvision==0.22.1 --index-url "${PYTORCH_CPU_INDEX}"
  fi
else
  [[ -z "${TORCH_WHEEL:-}${TORCH_NPU_WHEEL:-}${TORCHVISION_WHEEL:-}" ]] || { echo 'Do not supply core wheels in vendor mode.' >&2; exit 3; }
fi
"${PY}" - <<PY
import torch, torch_npu, torchvision
if "${ASCEND_STACK_MODE}" == "reference":
    assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
    assert torch_npu.__version__ == '2.7.1.post4', torch_npu.__version__
    assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'torchvision', torchvision.__version__)
PY

core_snapshot() { "${PY}" - <<'PY'
import platform, torch, torch_npu, torchvision, numpy, scipy, transformers, huggingface_hub
print('python', platform.python_version())
for name, value in (('torch',torch.__version__),('torch_npu',torch_npu.__version__),('torchvision',torchvision.__version__),('numpy',numpy.__version__),('scipy',scipy.__version__),('transformers',transformers.__version__),('huggingface_hub',huggingface_hub.__version__)): print(name, value)
PY
}
accelerator_snapshot() { "${PY}" - <<'PY'
from importlib.metadata import version
for distribution in ('torch','torch-npu','torchvision'): print(distribution, version(distribution))
PY
}
verify_reference_stack() { "${PY}" - <<'PY'
import platform, torch, torch_npu, torchvision, numpy, scipy, transformers, huggingface_hub
assert platform.python_version_tuple()[:2] == ('3','11')
expected={'torch':'2.7.1','torch_npu':'2.7.1.post4','torchvision':'0.22.1','numpy':'1.26.4','scipy':'1.15.3','transformers':'5.5.4','huggingface_hub':'1.10.1'}
actual={'torch':torch.__version__.split('+')[0],'torch_npu':torch_npu.__version__,'torchvision':torchvision.__version__.split('+')[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__}
assert actual == expected, f'reference stack drifted: {actual}'
PY
}

echo '== Phase 5: project dependencies with protected runtime =='
INSTALL_CONSTRAINTS=(-c "${PROJECT_CONSTRAINTS}"); VENDOR_RUNTIME_CONSTRAINTS=""
if [[ "${ASCEND_STACK_MODE}" == reference ]]; then INSTALL_CONSTRAINTS+=(-c "${REFERENCE_CONSTRAINTS}"); else
  VENDOR_RUNTIME_CONSTRAINTS="$(mktemp /tmp/uavflow-vendor-runtime-XXXXXX.txt)"
  accelerator_snapshot | sed 's/ /==/' > "${VENDOR_RUNTIME_CONSTRAINTS}"
  INSTALL_CONSTRAINTS+=(-c "${VENDOR_RUNTIME_CONSTRAINTS}"); cat "${VENDOR_RUNTIME_CONSTRAINTS}"
fi
ACCELERATOR_BEFORE="$(mktemp /tmp/uavflow-accelerator-before-XXXXXX.txt)"; ACCELERATOR_AFTER="$(mktemp /tmp/uavflow-accelerator-after-XXXXXX.txt)"
accelerator_snapshot > "${ACCELERATOR_BEFORE}"
"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" numpy==1.26.4 scipy==1.15.3 transformers==5.5.4 huggingface-hub==1.10.1
CORE_BEFORE="$(mktemp /tmp/uavflow-core-before-XXXXXX.txt)"; CORE_AFTER="$(mktemp /tmp/uavflow-core-after-XXXXXX.txt)"; core_snapshot > "${CORE_BEFORE}"
"${PY}" -m pip install --dry-run "${INSTALL_CONSTRAINTS[@]}" --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
"${PY}" -m pip install "${INSTALL_CONSTRAINTS[@]}" --upgrade-strategy only-if-needed -r "${ROOT}/requirements-ascend.txt"
core_snapshot > "${CORE_AFTER}"; accelerator_snapshot > "${ACCELERATOR_AFTER}"; diff -u "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"
[[ "${ASCEND_STACK_MODE}" == reference ]] && verify_reference_stack || echo 'Vendor protected runtime remained unchanged.'
echo 'Protected stack before:'; cat "${CORE_BEFORE}"; echo 'Protected stack after:'; cat "${CORE_AFTER}"
rm -f "${CORE_BEFORE}" "${CORE_AFTER}" "${ACCELERATOR_BEFORE}" "${ACCELERATOR_AFTER}"; [[ -z "${VENDOR_RUNTIME_CONSTRAINTS}" ]] || rm -f "${VENDOR_RUNTIME_CONSTRAINTS}"

echo '== Phase 6: isolated Triton-Ascend, pinned FLA and DA3 =='
echo 'Skipping optional/offline-only pycolmap.'
mkdir -p "${TRITON_ASCEND_TARGET}" "$(dirname "${FLA_ASCEND_DIR}")"
"${PY}" -m pip install --target "${TRITON_ASCEND_TARGET}" --no-deps triton-ascend==3.2.1 pybind11
[[ -d "${FLA_ASCEND_DIR}/.git" ]] || git clone "${FLA_REPO}" "${FLA_ASCEND_DIR}"; git -C "${FLA_ASCEND_DIR}" fetch --all --tags; git -C "${FLA_ASCEND_DIR}" checkout --detach "${FLA_COMMIT}"
[[ -d "${DA3_DIR}/.git" ]] || git clone "${DA3_REPO}" "${DA3_DIR}"; git -C "${DA3_DIR}" fetch --all --tags; git -C "${DA3_DIR}" checkout --detach "${DA3_COMMIT}"; "${PY}" -m pip install --no-deps -e "${DA3_DIR}"

echo '== Phase 7: Ascend compatibility smoke =='
export REPO_ROOT="${ROOT}" ASCEND_ENV_ROOT ASCEND_VENV TRITON_ASCEND_TARGET FLA_ASCEND_DIR DA3_DIR
export PYTHONPATH="${TRITON_ASCEND_TARGET}:${FLA_ASCEND_DIR}:${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" - <<'PY'
import torch, torch_npu, torchvision
assert torch.npu.is_available() and torch.npu.device_count() > 0
x=torch.randn(64,64,device='npu',dtype=torch.bfloat16,requires_grad=True); y=(x@x).float().square().mean(); y.backward(); torch.npu.synchronize()
assert torch.isfinite(y).item() and torch.isfinite(x.grad).all().item()
from transformers import Qwen3_5ForConditionalGeneration
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs(); from depth_anything_3.api import DepthAnything3
import triton, fla
print('NPU BF16 forward/backward PASS', torch.npu.device_count(), torchvision.__version__, triton.__version__)
PY

echo '== Phase 8: consistency and environment report =='
"${PY}" -m pip check
REPORT="${ROOT}/.ascend/environment-report.json"; mkdir -p "$(dirname "${REPORT}")"
export UAVFLOW_REPORT="${REPORT}" UAVFLOW_CANN_ENV="${CANN_ENV}" UAVFLOW_CANN_VERSION="${CANN_DETECTED_VERSION}" UAVFLOW_CANN_METADATA="${CANN_METADATA}" UAVFLOW_ARCH="${ARCH}"
"${PY}" - <<'PY'
import datetime, importlib.metadata as md, json, os, platform, subprocess
from pathlib import Path
import numpy, scipy, torch, torch_npu, torchvision, transformers, huggingface_hub, triton
def cmd(*args):
  try: return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout.strip()
  except OSError as exc: return repr(exc)
root=Path(os.environ['REPO_ROOT'])
report={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uname':platform.uname()._asdict(),'architecture':os.environ['UAVFLOW_ARCH'],'npu_smi':cmd('npu-smi','info'),'driver_version_info':Path('/usr/local/Ascend/driver/version.info').read_text(errors='replace') if Path('/usr/local/Ascend/driver/version.info').is_file() else None,'firmware_version_info':Path('/usr/local/Ascend/firmware/version.info').read_text(errors='replace') if Path('/usr/local/Ascend/firmware/version.info').is_file() else None,'cann_environment':os.environ['UAVFLOW_CANN_ENV'],'cann_metadata':os.environ['UAVFLOW_CANN_METADATA'],'cann_version':os.environ['UAVFLOW_CANN_VERSION'],'python':platform.python_version(),'environment_path':os.environ['ASCEND_ENV_ROOT'],'torch':torch.__version__,'torch_npu':torch_npu.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__,'triton_ascend_distribution':md.version('triton-ascend'),'triton_import':triton.__version__,'fla_commit':cmd('git','-C',os.environ['FLA_ASCEND_DIR'],'rev-parse','HEAD'),'da3_commit':cmd('git','-C',os.environ['DA3_DIR'],'rev-parse','HEAD')}
Path(os.environ['UAVFLOW_REPORT']).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
PY
echo "Ascend environment setup complete. Report: ${REPORT}"
