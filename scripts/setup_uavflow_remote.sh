#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating the Python 3.12 environment.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP="${PYTHON_BIN} -m pip"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
DA3_COMMIT="${DA3_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"

cd "${ROOT}"
${PIP} install pip==26.1.2 setuptools==80.10.2 wheel==0.47.0
${PIP} install torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX_URL}"
# Install the complete pinned stack in one resolver transaction.  Installing
# DA3's eager-import dependencies one by one can otherwise replace NumPy 1.26
# with NumPy 2.x and leave SciPy/PyTorch ABI-incompatible.
${PIP} install -r requirements-uavflow.txt

if [[ ! -d Depth-Anything-3/.git ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
fi
git -C Depth-Anything-3 fetch --all --tags
git -C Depth-Anything-3 checkout "${DA3_COMMIT}"
# Do not let DA3's broad upstream dependency list re-resolve the pinned stack.
# requirements-uavflow.txt already contains every dependency imported by the
# UAV-Flow training path. Heavy Open3D/gsplat applications remain unnecessary.
${PIP} install --no-deps -e Depth-Anything-3

${PIP} check
PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" scripts/verify_uavflow_remote.py --imports-only

echo "Environment ready. Run the full data audit before training."
