#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating the Python 3.12 environment.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP="${PYTHON_BIN} -m pip"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
DA3_COMMIT="${DA3_COMMIT:-2c21ea849ceec7b469a3e62ea0c0e270afc3281a}"

cd "${ROOT}"
${PIP} install --upgrade pip setuptools wheel
${PIP} install torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX_URL}"
${PIP} install -r requirements-uavflow.txt

if [[ ! -d Depth-Anything-3/.git ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
fi
git -C Depth-Anything-3 fetch --all --tags
git -C Depth-Anything-3 checkout "${DA3_COMMIT}"
# The wrapper imports DA3 from its source tree. Avoid DA3's optional rendering
# stack (Open3D/gsplat/pycolmap), which is not used by UAV-Flow training.
${PIP} install --no-deps -e Depth-Anything-3

echo "Environment ready. Run: python scripts/verify_uavflow_remote.py --help"
