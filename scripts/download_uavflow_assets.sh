#!/usr/bin/env bash
set -euo pipefail

# Downloads public UAV-Flow and model assets. The derived depth sidecars are
# downloaded only after DEPTH_DATASET_REPO is set to the final published repo.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data_remote}"
MODEL_ROOT="${MODEL_ROOT:-${ROOT}/checkpoints}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DEPTH_HUB="${DEPTH_HUB:-modelscope}"
DEPTH_DATASET_REPO="${DEPTH_DATASET_REPO:-acetaffy123/UAV-Flow-Sim-Depth}"

mkdir -p "${DATA_ROOT}" "${MODEL_ROOT}"
export HF_ENDPOINT

hf download wangxiangyu0814/UAV-Flow-Sim \
  --repo-type dataset --local-dir "${DATA_ROOT}/UAV-Flow-Sim"
hf download Qwen/Qwen3.5-2B \
  --local-dir "${MODEL_ROOT}/qwen3.5-2b"
hf download google-t5/t5-base \
  --local-dir "${MODEL_ROOT}/t5-base"
hf download SeonghuJeon/3da-libero-training-assets \
  checkpoints/track4world_da3.pth --repo-type dataset --local-dir "${ROOT}"

DEPTH_ARCHIVE="${DATA_ROOT}/UAV-Flow-Sim-Depth-Archive"
if [[ "${DEPTH_HUB}" == "huggingface" ]]; then
  hf download "${DEPTH_DATASET_REPO}" --repo-type dataset --local-dir "${DEPTH_ARCHIVE}"
elif [[ "${DEPTH_HUB}" == "modelscope" ]]; then
  command -v modelscope >/dev/null || {
    echo "Install ModelScope first: python -m pip install modelscope" >&2
    exit 2
  }
  modelscope download "${DEPTH_DATASET_REPO}" \
    --repo-type dataset --local-dir "${DEPTH_ARCHIVE}"
else
  echo "DEPTH_HUB must be huggingface or modelscope, got ${DEPTH_HUB}" >&2
  exit 2
fi

python scripts/extract_uavflow_depth.py \
  --dataset-root "${DEPTH_ARCHIVE}" \
  --output "${DATA_ROOT}/UAV-Flow-Sim-Depth"
