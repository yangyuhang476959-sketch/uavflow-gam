#!/usr/bin/env bash
set -euo pipefail

# Curated main-effect + targeted-interaction matrix. The manifest builder
# deduplicates overlapping blocks, so a configuration is trained only once.
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_FILE="${UAVFLOW_ENV_FILE:-${ROOT}/server.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/results/remote_ablation}"
MATRIX_ROOT="${MATRIX_ROOT:-${OUTPUT_BASE}/compute_rich}"
SHARD_COUNT="${MATRIX_SHARD_COUNT:-1}"
SHARD_INDEX="${MATRIX_SHARD_INDEX:-0}"
ONLY_CELL="${MATRIX_CELL_INDEX:-}"
DRY_RUN="${MATRIX_DRY_RUN:-0}"
MATRIX_PYTHON="${PYTHON_BIN:-python}"

if (( SHARD_COUNT <= 0 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=${SHARD_INDEX}, count=${SHARD_COUNT}" >&2
  exit 2
fi

mkdir -p "${MATRIX_ROOT}"
MANIFEST="${MATRIX_ROOT}/matrix_manifest.tsv"
(
  flock 9
  if [[ ! -s "${MANIFEST}" ]]; then
    "${MATRIX_PYTHON}" \
      "${ROOT}/experiments/uavflow_remote_ablation/build_compute_rich_matrix.py" \
      --output-root "${MATRIX_ROOT}"
  fi
) 9>"${MANIFEST}.lock"

TOTAL=$(($(wc -l < "${MANIFEST}") - 1))
if [[ -n "${ONLY_CELL}" ]] && (( ONLY_CELL < 0 || ONLY_CELL >= TOTAL )); then
  echo "MATRIX_CELL_INDEX must be in [0,$((TOTAL - 1))]" >&2
  exit 2
fi
echo "compute_rich_cells=${TOTAL} shard=${SHARD_INDEX}/${SHARD_COUNT} manifest=${MANIFEST}"

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r index label blocks pose lang scale dynamic target k context schedule stage2_schedule; do
  if [[ -n "${ONLY_CELL}" && "${index}" != "${ONLY_CELL}" ]]; then
    continue
  fi
  if [[ -z "${ONLY_CELL}" ]] && (( index % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY cell=${index}/$((TOTAL - 1)) blocks=${blocks} ${label}"
    continue
  fi

  export RUN_IDS=B0
  export RUN_LABEL="${label}"
  export EXTRA_OVERRIDES_FILE="${MATRIX_ROOT}/cell_overrides/${label}.txt"
  export OUTPUT_ROOT="${MATRIX_ROOT}/runs"
  export SPLIT_FILE="${MATRIX_ROOT}/shared_split_seed42.json"
  export RUN_STAGE2=1
  export STAGE2_LR_SCHEDULE=cosine
  export STAGE2_WARMUP_STEPS=500
  export STAGE2_MIN_LR_RATIO=0.05
  if [[ "${schedule}" == "constant" ]]; then
    export STAGE1_LR_SCHEDULE=constant
    export STAGE1_WARMUP_STEPS=0
    export STAGE1_MIN_LR_RATIO=0.05
  else
    export STAGE1_LR_SCHEDULE=cosine
    export STAGE1_WARMUP_STEPS=500
    export STAGE1_MIN_LR_RATIO=0.01
  fi

  echo "[$(date '+%F %T')] cell=${index}/$((TOTAL - 1)) blocks=${blocks} ${label}"
  bash "${ROOT}/experiments/uavflow_remote_ablation/run_server.sh"
done

echo "[$(date '+%F %T')] compute-rich shard complete: ${SHARD_INDEX}/${SHARD_COUNT}"
