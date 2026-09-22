#!/usr/bin/env bash
set -euo pipefail

# Portable sequential launcher for the UAV-Flow full-rank ablation. Every run
# is restartable: a completed stage has a _SUCCESS marker; an interrupted stage
# resumes from its newest epoch checkpoint.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG="${CONFIG:-${ROOT}/experiments/uavflow_remote_ablation/base.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/remote_ablation}"
SPLIT_FILE="${SPLIT_FILE:-${OUTPUT_ROOT}/shared_split_seed42.json}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
if [[ -n "${BATCH_SIZE:-}" ]]; then
  if (( BATCH_SIZE * NPROC != GLOBAL_BATCH_SIZE )); then
    echo "BATCH_SIZE(${BATCH_SIZE}) * NPROC(${NPROC}) != GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE})" >&2
    exit 2
  fi
else
  if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
    echo "GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) must be divisible by NPROC(${NPROC})" >&2
    exit 2
  fi
  BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC))
fi
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-0}"
DEPTH_FIXED_SCALE_METERS="${DEPTH_FIXED_SCALE_METERS:-100.0}"
BASE_LR="${BASE_LR:-5.0e-5}"
STAGE1_LR_SCHEDULE="${STAGE1_LR_SCHEDULE:-${LR_SCHEDULE:-constant}}"
STAGE1_WARMUP_STEPS="${STAGE1_WARMUP_STEPS:-${WARMUP_STEPS:-0}}"
STAGE1_MIN_LR_RATIO="${STAGE1_MIN_LR_RATIO:-${MIN_LR_RATIO:-0.05}}"
# Stop fine-tuning is deliberately standardized to the schedule used by the
# earlier CLIP Stop runs, independent of the Stage-1 schedule ablation.
STAGE2_LR_SCHEDULE="${STAGE2_LR_SCHEDULE:-cosine}"
STAGE2_WARMUP_STEPS="${STAGE2_WARMUP_STEPS:-500}"
STAGE2_MIN_LR_RATIO="${STAGE2_MIN_LR_RATIO:-0.05}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
RUN_IDS="${RUN_IDS:-B0,S1COS,P1,L1,D1,D2,D1LOG,W3,W5,W10,H0,HB,F3,F7,F10,M1,C1_PL,C2_D2HB,C3_W3HB,C4_F3W3,C5_F10HB,CA1_HB}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_LABEL="${RUN_LABEL:-}"
EXTRA_OVERRIDES_FILE="${EXTRA_OVERRIDES_FILE:-}"

: "${UAVFLOW_SIM_ROOT:?Set UAVFLOW_SIM_ROOT to the UAV-Flow-Sim parquet root}"
: "${UAVFLOW_DEPTH_ROOT:?Set UAVFLOW_DEPTH_ROOT to the consolidated/hybrid depth root}"
: "${DA3_CHECKPOINT:?Set DA3_CHECKPOINT to Track4World DA3 checkpoint}"
: "${QWEN_MODEL:?Set QWEN_MODEL to the local Qwen3.5-2B directory}"
: "${T5_MODEL:?Set T5_MODEL to the local T5 directory}"

ACTION_STATS_DIR="${ACTION_STATS_DIR:-${ROOT}/data/uavflow_stats_sim_openvla_yaw4d}"
IDM_CHECKPOINT="${IDM_CHECKPOINT:-${ROOT}/results/robot/unused-idm.pt}"
GT_DEPTH_PRIMARY="${UAVFLOW_DEPTH_ROOT}"
if [[ -d "${UAVFLOW_DEPTH_ROOT}/hybrid" ]]; then
  # Canonical published layout produced by hf_dataset/extract.py.
  GT_DEPTH_PRIMARY="${UAVFLOW_DEPTH_ROOT}/hybrid"
  DEPTH_FALLBACKS="${DEPTH_FALLBACKS:-['${UAVFLOW_DEPTH_ROOT}/replay']}"
  INSTRUCTION_OVERRIDES="${INSTRUCTION_OVERRIDES:-${UAVFLOW_DEPTH_ROOT}/metadata/instruction_overrides.json}"
else
  # Local consolidated working layout.
  DEPTH_FALLBACKS="${DEPTH_FALLBACKS:-[]}"
  INSTRUCTION_OVERRIDES="${INSTRUCTION_OVERRIDES:-${UAVFLOW_DEPTH_ROOT}/instruction_overrides.json}"
fi
if [[ ! -f "${INSTRUCTION_OVERRIDES}" ]]; then
  INSTRUCTION_OVERRIDES=null
fi

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "optimizer: per_gpu_batch=${BATCH_SIZE} global_batch=$((BATCH_SIZE * NPROC)) deep_lr=${BASE_LR} predictor_lr=$(awk -v x="${BASE_LR}" 'BEGIN {printf "%.8g", x*0.2}') head_lr=$(awk -v x="${BASE_LR}" 'BEGIN {printf "%.8g", x*10}')"
echo "stage1_schedule=${STAGE1_LR_SCHEDULE} warmup=${STAGE1_WARMUP_STEPS} min_ratio=${STAGE1_MIN_LR_RATIO}"
echo "stage2_stop_schedule=${STAGE2_LR_SCHEDULE} warmup=${STAGE2_WARMUP_STEPS} min_ratio=${STAGE2_MIN_LR_RATIO}"

COMMON_OVERRIDES=(
  --set "stage1.da3_checkpoint=${DA3_CHECKPOINT}"
  --set "stage1.idm_checkpoint=${IDM_CHECKPOINT}"
  --set "stage1.action_stats_dir=${ACTION_STATS_DIR}"
  --set "stage1.qwen_model=${QWEN_MODEL}"
  --set "stage1.t5_model=${T5_MODEL}"
  --set "dataset.parquet_root=${UAVFLOW_SIM_ROOT}"
  --set "dataset.gt_depth_root=${GT_DEPTH_PRIMARY}"
  --set "dataset.gt_depth_fallback_roots=${DEPTH_FALLBACKS}"
  --set "dataset.instruction_overrides_path=${INSTRUCTION_OVERRIDES}"
  --set "loss.depth_fixed_scale_meters=${DEPTH_FIXED_SCALE_METERS}"
  --set "training.lr=${BASE_LR}"
  --set "training.batch_size=${BATCH_SIZE}"
  --set "training.num_workers=${NUM_WORKERS}"
  --set "training.save_latest_every=${CHECKPOINT_EVERY_STEPS}"
)
if [[ -n "${MAX_TRAJECTORIES:-}" ]]; then
  COMMON_OVERRIDES+=(--set "dataset.max_trajectories=${MAX_TRAJECTORIES}")
fi

# The lock makes a shared split safe when a compute-rich matrix is sharded
# across several workers/nodes on a shared filesystem.
(
  flock 9
  if [[ ! -s "${SPLIT_FILE}" ]]; then
    echo "[$(date '+%F %T')] preparing immutable split: ${SPLIT_FILE}"
    "${PYTHON_BIN}" "${ROOT}/experiments/uavflow_remote_ablation/prepare_split.py" \
      --config "${CONFIG}" --output "${SPLIT_FILE}" \
      "${COMMON_OVERRIDES[@]}"
  fi
) 9>"${SPLIT_FILE}.lock"
COMMON_OVERRIDES+=(--set "dataset.split_file=${SPLIT_FILE}")

variant_overrides() {
  local id="$1"
  case "${id}" in
    B0|S1COS) ;;
    P1)
      printf '%s\n' 'model.use_pose_history=true'
      ;;
    L1)
      printf '%s\n' 'stage1.text_encoder_type=t5' 'model.language_len=77'
      ;;
    D1)
      printf '%s\n' 'loss.depth_scale_mode=fixed_metric'
      ;;
    D2)
      printf '%s\n' 'loss.depth_scale_mode=scale_separated'
      ;;
    D1LOG)
      printf '%s\n' \
        'loss.depth_scale_mode=fixed_metric' \
        'loss.depth_linear_weight=0.5' \
        'loss.depth_log_weight=0.5' \
        'loss.depth_log_epsilon_normalized=0.001' \
        'loss.depth_grad_weight=1.0'
      ;;
    W3|W5|W10)
      printf '%s\n' "loss.depth_semantic_weight=${id#W}"
      ;;
    H0)
      printf '%s\n' 'loss.depth_target_mode=current'
      ;;
    HB)
      printf '%s\n' 'loss.depth_target_mode=both'
      ;;
    CA1_HB)
      printf '%s\n' 'loss.depth_target_mode=both' 'model.current_geometry_action_enabled=true'
      ;;
    F3|F7|F10)
      local k="${id#F}"
      printf '%s\n' \
        "dataset.visual_anchor_stride=${k}" \
        "dataset.chunk_size=${k}" \
        "dataset.endpoint_absorbing_max_starts=${k}" \
        "model.action_chunk_size=${k}"
      ;;
    M1)
      # Five physical frames are needed for H=4 plus its next K=5 endpoint.
      printf '%s\n' \
        'dataset.future_steps=4' \
        'model.context_lengths=[1,2,3,4]' \
        'model.context_weights=[0.25,0.25,0.25,0.25]'
      ;;
    C1_PL)
      # Tests whether a direct state token changes the T5-versus-VLM result.
      printf '%s\n' \
        'model.use_pose_history=true' \
        'stage1.text_encoder_type=t5' \
        'model.language_len=77'
      ;;
    C2_D2HB)
      # Shape/scale separation may be most useful when current geometry also
      # anchors the harder endpoint-depth prediction.
      printf '%s\n' \
        'loss.depth_scale_mode=scale_separated' \
        'loss.depth_target_mode=both'
      ;;
    C3_W3HB)
      # Mild dynamic emphasis plus current/future supervision tests whether
      # sparse-object blur is a weighting problem after stabilizing geometry.
      printf '%s\n' \
        'loss.depth_semantic_weight=3' \
        'loss.depth_target_mode=both'
      ;;
    C4_F3W3)
      # A nearer action/geometry endpoint should reduce multimodal blur; the
      # dynamic multiplier tests whether residual small-object error remains.
      printf '%s\n' \
        'dataset.visual_anchor_stride=3' \
        'dataset.chunk_size=3' \
        'dataset.endpoint_absorbing_max_starts=3' \
        'model.action_chunk_size=3' \
        'loss.depth_semantic_weight=3'
      ;;
    C5_F10HB)
      # At the deliberately hard long horizon, current-depth supervision tests
      # whether an observation reconstruction anchor prevents future blur.
      printf '%s\n' \
        'dataset.visual_anchor_stride=10' \
        'dataset.chunk_size=10' \
        'dataset.endpoint_absorbing_max_starts=10' \
        'model.action_chunk_size=10' \
        'loss.depth_target_mode=both'
      ;;
    *)
      echo "Unknown run ID: ${id}" >&2
      return 2
      ;;
  esac
}

latest_checkpoint() {
  local directory="$1"
  if [[ -s "${directory}/last.pt" ]]; then
    printf '%s\n' "${directory}/last.pt"
    return 0
  fi
  find "${directory}" -maxdepth 1 -type f -name 'ckpt_*.pt' -printf '%p\n' \
    2>/dev/null | sort -V | tail -n 1
}

run_stage() {
  local id="$1" stage="$2" init_checkpoint="$3"
  local output_id="${RUN_LABEL:-${id}}"
  local out="${OUTPUT_ROOT}/${output_id}/${stage}"
  mkdir -p "${out}"
  if [[ -s "${out}/_SUCCESS" ]]; then
    echo "[$(date '+%F %T')] skip completed ${id}/${stage}"
    return 0
  fi

  local variant=()
  while IFS= read -r value; do
    [[ -n "${value}" ]] && variant+=(--set "${value}")
  done < <(variant_overrides "${id}")
  if [[ -n "${EXTRA_OVERRIDES_FILE}" ]]; then
    [[ -f "${EXTRA_OVERRIDES_FILE}" ]] || {
      echo "Missing EXTRA_OVERRIDES_FILE=${EXTRA_OVERRIDES_FILE}" >&2
      return 2
    }
    while IFS= read -r value; do
      [[ -z "${value}" || "${value}" == \#* ]] && continue
      variant+=(--set "${value}")
    done < "${EXTRA_OVERRIDES_FILE}"
  fi

  local stage_args=()
  local checkpoint_args=()
  if [[ "${stage}" == "stage2_stop" ]]; then
    checkpoint_args+=(--init-checkpoint "${init_checkpoint}")
    stage_args+=(
      --set 'model.stop_head_enabled=true'
      --set 'model.stop_head_mode=legacy_action_token'
      --set 'loss.stop_weight=1.0'
      --set 'loss.stop_pos_weight=5.0'
      --set "training.lr_schedule=${STAGE2_LR_SCHEDULE}"
      --set "training.warmup_steps=${STAGE2_WARMUP_STEPS}"
      --set "training.min_lr_ratio=${STAGE2_MIN_LR_RATIO}"
      --set "training.max_epochs=${STAGE2_EPOCHS}"
    )
  else
    local stage1_schedule="${STAGE1_LR_SCHEDULE}"
    local stage1_warmup="${STAGE1_WARMUP_STEPS}"
    local stage1_min_ratio="${STAGE1_MIN_LR_RATIO}"
    if [[ "${id}" == "S1COS" ]]; then
      stage1_schedule=cosine
      stage1_warmup=500
      stage1_min_ratio=0.01
    fi
    stage_args+=(
      --set "training.lr_schedule=${stage1_schedule}"
      --set "training.warmup_steps=${stage1_warmup}"
      --set "training.min_lr_ratio=${stage1_min_ratio}"
      --set "training.max_epochs=${STAGE1_EPOCHS}"
    )
  fi

  local resume=""
  resume="$(latest_checkpoint "${out}")"
  if [[ -n "${resume}" ]]; then
    # Exact optimizer/data-cursor continuation after interruption.
    checkpoint_args=(--resume "${resume}")
  fi

  local commit="unknown" dirty="unknown" split_sha="missing" depth_sha="missing"
  commit="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  if git -C "${ROOT}" diff --quiet --ignore-submodules HEAD -- 2>/dev/null && \
     git -C "${ROOT}" diff --cached --quiet --ignore-submodules HEAD -- 2>/dev/null; then
    dirty=false
  else
    dirty=true
  fi
  [[ -f "${SPLIT_FILE}" ]] && split_sha="$(sha256sum "${SPLIT_FILE}" | awk '{print $1}')"
  local depth_manifest="${UAVFLOW_DEPTH_ROOT}/metadata/episodes.csv"
  [[ -f "${depth_manifest}" ]] && depth_sha="$(sha256sum "${depth_manifest}" | awk '{print $1}')"
  printf '%s\n' \
    "id=${id}" "label=${output_id}" "stage=${stage}" "start=$(date -Is)" \
    "output=${out}" "host=$(hostname)" "project_commit=${commit}" \
    "project_dirty=${dirty}" "split_sha256=${split_sha}" \
    "depth_manifest_sha256=${depth_sha}" \
    "nproc=${NPROC}" "per_gpu_batch=${BATCH_SIZE}" \
    "global_batch=$((NPROC * BATCH_SIZE))" > "${out}/run_state.txt"
  echo "[$(date '+%F %T')] start ${id}/${stage} -> ${out}"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
    "${ROOT}/experiments/uavflow_predictor_idm/train.py" \
    --config "${CONFIG}" \
    "${checkpoint_args[@]}" \
    "${COMMON_OVERRIDES[@]}" \
    "${variant[@]}" \
    --set "training.results_dir=${out}" \
    "${stage_args[@]}" \
    2>&1 | tee -a "${out}/console.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ${status} -ne 0 ]]; then
    printf 'status=failed\nexit_code=%d\nend=%s\n' \
      "${status}" "$(date -Is)" >> "${out}/run_state.txt"
    return "${status}"
  fi
  local final_ckpt
  final_ckpt="$(latest_checkpoint "${out}")"
  [[ -s "${final_ckpt}" ]] || { echo "No final checkpoint for ${id}/${stage}" >&2; return 3; }
  printf '%s\n' "${final_ckpt}" > "${out}/_SUCCESS"
  printf 'status=complete\ncheckpoint=%s\nend=%s\n' \
    "${final_ckpt}" "$(date -Is)" >> "${out}/run_state.txt"
}

IFS=',' read -r -a IDS <<< "${RUN_IDS}"
for id in "${IDS[@]}"; do
  id="${id//[[:space:]]/}"
  [[ -n "${id}" ]] || continue
  run_stage "${id}" stage1 ""
  if [[ "${RUN_STAGE2}" == "1" ]]; then
    output_id="${RUN_LABEL:-${id}}"
    stage1_ckpt="$(<"${OUTPUT_ROOT}/${output_id}/stage1/_SUCCESS")"
    run_stage "${id}" stage2_stop "${stage1_ckpt}"
  fi
done

echo "[$(date '+%F %T')] requested ablations complete: ${RUN_IDS}"
