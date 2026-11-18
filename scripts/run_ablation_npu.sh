#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: NPROC=3 EPOCHS=15 bash scripts/run_ablation_npu.sh CONFIG [CHECKPOINT]" >&2
  exit 2
fi
config_path=$1
resume_path=${2:-}
export NPROC=${NPROC:-3}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export EPOCHS=${EPOCHS:-}
export UAVFLOW_ACCELERATOR=npu
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! $NPROC =~ ^[1-9][0-9]*$ || ! $NNODES =~ ^[1-9][0-9]*$ || ! $NODE_RANK =~ ^[0-9]+$ ]]; then
  echo 'NPROC and NNODES must be positive integers; NODE_RANK must be nonnegative' >&2; exit 2
fi
if (( NODE_RANK >= NNODES )); then echo 'NODE_RANK must be less than NNODES' >&2; exit 2; fi
export TOTAL_NPUS=$((NPROC * NNODES))
case "$TOTAL_NPUS" in 1|2|3|4|6|8|12|24) ;; *) echo 'Total NPUs must be 1, 2, 3, 4, 6, 8, 12 or 24 to preserve global batch 24' >&2; exit 2;; esac
if (( NNODES > 1 )) && [[ -z ${MASTER_ADDR:-} ]]; then
  echo 'Set MASTER_ADDR to the address of node 0 for multi-node training' >&2; exit 2
fi
if [[ -n $EPOCHS && ! $EPOCHS =~ ^[1-9][0-9]*$ ]]; then echo 'EPOCHS must be a positive integer' >&2; exit 2; fi
resolved_config=$(mktemp "${TMPDIR:-/tmp}/uavflow-npu-config.XXXXXXXX.yaml")
trap 'rm -- "$resolved_config"' EXIT
python - "$config_path" "$resolved_config" <<'PY'
import os
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
world = int(os.environ['TOTAL_NPUS'])
if os.environ['EPOCHS']:
    cfg.training.max_epochs = int(os.environ['EPOCHS'])
cfg.training.batch_size = 8 if world == 1 else 24 // world
cfg.training.gradient_accumulation_steps = 3 if world == 1 else 1
cfg.training.allow_warp24_two_gpu_repartition = world == 2
cfg.training.allow_warp24_multi_device_repartition = world in (4, 6, 8, 12, 24)
cfg.training.eval_batch_size = 'auto'
OmegaConf.save(cfg, sys.argv[2])
print(f'Devices={world}, per-device microbatch={cfg.training.batch_size}, '
      f'accumulation={cfg.training.gradient_accumulation_steps}, global batch=24, '
      f'epochs={cfg.training.max_epochs}', flush=True)
PY
args=(--config "$resolved_config")
if [[ -n $resume_path ]]; then args+=(--resume "$resume_path"); fi
if (( NNODES == 1 )); then
  torchrun --standalone --nproc_per_node="$NPROC" -m experiments.uavflow_predictor_idm.train "${args[@]}"
else
  torchrun --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
    -m experiments.uavflow_predictor_idm.train "${args[@]}"
fi
