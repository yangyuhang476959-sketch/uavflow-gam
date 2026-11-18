#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: NPROC=3 EPOCHS=15 bash scripts/run_ablation_npu.sh CONFIG [CHECKPOINT]" >&2
  exit 2
fi
config_path=$1
resume_path=${2:-}
export NPROC=${NPROC:-3}
export EPOCHS=${EPOCHS:-}
export UAVFLOW_ACCELERATOR=npu
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
case "$NPROC" in 1|2|3|4|6|8) ;; *) echo 'NPROC must be 1, 2, 3, 4, 6 or 8' >&2; exit 2;; esac
if [[ -n $EPOCHS && ! $EPOCHS =~ ^[1-9][0-9]*$ ]]; then echo 'EPOCHS must be a positive integer' >&2; exit 2; fi
resolved_config=$(mktemp "${TMPDIR:-/tmp}/uavflow-npu-config.XXXXXXXX.yaml")
trap 'rm -- "$resolved_config"' EXIT
python - "$config_path" "$resolved_config" <<'PY'
import os
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
world = int(os.environ['NPROC'])
if os.environ['EPOCHS']:
    cfg.training.max_epochs = int(os.environ['EPOCHS'])
cfg.training.batch_size = 8 if world == 1 else 24 // world
cfg.training.gradient_accumulation_steps = 3 if world == 1 else 1
cfg.training.allow_warp24_two_gpu_repartition = world == 2
cfg.training.allow_warp24_multi_device_repartition = world in (4, 6, 8)
cfg.training.eval_batch_size = 'auto'
OmegaConf.save(cfg, sys.argv[2])
print(f'Devices={world}, per-device microbatch={cfg.training.batch_size}, '
      f'accumulation={cfg.training.gradient_accumulation_steps}, global batch=24, '
      f'epochs={cfg.training.max_epochs}', flush=True)
PY
args=(--config "$resolved_config")
if [[ -n $resume_path ]]; then args+=(--resume "$resume_path"); fi
torchrun --standalone --nproc_per_node="$NPROC" -m experiments.uavflow_predictor_idm.train "${args[@]}"
