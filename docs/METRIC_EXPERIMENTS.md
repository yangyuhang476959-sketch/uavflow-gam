# Three maintained metric-depth experiments

Environment and data preparation remain in [NPU_SETUP.md](../NPU_SETUP.md)
(Ascend) and [REMOTE_TRAINING.md](REMOTE_TRAINING.md) (NVIDIA). No installer,
Driver/CANN selection, project dependency pins, isolated Triton or FLA setup
is removed or changed by this release cleanup.

## Architecture and defaults

| Entry | Model | Current geometry read | DA3 LoRA |
|---|---|---|---|
| `01_gam.py` | Local GAM/T5 + numeric pose baseline; F0/Ft context; future Predictor; one future deep view | None: preserve baseline | blocks 13–39, rank64/alpha128 |
| `02_current_bank.py` | Native Qwen five-slot proposal + GAM future Predictor + residual correction | Future Action slots read Current patches by per-output-layer CA | blocks 13–39, rank64/alpha128 |
| `03_rgbwarp.py` | Qwen proposal + predicted Current depth -> RGB warp -> shared Future DA3 + bounded residual | Future patches read geometrically corresponding Current patches after blocks 19/27/33/39 | **all blocks 0–39**, rank64/alpha128 |

All original DA3 weights and DPT stay frozen. Both correction variants have
Current-depth supervision and enabled Current Bank reads, but these reads are
not identical operations: old Action->Current CA is not the new patch
correspondence CA. The GAM baseline is not silently given another Current branch.
Qwen uses all-linear LoRA rank32/alpha16/dropout0, native causal attention,
last-layer five slot hidden states and decoder-only learned step embeddings.

All three use H1, visual stride5, five local-yaw action increments, target t+5.
Depth is **fixed metric**: GT metres / 100; prediction *100 is metres, with no
per-frame/first-frame scale normalization. Current/Future losses in the two
correction models use the same fixed physical scale.

Losses are deliberately not relabelled as identical:

- GAM: `3*final_action_L1 + 1*shallow_feature_MSE + 3*future_depth`.
- Old Current Bank correction: existing action mixture
  `3*(proposal_L1/3 + 2*final_L1/3) + 1*feature_MSE + 3*depth`;
  depth is the existing average of Current and Future branch losses.
- RGB warp: `3*(proposal_L1 + final_L1) + current_depth + future_depth`;
  no feature loss.
- GAM/old correction retain linear depth L1 + unmodified GAM finite-difference
  L1 gradient. RGB warp uses depth MSE + that same gradient. No log loss,
  semantic/dynamic reweighting, uncertainty or Stop loss is enabled.

Default **5 epochs**, constant LR, no warmup; batch4 per process, AdamW with
weight_decay0, grad_clip1. Predictor/Qwen/state LR=1e-5; DA3 LoRA=5e-5;
action heads and new warp/read modules=5e-4. No model has been automatically
started by this release update. Full-model GPU/NPU memory must be measured;
old R1 benchmark measurements do not certify these new variants.

## Pose Token train/test fix

New correction runs select `dataset.openvla_prompt_pose_mode=evaluator_yaw4d`.
They inject an MLP-encoded numeric state at `<|uav_state|>` before the five
action slots; numbers are not also printed into the prompt. State is xyz in
centimetres relative to first-frame yaw-only heading, plus wrapped yaw degrees.
Training Dataset and inference use the same `episode_yaw4d_state` helper.
Raw pose order is `[x_cm,y_cm,z_cm,roll_deg,yaw_deg,pitch_deg]`.

Historical Pure-Qwen Pose Token runs used `preprocessed` full-first-rotation
state. Changing a launcher does **not** repair their already learned weights.
Legacy checkpoints retain their saved state convention. Do not claim an exact
resume into the new yaw4d convention; train a new run, or explicitly transfer
weights as initialization with the changed convention acknowledged. Inference
rejects raw yaw4d conversion for a legacy preprocessed checkpoint.

GAM's existing direct numeric-pose history is a different input from the Qwen
Pose Token. Its baseline path is retained, not replaced by a Qwen state input.

## Data split and augmentation

Use unchanged official Sim shards plus replay/hybrid depth sidecars. Both are
joined by episode/frame ID. Default 95/5 class-stratified episode train/val split,
seed42, is **our validation protocol**, not a claimed official OpenVLA split.
An optional `SPLIT_FILE` freezes explicit IDs across runs. No episode can occur
in both train and validation. The separate official273 closed-loop tasks are
not this 5% validation set. No extra start/end repeats or artificial absorbing
rebalance; retain natural terminal/absorbing windows.

## Commands

Use the activated platform environment and source paths from NPU_SETUP.md:

```bash
cd /path/to/uavflow-gam
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_SIM_ROOT=/path/to/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=/path/to/UAV-Flow-Sim-Depth
export MODEL_ROOT=/path/to/checkpoints
export OUTPUT_ROOT=/path/to/training-output
# Optional DA3_CHECKPOINT, QWEN_MODEL, T5_MODEL, ACTION_STATS_DIR, SPLIT_FILE.

# Inspect without loading model weights:
python experiments/metric_jobs/01_gam.py --print-config
python experiments/metric_jobs/02_current_bank.py --print-config
python experiments/metric_jobs/03_rgbwarp.py --print-config

# Each is an independent single-process job:
python experiments/metric_jobs/01_gam.py
python experiments/metric_jobs/02_current_bank.py
python experiments/metric_jobs/03_rgbwarp.py

# One experiment across an eight-device node, global batch32:
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  experiments/metric_jobs/02_current_bank.py --set training.batch_size=4

# Real smoke: use a separate output directory; not a scientific run.
OUTPUT_ROOT="$OUTPUT_ROOT/smoke" python experiments/metric_jobs/02_current_bank.py \
  --max-trajectories 20 --set training.max_epochs=1 --set training.batch_size=1

# Explicit exact resume; same data/world size/batch/architecture:
python experiments/metric_jobs/02_current_bank.py --resume /path/to/checkpoint.pt
python experiments/metric_jobs/02_current_bank.py --resume /path/to/checkpoint.pt \
  --eval-only --set training.eval_max_batches=0
```

Set `UAVFLOW_ACCELERATOR=npu`, source the selected CANN environment, and retain
the isolated Triton/FLA PYTHONPATH for Ascend as documented. These entry points
do not install/repair environments or select devices for you. The old
`run_experiment R1` backend remains only because the existing environment
benchmark depends on it; it is not one of the three production jobs.

Epoch checkpoints, optimizer/scaler/RNG/data-cursor restoration and AMP remain
in the shared trainer. Validation normally runs every2000 steps (100 batches),
checkpoint archives every epoch. RGB-warp best selection uses final action L1;
old correction retains its existing weighted action selection. Changing a depth
definition is a **new experiment**, not an exact continuation of a relative-depth run.

## Inference

The correction checkpoint loader uses its saved topology and state convention:

```python
from experiments.uavflow_support_aware.inference import load_policy
policy = load_policy("/path/to/checkpoint.pt", device="cuda:0")
output = policy.predict(current_rgb, instructions,
    reference_images=first_rgb,  # old Current Bank model needs F0; RGB warp ignores it
    first_sim_pose=first_raw_pose_batch, current_sim_pose=current_raw_pose_batch)
actions = output["actions_meters_radians"]  # [B,5,4]
```

Images are float[0,1], shape `[B,1,3,224,224]`. New policies need no future GT
image/depth/action. Loading old weights does not change their pose semantics.
UE service integration and closed-loop accelerator runs remain separate checks.
