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
| `03_rgbwarp.py` | Qwen proposal + predicted Current depth -> RGB warp -> shared Future DA3 + gated residual | Future patches read geometrically corresponding Current patches after blocks 19/27/33/39 | **all blocks 0–39**, rank64/alpha128 |

All original DA3 weights and DPT stay frozen. Both correction variants have
Current-depth supervision and enabled Current Bank reads, but these reads are
not identical operations: old Action->Current CA is not the new patch
correspondence CA. The GAM baseline is not silently given another Current branch.
Qwen uses all-linear LoRA rank32/alpha16/dropout0, native causal attention,
last-layer five slot hidden states and decoder-only learned step embeddings.

Both correction entries use independent Proposal/Correction heads and
`final = proposal + sigmoid(g) * delta`. Gate is one learned global scalar,
initialized to0.1, not a confidence predictor. No tanh or fixed residual upper
bound is applied. It is separate from Predictor feature-residual/CA gates.
Old ungated or bounded checkpoints cannot silently resume as this new topology;
the gate is saved in architecture state and optimized with the action-head group.

RGB-warp now uses radius-1.5-pixel, nearest-K8 depth-sorted alpha-over, without
RGB mass normalization. Its missing-coverage embedding receives detached
`1-opacity`; patch correspondence follows positive alpha-over contributors.
DA3 action tokens are only projected VLM hidden states, with no proposal-value
embedding. Candidate 2 already lacked that numeric embedding and is unchanged.
See [dual_stream_rgbwarp.md](dual_stream_rgbwarp.md) for checkpoint compatibility
and the distinction between gradient stability and navigation quality.

All three use H1, visual stride5, five local-yaw action increments, target t+5.
Depth is **fixed metric**: GT metres / 100; prediction *100 is metres, with no
per-frame/first-frame scale normalization. Current/Future losses in the two
correction models use the same fixed physical scale.

Aligned loss weights: Action3, Depth3; only GAM adds Feature1.

- GAM: `3*final_action_L1 + 1*shallow_feature_MSE + 3*future_depth`.
- Both correction variants:
  `3*(proposal_L1 + final_L1)/2 + 3*(current_depth + future_depth)/2`.
  Proposal/final each receive1.5; Current/Future depth each receive1.5.
  Neither correction has a feature loss, including the old Predictor model.
- Every depth branch uses linear L1 + unmodified GAM finite-difference L1
  gradient, both with internal weight1. RGB MSE metrics are diagnostic only.
  No log loss, semantic/dynamic reweighting, uncertainty or Stop loss is enabled.
- Gated residual merge is unchanged. Changed loss definitions/weights are not
  silently accepted as an exact resume of a previous loss configuration.

Default **5 epochs**, constant LR, no warmup; batch4 per process, AdamW with
weight_decay0, grad_clip1. Predictor/Qwen/state LR=1e-5; DA3 LoRA=5e-5;
action heads and new warp/read modules=5e-4. No model has been automatically
started by this release update. Full-model GPU/NPU memory must be measured;
old R1 benchmark measurements do not certify these new variants.

## Pose Token train/test fix

New correction runs select `dataset.openvla_prompt_pose_mode=evaluator_yaw4d`.
They inject an MLP-encoded numeric state at `<|uav_state|>` before the five
action slots; numbers are not also printed into the prompt. New maintained
entries enable a seven-dimensional state:
`[x_m,y_m,z_m,sin(relative_yaw),cos(relative_yaw),signed_accumulated_yaw_rad,path_m]`.
XYZ is relative to first-frame yaw-only heading. Accumulated yaw sums wrapped
**adjacent observed** yaw increments without wrapping the final sum. Path is
the sum of observed three-dimensional displacement magnitudes, not endpoint
distance. Qwen's state MLP receives these channels divided by
`[10,10,10,1,1,2*pi,10]`; no clipping is applied. This is still one numeric state
token, not seven extra action slots or text fields.
Training Dataset and inference use the same `episode_pose_progress` helper.
Raw pose order is `[x_cm,y_cm,z_cm,roll_deg,yaw_deg,pitch_deg]`.
Progress uses the entire observed prefix, never future/GT actions or planned
proposal chunks. Repeated stationary absorbing frames add no motion. Online
services must pass `observed_pose_history` (one episode-start-to-now raw history
per batch item), or maintain and pass a canonical seven-dimensional state;
reset history at each new episode. Two endpoint poses alone are insufficient.
Yaw changes between consecutive observations must be less than 180 degrees.

Historical Pure-Qwen Pose Token runs used `preprocessed` full-first-rotation
state. Changing a launcher does **not** repair their already learned weights.
Legacy checkpoints retain their saved state convention. Do not claim an exact
resume into the new yaw4d convention; train a new run, or explicitly transfer
weights as initialization with the changed convention acknowledged. Inference
rejects raw yaw4d conversion for a legacy preprocessed checkpoint.

GAM's direct numeric-pose path remains a Predictor input, not a Qwen token.
It now appends accumulated yaw/(2*pi) and path/10m to its existing five-channel
q01/q99-normalized xyz and unchanged sin/cos. The F0 progress channels are zero.
The five-dimensional pose normalizer and auxiliary pose targets are unchanged.
Legacy configurations default progress off and retain their old 4D Qwen/5D GAM
state inputs. New entries enable `dataset.pose_progress_enabled`,
`model.pose_progress_enabled`, and (correction only)
`stage1.qwen_pose_progress_enabled`. Changed dimensions require a new run, not
an exact resume of an old checkpoint; strict resume checks these settings.

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
    observed_pose_history=raw_histories_from_episode_start_to_now)
actions = output["actions_meters_radians"]  # [B,5,4]
```

Images are float[0,1], shape `[B,1,3,224,224]`. New policies need no future GT
image/depth/action. Loading old weights does not change their pose semantics.
UE service integration and closed-loop accelerator runs remain separate checks.
