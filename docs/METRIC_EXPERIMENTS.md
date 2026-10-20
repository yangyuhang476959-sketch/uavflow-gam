# Metric-depth experiment handoff

Environment and data preparation remain in [NPU_SETUP.md](../NPU_SETUP.md)
(Ascend) and [REMOTE_TRAINING.md](REMOTE_TRAINING.md) (NVIDIA). No installer,
Driver/CANN selection, project dependency pins, isolated Triton or FLA setup
is removed or changed by this release cleanup.

## Architecture and defaults

The five formal experiments are represented by executable jobs. Existing
Candidate 3/4 names and entry filenames are retained for compatibility:

| ID | Entry | Design | DA3 trainable scope |
|---|---|---|---|
| 1 | `experiments/uavflow_qwen_pure/train.py` | Pure Qwen, yaw-only numeric Pose Token, five Qwen action slots; no GAM/DA3 | None |
| 2 | `experiments/metric_jobs/01_gam.py` | GAM/T5 metric-depth baseline | Blocks 13–39, rank64/alpha128 LoRA |
| Current Bank | `experiments/metric_jobs/02_current_bank.py` | VLA proposal -> GAM Future Predictor -> DA3 correction; Action reads Current Geometry Bank through per-layer CA | Blocks 13–39, rank64/alpha128 LoRA |
| 3 | `experiments/metric_jobs/03_rgbwarp.py` | Qwen proposal -> Current metric depth -> bilinear RGB warp -> Future DA3 with proposal Action Tokens -> gated residual | All blocks 0–39, rank64/alpha128 LoRA |
| 4 | `experiments/metric_jobs/04_rgbwarp_no_action.py` | Same as 3, replacing VLM-derived Future DA3 Action Tokens with five randomly initialized learned Action Tokens | All blocks 0–39, rank64/alpha128 LoRA |

Current Bank is a formal experiment alongside the other four. Prioritize
Candidates 3 and 4, then run the remaining comparisons. Original DA3 weights
and DPT remain frozen; only listed
LoRA adapters train. The Qwen Pure model uses all-linear LoRA rank32/alpha16,
native causal attention, final-layer five action-placeholder hidden states and
decoder-only learned action-step embeddings. Its numeric Pose Token contains
the evaluator's episode-first-frame-relative yaw-only XYZ in centimetres and
wrapped yaw in degrees; the pose is not also written as text.

Candidate 4 still lets the Proposal affect Future geometry through the
differentiable RGB warp. It removes the Proposal Action Tokens from the Future
DA3 sequence and inserts five randomly initialized learned Action Tokens at
the same deep boundary used by Candidate 3. Real future images,
depth and actions are not policy inputs (future targets remain training/eval
supervision only).

These five Candidate 4 Action Tokens are trainable parameters initialized once
from `N(0, 0.02)`, matching GAM's learned action-slot initialization rule. They
interact with Future geometry inside DA3 exactly as in Candidate 3 and are
decoded by the same correction head. They are not redrawn on each forward.
The Qwen-derived proposal and its direct action supervision remain unchanged.
This isolates VLM-derived versus randomly initialized correction Action Tokens;
there is no extra post-DA3 readout attention. The historical entry/config name
`rgbwarp_no_action` now means no VLM action seed, not an action-free DA3 pass.
Checkpoints of the previous post-DA3-query topology cannot be resumed into this
changed topology; start a new Candidate 4 run.

Both correction entries use independent Proposal/Correction heads and
`final = proposal + sigmoid(g) * delta`. Gate is one learned global scalar,
initialized to0.1, not a confidence predictor. No tanh or fixed residual upper
bound is applied. It is separate from Predictor feature-residual/CA gates.
Old ungated or bounded checkpoints cannot silently resume as this new topology;
the gate is saved in architecture state and optimized with the action-head group.

New RGB-warp runs default to **bilinear forward splatting**, matching the
locally trained full-LoRA Bilinear experiment. The missing-coverage embedding
uses the binary warp-valid mask, and patch correspondence follows the splat
contributors. Alpha-over remains available with
`--set model.rgbwarp_correction.warp_mode=alpha` (radius1.5, nearest-K8); its
missing-coverage embedding uses detached `1-opacity`. Saved checkpoint modes
remain authoritative; changing the new-run default does not convert checkpoints.
DA3 action tokens are only projected VLM hidden states, with no proposal-value
embedding. Candidate 2 already lacked that numeric embedding and is unchanged.
See [dual_stream_rgbwarp.md](dual_stream_rgbwarp.md) for checkpoint compatibility
and the distinction between gradient stability and navigation quality.

All five use H1, visual stride5, five local-yaw action increments, target t+5.
Depth is **fixed metric**: GT metres / 100; prediction *100 is metres, with no
per-frame/first-frame scale normalization. Current/Future losses in the two
correction models use the same fixed physical scale.

Aligned loss weights: Action3, Depth3; only GAM adds Feature1.

- GAM: `3*final_action_L1 + 1*shallow_feature_MSE + 3*future_depth`.
- Both correction variants:
  `3*(proposal_L1 + final_L1)/2 + 3*(current_depth + future_depth)/2`.
  Proposal/final each receive1.5; Current/Future depth each receive1.5.
  Neither correction has a feature loss, including the old Predictor model.
- Every depth branch uses `linear L1 + log-depth L1 + unmodified GAM
  finite-difference L1 gradient`, averaged with internal weights **1/3 each**.
  The outer Depth weight stays3: `3*(linear + log + gradient)/3`, like splitting
  the action budget between proposal/final. This is equal coefficient weighting,
  not an adaptive guarantee of equal observed loss values. Gradient is still computed on
  linear normalized depth, not log depth. Log uses natural logarithms of
  prediction and GT metres/100, clamped below at the existing normalized
  epsilon `1e-6`. Current/Future both use this same recipe; each term is logged.
  RGB MSE metrics are diagnostic only. No semantic/dynamic reweighting,
  uncertainty or Stop loss is enabled. The separate local CPU warp comparison
  explicitly keeps log weight0 and is not changed to this new recipe.
- Gated residual merge is unchanged. Changed loss definitions/weights are not
  silently accepted as an exact resume of a previous loss configuration.

Repository defaults are **10 epochs**, constant LR, no warmup; batch4 per process, AdamW with
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
python experiments/uavflow_qwen_pure/train.py --print-config
python experiments/metric_jobs/01_gam.py --print-config
python experiments/metric_jobs/02_current_bank.py --print-config
python experiments/metric_jobs/03_rgbwarp.py --print-config
python experiments/metric_jobs/04_rgbwarp_no_action.py --print-config

# Run each as an independent eight-device job: per-device batch4, global batch32.
# All five use constant LR (no decay) and zero warmup for all 10 epochs.
torchrun --standalone --nproc_per_node=8 \
  experiments/uavflow_qwen_pure/train.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 \
  experiments/metric_jobs/01_gam.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 \
  experiments/metric_jobs/02_current_bank.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 \
  experiments/metric_jobs/03_rgbwarp.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4 \
  --set model.rgbwarp_correction.warp_mode=bilinear

torchrun --standalone --nproc_per_node=8 \
  experiments/metric_jobs/04_rgbwarp_no_action.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4 \
  --set model.rgbwarp_correction.warp_mode=bilinear

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
benchmark depends on it; it is not one of these five formal experiments.

GAM and RGB-warp jobs use the shared trainer's optimizer/scaler/RNG/cursor
resume. Pure-Qwen has its own optimizer and sampler-cursor checkpoints (BF16,
no GradScaler); worker-side stochastic image crops may differ after resume.
Validation normally runs every2000 steps (100 batches) in the shared trainer;
checkpoint archives every epoch. RGB-warp best selection uses final action L1;
old correction retains its existing weighted action selection. Changing a depth
definition is a **new experiment**, not an exact continuation of a relative-depth run.

## Handoff scope and priority

Prioritize 3 and 4, as requested. The 10-epoch handoff explicitly overrides
the maintained five-epoch defaults. LR is constant throughout; warmup is zero.
No cosine scheduler is used for these five jobs.

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


## Portable runtime recovery and gradient accumulation

All metric runners and the standalone Pure-Qwen trainer accept
`--set training.gradient_accumulation_steps=N` (default `1`, positive integer).
`training.batch_size` remains the **per-device microbatch**; the effective
optimizer batch is `batch_size * world_size * gradient_accumulation_steps`.
Existing remote batch settings are unchanged. For example, total batch32 on
2 devices can use per-device16/accumulation1, 8/2, or 4/4:

```bash
python -m torch.distributed.run --nproc_per_node=2 experiments/metric_jobs/03_rgbwarp.py \
  --set training.batch_size=8 --set training.gradient_accumulation_steps=2
```

The same two overrides work with `01_gam.py`, `02_current_bank.py`,
`04_rgbwarp_no_action.py`, and
`python -m experiments.uavflow_qwen_pure.train`. They are independent of the
accelerator backend. LR scheduling, validation/save intervals, and checkpoint
`step`/`batch` cursors count **optimizer updates**, not microbatches. Startup logs
print microbatch, accumulation, and effective global batch. Pure retains its
partial final batch and weights unequal microbatches by valid action elements;
the shared trainer drops incomplete accumulation groups.

Resume retains optimizer/scaler state and seeks the sampler by effective
per-device batch. Compatibility checks reject changed batch/accumulation
settings; do not change these on resume without a verified cursor conversion.
Older checkpoints without an accumulation setting mean accumulation1.
The exact requested final-step checkpoint preserves the current epoch and
next-update cursor rather than incorrectly advancing to the next epoch.

Correction metric runners enable configurable DDP unused-parameter detection
for optional/disabled heads. RGB-warp depth validation accepts both the released
equal-weight linear/log/gradient recipe and the legacy no-log recipe. Explicit
DA3 LoRA start blocks are honored in training and inference; omission defaults
to block0 for RGB-warp. Camera pose composition/inversion uses FP32 even under
BF16 autocast, with action gradients preserved. Predictor-free Warp startup
logging no longer assumes `GAMFuturePredictor` attributes. Inference freezes all
base DA3 parameters before enabling trained LoRA adapters, matching training.

Optional `training.warp_health.enabled=true` records dual-rank coverage/depth
health and stops before an invalid update. `patience` defaults to100 consecutive
updates (recovery resets the counters), `minimum_coverage` to0.001, and
`collapse_fraction` to0.99. Counters are saved/restored in checkpoints. This
checks persistent empty warp or almost-all-zero current depth; it does not
classify every temporary scale error as collapse. It is off unless enabled.
No machine-specific shutdown, paths, batch tuning, or CUDA package changes are
part of these portable fixes.
