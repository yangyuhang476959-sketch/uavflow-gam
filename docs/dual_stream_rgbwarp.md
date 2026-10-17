# RGB-warp VLA-GAM correction (opt-in)

This is one of the three maintained experiments in
[METRIC_EXPERIMENTS.md](METRIC_EXPERIMENTS.md). Historical ten-cell jobs and
support/uncertainty variants are not the active handoff. Entry point:
`experiments.uavflow_predictor_idm.rgbwarp_train`. No formal training has been
started for this implementation; accelerator memory/performance and closed-loop
quality remain to be measured. CPU regression tests do not constitute NPU validation.

## Forward and trainable parameters

1. Current RGB + instruction + numeric pose token + five native action slots
   enter Qwen3.5-2B. Keep its causal attention and last-layer slot hidden states;
   no mean pooling, post-Qwen mixer, autoregressive action decoder or Future
   Predictor. Prompt is the existing action-question template, without a printed
   Current State field: pose is injected at `<|uav_state|>` before the five slots.
2. LN/Linear projects the five Qwen hidden states to DA3 width. The existing
   parallel action head decodes five normalized proposal actions. Decoder-only
   learned step encoding is retained.
3. Current RGB passes through DA3 once, **without action tokens**. Frozen DPT
   decodes current depth; requested deep layers also export current patch memory.
4. Denormalize all five proposal increments (m/m/m/rad), compose yaw-local rigid
   motions, convert body motion to camera motion by the configured extrinsic,
   and forward-splat current RGB using predicted current z-depth. Projection and
   accumulation are FP32; no GT action/depth drives the warp at train or test time.
   The default rasterizer is finite-disk **alpha-over**, radius 1.5 target pixels,
   nearest 8 points per pixel, gamma=1, black background. RGB is NOT divided by
   accumulated mass; weak coverage stays weak. This is SynSin-inspired PyTorch
   code, not a claimed exact reproduction of the author's renderer.
5. Feed warped RGB through the **same** DA3 again. A zero-initialized Conv14/14
   embeds detached `1-opacity` (continuous missing coverage) and adds it to patch
   embeddings before block 0. Coverage is geometric opacity, not learned confidence.
   Five action tokens are **only projected Qwen hidden states** (LN + Linear);
   no proposal-value embedding is added. Numeric proposal values drive the RGB
   warp, not the action-token initialization. Existing GAM packing inserts them at the deep boundary:
   `[camera, A1..A5, registers, patches]`; native spatial RoPE is retained.
6. Optional correspondence CA after blocks 19/27/33/39 updates **future patches
   only**, querying same-layer current patches (1536-D, not DPT's 3072-D concat).
   Its 256-D/8-head read is restricted by a binary future-patch/current-patch
   correspondence matrix. Empty rows receive exactly zero update; zero-initialized
   output projection preserves the initial stream. Actions see updates through
   subsequent native SA; the last-block CA directly affects depth, not an already
   computed last-layer action hidden.
   Effective front-to-back alpha weights affect warped RGB and visible pixel contributions,
   but are **not** used as soft CA priors: the patch matrix is binary (at least
   one retained source-pixel contribution). Among allowed Current patches the
   read uses ordinary learned QK softmax. There is no extra XYZ-relative bias or
   `log(warp_mass)` term. The detached mask provides geometric correspondence,
   not calibrated confidence or differentiable correspondence weights.
7. Frozen DPT reads future patches; the existing correction head reads five
   refined action tokens. Final normalized actions are `proposal + sigmoid(g)*z`.
   Correction head starts at zero, hence initial final actions equal proposal.

There is **one DA3 object, one shared full-backbone LoRA adapter and one DPT**,
not two independent models/adapters. Original patch embedding, registers/camera
parameters and DPT stay frozen. DA3 LoRA covers all Linear modules in blocks
0..39 (rank 64, alpha 128, dropout 0); Qwen uses existing all-linear LoRA
(rank 32, alpha 16, dropout 0), including its visual/linear-attention linears.
The mask projection, correspondence read, state encoder, projections and action
heads train. Both stream losses update the same DA3 LoRA.

The correction has no tanh, clipping or fixed per-axis bound. A single learned
scalar `sigmoid(g)` starts at 0.1 and scales all five steps/four dimensions.
Configure its initial value with `model.action_residual_gate_init=0.1`.
This is a residual-strength parameter, not sample-dependent uncertainty; since
the head output is unbounded, a sigmoid gate does not impose an action bound.
It is independent of the unused legacy Predictor feature-residual gate and
of the Current Bank CA gates. The action head and residual head stay independent.

## Loss and the fixed future target

```
L = 3 * (L1(proposal, GT actions) + L1(final, GT actions)) / 2
    + 3 * (D_current + D_future) / 2
D_branch = linear_L1(predicted_depth, GT_depth / 100) + GAM_grad
```

Action L1 uses the existing valid-action mask. Depth uses GT-valid pixels;
gradient uses valid endpoint pairs and the repository's **x-mean + y-mean L1
finite difference**, not tolerant/log gradient. Current/future depth losses are
averaged before the outer depth weight3. Proposal/final action losses also
split the outer action weight3 equally. No feature, log-depth, dynamic reweighting,
pose or stop loss. MSE metrics are diagnostic only, not part of the objective.

**GT future is the real t+5 frame and remains fixed.** It is NOT reprojected to
the proposal camera; warp holes are NOT removed from its supervision. This
follows the requested control-regularization design: when proposal differs from
GT motion, it is not pixel-aligned reconstruction of the proposal's hypothetical
future. Dynamic changes and newly revealed areas also require learned completion.

Both GT depth maps are divided by fixed 100 m; predicted current depth is
multiplied by 100 before geometry. This is the learned metric-output contract,
not proof that untouched DA3 outputs are already calibrated metric depths.
Early depth-scale errors can therefore make early warps inaccurate. No GT-based
per-sample scale fit is used to hide this train/test issue.

Future loss gradients reach **proposal and predicted current depth** via RGB
disk opacity and alpha-over weights. Frozen DPT remains in autograd. Pixel
indexing, depth ordering, mask input and correspondence topology are detached: the warp is piecewise
differentiable, not differentiable through appearance/disappearance decisions.
At identity motion, constant RGB regions or full occlusion, useful warp gradients
can vanish. LoRA reduces trainable degrees of freedom but does not mathematically
prevent learned geometry shortcuts; that must be checked experimentally.

## Geometry and pose contract

New entrypoints use the seven-channel observed pose/progress token documented
in [METRIC_EXPERIMENTS.md](METRIC_EXPERIMENTS.md#pose-token-traintest-fix).
Its accumulated yaw and travelled path condition Qwen, not correspondence CA.
Legacy four-channel pose-token checkpoints remain supported with progress off.

- z-depth, metres; horizontal FOV defaults to 90 degrees at 224x224, centered K.
  Supply actual `camera_intrinsics` in a batch/service when known.
- Default camera-to-body extrinsic is axis conversion only (CV right/down/forward
  to body forward/right/up), zero mounting offset. Override
  `heading_from_camera` when a calibrated offset is available; none is invented.
- Project each source point into a radius-1.5-pixel disk, retain the nearest
  8 points by target z, then use front-to-back alpha-over. Tied depths use stable
  source-index ordering. Holes stay black; no triangle, dilation or hole filling.
  No coverage normalization or numeric action-value embedding is used.
- Warp validity uses predicted current depth/motion only. Binary patch
  correspondence records positive effective alpha contributions, excluding fully
  occluded farther points; no GT future mask
  is used for attention or observation preprocessing.
- Numeric state is the seven-channel relative pose/progress documented above,
  from the shared `episode_pose_progress` helper. The legacy four-channel
  cm/degree state is retained only for progress-disabled checkpoints.
  Raw simulator pose order is
  `[x_cm,y_cm,z_cm,roll_deg,yaw_deg,pitch_deg]`.
- H=1; one current view; five actions; visual stride 5, hence target t+5.
  No F0 RGB, historical action or historical pose enters DA3.

## Execution

Use the existing NVIDIA/Ascend environment guide; no new installation workflow.
Set actual paths (replace all examples):

```bash
cd /path/to/UAVFlow-GAM-Release
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_SIM_ROOT=/path/to/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=/path/to/UAV-Flow-Sim-Depth
export MODEL_ROOT=/path/to/checkpoints
export OUTPUT_ROOT=/path/to/training-results
# MODEL_ROOT contains track4world_da3.pth and qwen3.5-2b/.
# Alternatively set DA3_CHECKPOINT and QWEN_MODEL individually.

python -m experiments.uavflow_predictor_idm.rgbwarp_train --print-config
python -m experiments.uavflow_predictor_idm.rgbwarp_train
# Same entry point under existing torchrun/DDP, e.g. two GPUs:
torchrun --standalone --nproc_per_node=2 --module \
  experiments.uavflow_predictor_idm.rgbwarp_train --set training.batch_size=4
```

Default: five epochs, constant LR, no warmup; per-process batch 4 (not a measured
memory guarantee). Qwen LR 1e-5, DA3 LoRA 5e-5, heads/new modules 5e-4. BF16/FP32
and GradScaler policy are inherited unchanged. Natural absorbing windows, no
extra start/end repetition, same stratified split/seed as the handoff matrix.
Validation every 2000 updates, maximum 100 batches; checkpoints each epoch;
`best_action.pt` uses **final/refined validation action L1**, while logs
separately report proposal, refined and averaged action loss. No stop phase.

```bash
# Exact resume: same topology, data, world size and batch/cursor configuration.
python -m experiments.uavflow_predictor_idm.rgbwarp_train --resume /path/to/checkpoint.pt
python -m experiments.uavflow_predictor_idm.rgbwarp_train \
  --resume /path/to/checkpoint.pt --eval-only

# Independent ablations; don't resume a changed topology as an exact resume.
python -m experiments.uavflow_predictor_idm.rgbwarp_train \
  --set model.rgbwarp_correction.use_correspondence_attention=false
python -m experiments.uavflow_predictor_idm.rgbwarp_train \
  --set model.rgbwarp_correction.use_pixel_binary_mask=false

# Optional compatible pure-Qwen numeric_token proposal initialization:
python -m experiments.uavflow_predictor_idm.rgbwarp_train \
  --proposal-checkpoint /path/to/compatible-pure-qwen-numeric-token.pt
```

Proposal transfer checks pose mode, trainable Qwen keys, head/projection shapes
and normalizer equivalence; a prompt-pose or numeric-head model is not silently
reinterpreted as numeric-token. Existing GAM/deep-only checkpoints are not exact
resumes of this topology. Standard optimizer/scaler/RNG/data-cursor saves remain
in the existing trainer.

Observation-only Python inference:

```python
from experiments.uavflow_support_aware.inference import load_policy
policy = load_policy("/path/to/checkpoint.pt", device="cuda:0")
out = policy.predict(current_rgb, instructions,
    observed_pose_history=raw_histories_from_episode_start_to_now)
actions = out["actions_meters_radians"]  # [B,5,4]
```

RGB must be `[B,1,3,224,224]`, float [0,1], with training-equivalent image resize.
No future image/depth/action is needed. This exposes a policy callable; wiring
and testing it against the legacy UE service/preview is separate work, not
claimed complete. CPU tests exercise native wrapper packing with a tiny backbone;
real Qwen+DA3 forward/backward on the target accelerator is still required.

## Compatibility and verification

Entry configs explicitly save `warp_mode=alpha`, `alpha_radius=1.5` and
`alpha_points_per_pixel=8`. Architecture loading rejects old bilinear settings;
old checkpoints containing removed `action_values` weights are not exact resumes.
The prior bilinear function remains available for diagnostic comparisons, not
as the new public job default. Optimizer/scaler/RNG/data cursor behavior is unchanged.

CPU tests cover actual shared-backbone tiny-model forward/backward through
future depth to proposal/current depth, exact training/probe alpha parity,
partial/empty coverage, batch isolation, alpha-derived correspondence, fully
occluded contributions, continuous mask input, FP32 AMP geometry and strict
checkpoint rejection. Original bilinear tests remain as reference diagnostics.

A CUDA warp-only sweep on three actual stride5 frame pairs used 223 distinct
action chunks (GT scales, offsets, reversals, pure lateral/yaw and random noise).
Alpha r=1.5/K=8 had much smaller random-VJP outliers than normalized bilinear
in those samples, but changes brightness/blur/coverage. Lower gradient magnitude
is NOT proof of better control or useful future-depth supervision.
These probes do not validate full Qwen+DA3 training memory or closed-loop quality.
No new Ascend validation was performed; sort/scatter/cumprod and a real R1-style
forward/backward smoke still need verification on the target NPU stack.
