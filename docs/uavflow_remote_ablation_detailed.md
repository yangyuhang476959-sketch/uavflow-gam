# UAV-Flow remote full-rank ablation: detailed design

## 1. Goal and fixed baseline

This study asks which conditioning, geometric supervision and temporal scale
improve UAV navigation without turning the study into a factorial grid. The
reference run (`B0`) is:

- frozen Qwen3.5-2B VLM conditioner;
- current image plus the OpenVLA-style instruction/state prompt, with no
  separate numeric pose token;
- one observed current anchor (`H=1`) plus fixed first-frame reference;
- coupled future/chunk scale K=5: five 4-DoF actions and geometry at `F(t+5)`;
- relative GAM-window point-normalized depth;
- outer loss `3 * action + 1 * feature + 3 * depth`;
- inner depth loss: valid-pixel L1 plus finite-difference gradient L1;
- dynamic pixel multiplier 1;
- DA3 blocks 0--12 and DPT head frozen;
- DA3 blocks 13--39 trained full-rank, not LoRA;
- constant LR: `deep=5e-5`, `predictor=1e-5`, `action_head=5e-4`.

The VLM is frozen in every run.

## 2. Dataset split and endpoint construction

All ablations reuse one materialized episode split:

- split mode: `stratified_instruction`;
- split seed: 42;
- validation ratio: 5%;
- no episode moves between train and validation across runs.

OpenVLA-UAV's UAV-specific loader has no effective validation split. This 95/5
split is for ablation/model selection. Retrain the selected configuration on
100% of official training episodes for final closed-loop comparison.

Training endpoint rules are:

- first normal episode window: five additional copies;
- terminal same-frame zero-action window: five additional copies;
- append K copies of final frame `FN` when `future/chunk=K`;
- deterministically up- or down-sample the resulting partial/full absorbing
  pool to exactly 20% of training tickets;
- `terminal_zero_action=true`;
- no photometric augmentation.

In current dataset terms, the coupled K construction is represented by
`future_steps=1`, `visual_anchor_stride=K`, `chunk_size=K`,
`endpoint_absorbing_window_count=1`, and
`endpoint_absorbing_train_fraction=0.20`. Validation removes the fixed five-copy
endpoint weighting but keeps one natural K-dependent absorbing pass, and logs
normal versus terminal metrics separately.

## 3. Two-stage optimization

### Stage 1: policy and geometry, 5 epochs

Train action, feature, depth, Predictor and full-rank DA3 blocks 13--39. Stop
loss is disabled. Save every epoch boundary and use the final fifth-epoch
checkpoint as the deterministic Stage-2 initializer.

### Stage 2: joint Stop training, 5 epochs

Initialize from the final fifth-epoch Stage-1 checkpoint. Insert a Stop head tapped from
the action hidden representation and jointly continue action, feature, depth
and Stop BCE losses. This is joint fine-tuning, not Stop-only fitting. Preserve
Stage 1 so action degradation remains measurable. Report Stop precision,
recall, F1 and threshold sweeps.

Stage 2 always uses the schedule established by the earlier CLIP Stop runs:
500-step warmup followed by cosine decay to 5% of the initial LR. The Stage-1
schedule ablation compares constant against released-GAM cosine (500-step
warmup, 1% minimum); Stop scheduling is therefore held fixed.

## 4. Coupled future/action scale

`future=K` means all three of the following:

1. predict K actions `[a_t, ..., a_(t+K-1)]`;
2. predict feature/depth at action endpoint `F(t+K)`;
3. append K terminal `FN` frames and zero-pad actions past the endpoint.

The scale study uses `K in {3,5,7,10}`. These are coupled temporal-scale
comparisons, not geometry-only offsets. For example, K=10 has a ten-action head,
targets `F(t+10)`, and creates ten partial/full absorbing starts.

At fixed K=5, depth-target ablations test geometric supervision itself. Feature
supervision remains at `F(t+5)` in all three rows:

- `current`: supervise depth at `F(t)` while predicting five actions;
- `both`: average current `F(t)` and endpoint `F(t+5)` depth losses.

For `both`, budgets do not double:

`L_depth = 0.5 * L_depth_current + 0.5 * L_depth_future5`.

Current and future targets share one GAM scene scale computed over the complete
sample window. They are not normalized independently: two independent scales
would erase part of the approach/retreat magnitude that the temporal objective
is meant to preserve. `scale_separated + both` likewise predicts one common
window log-scale, not one scale per target frame.

The outer coefficients remain `3/1/3`.

## 5. K-dependent terminal windows and fairness

For physical frames `F0...FN`, form:

`F_ext = [F0, ..., FN] + [FN] * K`.

For each original physical start `t=0...N`, the target is `F_ext[t+K]` and the
output is a K-action chunk. Slots beyond the final physical transition are
zero. Hence the physical start-index support can remain the same across K; no
K=10 common-manifest truncation is needed.

Changing K intentionally also changes:

- action-head dimensionality;
- prediction distance;
- number of partial absorbing starts;
- terminal-zero action slots per episode.

The first three are inseparable parts of the user's `future/action chunk`
definition. For a clean horizon comparison, the sample-level absorbing share
is fixed at 20% by deterministic bidirectional balancing. Record per-slot
action metrics, normal-only metrics, absorbing-only metrics, the number of
downsampled tickets and zero-slot fraction.

## 6. Depth definitions

### Relative/window point-norm

Use one shared scene scale for the complete sample window:

`D_relative = D_meters / scene_scale`.

### Fixed metric scale

Divide metric depth by one training-set constant:

`D_fixed_metric = D_meters / scale_dataset`.

Do not independently align each frame.

### Shape/scale separated

Predict normalized depth shape plus one scalar:

- dense `D_shape = D_meters / scene_scale`;
- scalar `log(scene_scale)`.

Combine original shape loss with scalar log-scale L1 and report reconstructed
metric errors.

## 7. Dynamic-pixel weighting

Apply dynamic weighting inside one normalized pixel mean:

`sum(w_p * error_p) / sum(w_p)`.

Static pixels use 1; dynamic pixels use `k in {1,3,5,10}`. Apply the same rule
to L1 pixels and gradient pairs, with a pair touching a dynamic pixel assigned
weight k. Do not add a second independently normalized dynamic loss. Keep the
outer depth coefficient 3.

## 8. GAM-style multi-window context

This is separate from coupled K. Baseline observes one current anchor. `M1`
samples `H in {1,2,3,4}` while K remains 5. With stride five, H=4 observes:

`[F(t-15), F(t-10), F(t-5), F(t)]`.

Dense causal supervision covers all valid anchor transitions, and each anchor
predicts its five-action chunk and next endpoint geometry. Average losses over
valid causal steps so larger H does not multiply loss scale. Use masks for
insufficient-history episode starts and report them separately.

Internally M1 must load five stride-5 anchors to support H=4 plus its target,
so the dataset's raw clampable tail would contain 20 starts. Cap
`endpoint_absorbing_max_starts=5` for M1: K remains five and the experiment
does not silently receive four times as many terminal windows.

## 9. One-factor experiment list

- `B0`: Qwen, no pose, relative depth, K=5, dynamic=1;
- `P1`: add direct normalized pose;
- `L1`: Qwen -> frozen T5;
- `D1`: relative -> fixed metric depth;
- `D2`: relative -> shape/scale-separated depth;
- `D1LOG`: fixed metric depth with `0.5 linear + 0.5 log + 1 GAM-gradient`;
- `W3/W5/W10`: dynamic multiplier 3/5/10;
- `H0`: at K=5, current depth only (feature remains future5);
- `HB`: at K=5, current plus future5 depth (feature remains future5);
- `F3/F7/F10`: coupled K=3/7/10;
- `M1`: sampled causal H={1,2,3,4}, K=5.

Together with the `S1COS` baseline scheduler control, the clean main-effect
set contains 16 Stage-1 runs. The compact executable matrix adds five targeted
interactions:

- `C1_PL`: direct pose + T5;
- `C2_D2HB`: shape/scale-separated + current/future depth;
- `C3_W3HB`: dynamic multiplier 3 + current/future depth;
- `C4_F3W3`: K=3 + dynamic multiplier 3;
- `C5_F10HB`: K=10 + current/future depth.

With the additional CA1_HB current-geometry action read control, this gives 22 Stage-1 runs, each followed by its matching Stage-2 joint Stop
run. The five interactions were selected to resolve specific ambiguities in
the main effects; this is not a Cartesian product.

For `D1LOG`, the normalized log clamp is `0.001`, equivalent to `0.1 m`
under the fixed `100 m` divisor. `W10` remains in the default compact matrix
as the strong dynamic-weight stress test.

`D1LOG` has no scale head: its log term is a per-pixel log-depth L1. In
contrast, `D2` adds one predicted log scene-scale scalar per window and uses
`depth_scale_loss_weight=1`; after the common outer depth coefficient 3, this
scale loss has effective total-loss coefficient 3. F0 is excluded from depth
normalization. One shared scale is computed from the sampled temporal window
(`[F(t), F(t+K)]` for H=1), including when current and future depth are both
supervised; the current/future losses are averaged.

### Compute-rich extension

The server-scale matrix remains ablation-oriented rather than taking the full
2304-cell Cartesian product. It deduplicates 64 configurations drawn from:

- the baseline and clean constant-versus-cosine comparison;
- `pose x language`;
- `depth scale x depth target`;
- `dynamic weight x depth target`;
- `K x depth target` and `K x context`;
- `depth scale x dynamic weight` and `depth scale x K`;
- smaller `context x target`, `language x target`, and `pose x K` blocks;
- a four-way conditioning check at the hard `K=10, dynamic=5, both-depth`
  setting;
- five representative cosine checks to test whether scheduler choice changes
  a non-baseline conclusion.

Every row retains the same split and endpoint construction. Qwen uses current
RGB plus instruction without a textual pose field in this matrix, so the
direct-pose axis remains independent.

## 10. Metrics emitted during training

Every Stage-1 run reports normalized and physical action L1, each action slot,
K-action accumulated pose error, feature loss/cosine, depth L1/gradient,
dynamic-mask coverage, boundary/start error, scene scale (and scale-head error
for D2), zero-slot fraction, and validation sample counts.

Every Stage-2 run additionally reports Stop accuracy/recall, positive and
negative probabilities, plus precision/recall/F1/FPR over thresholds
0.45--0.90. Task-class closed-loop SR remains a separate simulator evaluation.
