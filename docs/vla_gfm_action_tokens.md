# VLA–GFM action-token architectures

All variants use the same dataset, five-step action chunk, Future Predictor,
DA3 refine/deep blocks, geometry/depth objectives, and continuous 4-DoF action
normalization. They differ only in how semantic action tokens are initialized
and whether the action is decoded before or after GFM.

## Shared two-bank path

1. Ordinary frozen Condition receives exactly
   `current image + raw dataset instruction`. Query-VLA and Slot-VLA instead
   share the state-aware user prompt
   `current image + Current State: x,y,z,yaw + action question`. The state is
   the corrected OpenVLA-compatible `preprocessed_logs` representation:
   episode-first full-rotation coordinates, xyz in centimetres and yaw in
   degrees. Query-VLA ends at the user message and contains no assistant
   generation token. Slot-VLA adds one explicit assistant message whose content
   is the 5 (or 20) distinct action placeholders; only their final hidden states
   are retained. Literal `In:`/`Out:` markers are not used.
2. A semantic tokenizer creates five ordered action-plan tokens in Predictor
   width. No token mean pooling is used.
3. The five semantic seeds are added only to the latest/current timestep's five
   ordered Action Slots in GAM's Future Predictor. F0 retains five base slots
   without the current instruction seed. Current visual and Action tokens jointly
   predict future shallow geometry. The old per-Predictor-layer language
   cross-attention is disabled, so semantics has exactly one route: Action tokens.
4. Current shallow geometry separately traverses action-free DA3 deep blocks.
   The recommended Geometry Bank exports the current block output at DA3's
   native DPT levels `[19,27,33,39]` without concatenating a stale local copy.
5. In the future DA3 deep traversal, all five action tokens read those four
   matching Current Geometry Bank levels through gated cross-attention, then
   continue with predicted-future tokens. This is configurable for controlled
   cost/representation ablations.
6. A shared token-wise MLP maps the five refined tokens to `[5,4]` continuous
   UAV actions. Current geometry never reads Future or Action, so the observed
   representation remains causal and action-free.

## Modes

| `parallel_vla_gfm_mode` | Semantic tokenization | Policy action |
|---|---|---|
| `external_query` | 5 external ordered queries, one semantic CA block | refined GFM tokens |
| `qwen_tokens` | 5 distinct placeholders contextualized inside Qwen; optional 5-token post-Qwen bidirectional mixer | refined GFM tokens |
| `oft_gfm` | 20 internal dimension placeholders (`5×4`) plus full-bidirectional action-token refinement, grouped to 5 plan tokens | refined GFM tokens |
| `oft_direct` | same 20-token bidirectional refinement | direct pre-GFM scalar-per-token decode |

`oft_direct` is the causal-contribution ablation: the matched Future/depth
branches still train, but the deployed action bypasses GFM. `oft_gfm` tests
whether OFT-style dimension tokenization benefits from subsequent geometric
reasoning.

Qwen3.5 has hybrid causal and linear-attention layers, so the placeholders first
retain Qwen's pretrained causal mask. A small full-bidirectional action-token
stack then matches OFT's within-action communication. This is deliberately
called OFT-style rather than an exact OpenVLA-OFT mask replica: forcing only
Qwen's quadratic layers bidirectional while its linear layers remain causal
would be internally inconsistent. Token count/grouping, bidirectional action
mixing, and direct-vs-GFM decode are all controlled explicitly.

For the five-token mode, `model.parallel_action_post_bidir_layers=2` enables
the matched post-Qwen bidirectional mixer (`bidir-5`). Separately,
`stage1.qwen_action_attention_mode=full_attention_bidir` opens the 5x5 action
block only inside Qwen's ordinary full-attention layers. GatedDeltaNet layers
remain causal, so this option is explicitly a hybrid-mask ablation, not a claim
of fully bidirectional Qwen3.5. The baseline value for both controls is causal/0.

## Run

The release matrix has one scheduler-facing Python entry point per experiment.
For example, Query-VLA, Slot-VLA and its geometry-residual ablation are:

```bash
python experiments/uavflow_remote_ablation/jobs_v2/05_q0.py
python experiments/uavflow_remote_ablation/jobs_v2/08_s2.py
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py
```

The complete ten-command list and required environment variables are in
`docs/REMOTE_TRAINING.md`.  No shell launcher is required by the scheduler.

The primary external-Query and internal-Slot VLA runs both use dependency-free
LoRA on Qwen language-model `q/k/v/o` and MLP projections at rank 32, alpha 16
and dropout 0. Frozen-Qwen variants are retained only as conditioning/architecture
controls and are not labelled VLA-init results.

## Geometry Bank ablations

`model.current_geometry_bank_mode` controls both the read schedule and memory
representation without changing the Current DA3 forward or action-token count:

| value | read layers | memory per patch |
|---|---|---|
| `output_concat` | `19,27,33,39` | `[last_local; current]`, `2D` |
| `output_current` (recommended/default) | `19,27,33,39` | `current`, `D` |
| `global_current` | every odd/global deep block | `current`, `D` |
| `every_layer_concat` | every deep block | `[last_local; current]`, `2D` (legacy compatibility) |

The active ten-cell matrix uses `output_current`. Legacy alternatives remain
model configuration options, but are not active release jobs.

## Dual-view action fusion

Legacy dual-view policies averaged Current/Future action hidden states with a
fixed 0.5/0.5 coefficient. They now use a zero-initialized scalar gate per plan
step, so initialization is exactly backward-compatible while training can
select the reliable branch. Current, Future, and fused predictions share the
same action head. With `loss.dual_branch_action_aux_weight=0.5`, the normalized
action objective assigns 2/3 to fused action and 1/6 to each branch; the outer
GAM action coefficient remains unchanged.
