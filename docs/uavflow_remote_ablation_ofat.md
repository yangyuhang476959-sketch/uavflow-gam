# UAV-Flow remote full-rank ablation: one-factor-at-a-time matrix

All rows are compared directly with `B0`. Unless the changed field is shown,
the row inherits every setting from `B0`.

## Common baseline (`B0`)

| Field | Value |
|---|---|
| Conditioner | frozen Qwen3.5-2B VLM, current image + OpenVLA-style instruction/state prompt |
| Direct numeric pose | off |
| Context | H=1 plus fixed F0 reference |
| Coupled future scale | K=5: predict 5 x 4-DoF actions and geometry at F(t+5) |
| Depth definition | relative/window point-norm |
| Dynamic multiplier | 1 |
| Outer loss | 3 Action + 1 Feature + 3 Depth |
| Inner depth loss | valid-pixel L1 + finite-difference gradient L1 |
| DA3 tuning | blocks 13--39 full-rank; blocks 0--12 and DPT frozen |
| LR | constant: deep 5e-5, Predictor 1e-5, Action head 5e-4 |
| Stage 1 | policy + feature + depth, 5 epochs |
| Stage 2 | joint main losses + action-hidden Stop, 5 epochs; cosine with 500-step warmup |
| Data | fixed stratified 95/5 split; start+5, end+5, append K terminal FN frames |

## Runs

| ID | Single change from B0 | Purpose |
|---|---|---|
| B0 | none | Reference |
| S1COS | Stage-1 constant -> 500-step warmup + cosine to 1% | Clean Stage-1 scheduler ablation; Stage 2 remains cosine |
| P1 | enable direct normalized `[x,y,z,sin(yaw),cos(yaw)]` pose token | Does direct state help? |
| L1 | replace VLM with frozen T5 instruction tokens | T5 versus VLM |
| D1 | relative depth -> fixed dataset-scale metric depth | Is absolute metric scale useful? |
| D2 | relative depth -> separate shape plus predicted log-scale | Can shape/scale separation retain both? |
| D1LOG | D1 plus balanced `0.5 linear + 0.5 log + 1 gradient` inner depth loss | Does log-depth keep distant structure from dominating metric supervision? |
| W3 | dynamic pixel multiplier 1 -> 3 | Mild dynamic emphasis |
| W5 | dynamic pixel multiplier 1 -> 5 | Medium dynamic emphasis |
| W10 | dynamic pixel multiplier 1 -> 10 | Strong dynamic emphasis |
| H0 | A: predict current feature and depth, keep future action chunk | Does current reconstruction suffice? |
| HB | D: observed current + predicted future, joint dual-view deep pass | Does joint geometry interaction help? |
| F3 | coupled scale K: 5 -> 3 | 3-action chunk, F(t+3) geometry, append 3 terminal FN frames |
| F7 | coupled scale K: 5 -> 7 | 7-action chunk, F(t+7) geometry, append 7 terminal FN frames |
| F10 | coupled scale K: 5 -> 10 | 10-action chunk, F(t+10) geometry, append 10 terminal FN frames |
| M1 | H=1 -> sampled causal H={1,2,3,4}; K remains 5 | Reproduce GAM multi-window history |
| C1_PL | P1 + L1: direct pose and frozen T5 | Does pose change the language-encoder conclusion? |
| C2_D2HB | D2 + HB: shape/scale-separated and current+future depth | Can a current-depth anchor stabilize metric-scale recovery? |
| C3_W3HB | W3 + HB: dynamic multiplier 3 and current+future depth | Does mild dynamic emphasis help after geometry is stabilized? |
| C4_F3W3 | F3 + W3: K=3 and dynamic multiplier 3 | Is near-future dynamic blur primarily horizon- or weight-limited? |
| C5_F10HB | F10 + HB: K=10 and current+future depth | Can current geometry regularize the deliberately hard long horizon? |
| CA1_HB | C: separate current/future passes, shared per-layer action-to-current CA | Can current geometry improve future geometry via actions? |
| HC_DIRECT | B: bypass Predictor, observed current + language/F0 action adapter | Is the feature Predictor necessary? |
| HE_DUALPRED | E: predicted current and future in joint deep pass | Observed versus predicted current |
| HF_BRIDGE | F: action-only bridge between observed current and predicted future | Is direct visual interaction necessary? |

Total: 25 Stage-1 runs: original 21 minus two depth architecture controls plus
six A–F architectures. Each automatically launches its matching five-epoch
joint Stop Stage 2 from the selected Stage-1 checkpoint.

`D1LOG` uses a normalized log clamp of `0.001`, which is `0.1 m` under
the fixed `100 m` metric divisor. `W10` is retained as the strong dynamic-
weight stress test, so the default compact matrix contains 25 runs.

Do not confuse its per-pixel log-depth term with the `D2` log-scale loss:

- `D1LOG`: fixed metric target; no learned scale head and no log-scale loss.
  Its inner weights are linear/log/gradient = `0.5/0.5/1.0`.
- `D2`: predicts normalized depth shape plus one log scene-scale scalar per
  window. `depth_scale_loss_weight=1`, so under the outer depth weight 3 its
  effective coefficient in the total loss is 3.

The fixed episode reference image F0 is conditioning only and is never used to
compute depth normalization. Relative and scale-separated runs compute one
shared point-norm scale from the complete sampled temporal window. For baseline
H=1, K=5 this is `[F(t), F(t+5)]`; `current`, `future`, and `both` targets all
use that same scale. `both` averages its two target losses rather than summing
them, so it does not double the scale-loss coefficient.

See [architecture details and changed output names](DEPTH_ARCHITECTURES.md).
The 25-row compact matrix contains one clean baseline scheduler pair but does not
duplicate every scientific row under both schedules. The separate
compute-rich launcher therefore uses a deduplicated 64-cell design containing
all main effects and targeted two-factor/hard-setting interactions, without a
2304-cell Cartesian explosion.

## Meaning of `future=K`

`future=K` is one coupled temporal-scale variable. It jointly determines:

- action output: `[a_t, ..., a_(t+K-1)]`;
- feature/depth target: `F(t+K)`;
- terminal construction: append K copies of `FN`, with zero actions beyond FN.

For an original trajectory `F0...FN`, starts remain `t=0...N`. The final K
starts become progressively more absorbing because their K-action chunks
contain increasing numbers of terminal zeros. Thus K=10 deliberately contains
ten partial/full absorbing starts, whereas K=3 contains three.

Do not restrict all K runs to a K=10 common-start manifest. Preserve K's
physical partial/full terminal patterns, but deterministically balance their
sample-level union to 20% so terminal frequency does not confound the horizon
comparison.

The fixed endpoint weighting added on top is:

- five extra first-window samples per episode;
- five extra terminal same-frame zero-action samples per episode;
- one K-dependent absorbing candidate pass, bidirectionally balanced to 20%
  (`endpoint_absorbing_window_count=1`,
  `endpoint_absorbing_train_fraction=0.20`).

Report normal, start-repeat, terminal-repeat and absorbing metrics separately.
Also report each action slot separately because K changes both output size and
the proportion of terminal-zero slots.

## Interpretation rule

Pose, language, depth-definition, dynamic-weight, depth-target and
multi-window conclusions are pairwise `variant - B0` comparisons. `F3/F7/F10`
compare the complete coupled temporal scale K against K=5; they are not pure
depth-horizon tests. Do not select a row only from weighted training loss. Use
action physical error, accumulated pose, unweighted static/dynamic depth
quality and task-class closed-loop SR. Compare Stop only after confirming that
Stage 2 did not materially degrade its Stage-1 action policy.
