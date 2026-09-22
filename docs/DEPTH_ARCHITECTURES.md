# Compact geometry architectures A–F

These replace the old H0/HB depth architecture controls, not stored datasets
or experiment results. Original 21 minus two controls plus six architectures
gives **25 cells**. All other baseline settings and five-epoch policy +
five-epoch joint Stop stages remain; Stage 2 uses cosine decay.

| Architecture / ID | Current source | Future source | Deep processing |
|---|---|---|---|
| A / H0 | Predictor reconstructs current shallow feature | none | Current + action |
| B / HC_DIRECT | observed shallow, no Predictor execution | none | Current + action |
| C / CA1_HB | observed shallow | predicted shallow | Separate deep passes; after each block action reads matching current patches through shared CA |
| D / HB | observed shallow | predicted shallow | One full-interaction dual-view deep pass |
| E / HE_DUALPRED | predicted shallow | predicted shallow | One full-interaction dual-view deep pass |
| F / HF_BRIDGE | observed shallow | predicted shallow | One masked dual-view deep pass; action bridges views |

A/B still predict five future actions, but their depth is current depth. A's
feature target is **current**, not future. B has no feature prediction loss:
its retained Predictor object is frozen and never executed. A small learned
action query reads observed Ft/F0 patches and language to seed the deep action
token, so bypassing Predictor does not remove instruction conditioning.

D/E/F pack `[camera, action, registers, patches]` for each view. Current uses
the pretrained reference camera token and future the pretrained source token.
Spatial patch positions are preserved. Both deep action hidden states are
mean-pooled into the existing head: **one action chunk**, not two policies.
E runs the shared Predictor twice with learned current/future task embeddings,
and averages the two feature losses. It does not add a second Predictor.

F's global attention (rows query, columns key/value):

| Query | Current visual | Future visual | Action (both views) |
|---|---|---|---|
| Current visual | yes | no | no |
| Future visual | no | yes | yes |
| Action | yes | yes | yes |

Camera/register tokens obey their view's visual mask too. Local layers also
block current queries from reading action, preventing future information from
flowing back to Current via Action. Future can receive Current indirectly via
Action; there is no direct visual-view edge. This is not gradient detachment:
action loss can still train current features through action's reads.

Both-depth losses average current/future terms. Relative normalization retains
one shared sampled-window scale, not independent per-view scales. Teacher
features remain detached. New modes require H=1, one real camera and deep
action output. Multi-window M1 retains the legacy architecture. C2_D2HB,
C3_W3HB and C5_F10HB inherit the new D definition of HB.

## Outputs and resume

H0/HB/C2_D2HB/C3_W3HB/C5_F10HB outputs get `_g2`, including with RUN_LABEL,
to avoid silently reusing old completion markers/checkpoints. Exact resume
checks architecture. Checkpoints save its mode and new adapter/role weights.
External inference loaders must pass `model.geometry_architecture` and call
`load_architecture_state`, plus C's existing `load_current_geometry_read`.
Training handles both Stage-1/2 transfer and exact resume.

C uses `current_geometry_read_mode=per_layer` and output suffix `_layerca`.
Old terminal checkpoints cannot silently resume this changed architecture.
See CURRENT_GEOMETRY_ABLATION.md for per-layer communication details.

Detailed-64 intentionally remains a **legacy loss-target matrix**, not a
64-cell expansion of A–F: it includes multi-window combinations unsupported
by these new H=1-only modes.

## Run after standard environment/data setup

```bash
RUN_IDS=H0,HC_DIRECT,CA1_HB,HB,HE_DUALPRED,HF_BRIDGE \
  bash experiments/uavflow_remote_ablation/run_server.sh
# All 25 scheduler jobs, four GPUs each:
sbatch experiments/uavflow_remote_ablation/submit_slurm_compact.sh
```

The first command runs sequentially. CPU regression checks:

```bash
PYTHONPATH=src:. python -m unittest \
  experiments.uavflow_remote_ablation.test_geometry_architectures \
  experiments.uavflow_remote_ablation.test_current_geometry -v
```

Tests use small backbones for routing, masking, gradients, pooling, checkpoint
state and losses. Real pretrained multi-GPU throughput/memory remain to be
measured. This update does not start training.
