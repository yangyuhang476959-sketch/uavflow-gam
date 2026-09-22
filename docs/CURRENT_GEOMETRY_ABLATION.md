# CA1_HB: current geometry as an action input, not only an auxiliary loss

The compact matrix now has **22** cells. All previous 21 IDs and scheduler
indices remain unchanged. `CA1_HB` is appended at index **21**. The separate
compute-rich/detailed matrix is unchanged.

Compare **HB vs CA1_HB**, not just B0 vs CA1_HB. Both use Qwen, no numeric pose,
H=1, chunk/stride=5, relative depth, dynamic weight=1 and equal current/future
depth losses. Stage 1 stays 5 epochs with constant LR; Stage 2 stays 5 joint
action/Stop epochs with cosine decay. Sampling, augmentation and loss weights
are identical. Only `model.current_geometry_action_enabled` changes.

```
observed shallow -> shared DA3 deep -> current patches -> current depth loss
                                             |
predicted future + action token -> DA3 deep -> action hidden
                                             |
                  one residual CA(query=action, KV=current patches)
                                             |
                               existing ActionHeadV2 -> chunk=5
```

The read uses the last current deep level, removes CLS/register tokens, and
projects its 2D features to width 256. It has 8 heads and a scalar residual gate
initialized to 0.001. No geometric coordinates, image-token fusion, patch-patch
attention, Predictor K/V expansion, mapping or new loss is added. Its parameters
belong to the existing **Predictor LR group**, not the faster action-head group.
Current feature gradients are not detached. Stop reads the resulting refined
action hidden in Stage 2, using the existing legacy-action-token Stop head.

Current deep features are cached in the forward output and reused by current
depth loss: one current DA3 pass, not two. The original future path is unchanged.
Training adds the small CA over HB. Inference also needs the current DA3 deep
pass that loss-only HB can omit. Current DPT is not required for action inference.
No latency/VRAM/SR improvement is claimed without measurement.

This first ablation rejects H>1: the existing observed deep encoder is not
temporally causal. Extending it requires an explicit causal implementation,
not silently mixing later real observations into earlier actions.

The module is saved as `current_geometry_read` in every training checkpoint,
restored on exact resume and carried from Stage 1 to Stage 2. Missing weights
are an error for enabled runs. Old runs keep the module disabled; construction
does not change their random initialization. External inference loaders must
pass the saved flag to the constructor and call `load_current_geometry_read`
before serving a CA1_HB checkpoint; do not silently load it as ordinary HB.

## Run (after the normal environment/data setup)

```bash
RUN_IDS=CA1_HB bash experiments/uavflow_remote_ablation/run_server.sh
# Or scheduler-independent compact index:
bash experiments/uavflow_remote_ablation/run_compact_cell.sh 21
```

All compact launchers include the new cell. Four GPUs per cell means 22 cells
occupy 88 GPUs when launched simultaneously, subject to scheduler capacity.

CPU regression checks (no model download or GPU allocation):

```bash
PYTHONPATH=src:. python -m unittest experiments.uavflow_remote_ablation.test_current_geometry -v
```

Tests use a small fake DA3 to validate wiring, gradients, H rejection, disabled
baseline initialization, serialization and optimizer transfer. They do not
replace a real multi-GPU training or deployment smoke test.
