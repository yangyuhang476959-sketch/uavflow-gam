# C / CA1_HB: per-layer current-geometry reads

This replaces C's former terminal-only CA, without adding a matrix cell.
The compact matrix remains 25 experiments. Losses, sampling and stages remain:
Qwen, H=1, K=5, no direct pose, relative depth, dynamic=1; five policy epochs
then five joint action/Stop epochs with cosine decay.

For each executed DA3 deep block l:

1. Current advances through shared deep block l, without actions.
2. Predicted Future and Action advance through the same block l separately.
3. Action queries corresponding Current patches through residual CA.
4. Updated Action enters block l+1 and interacts with Future.

```text
Current patches at l ------> CA ------> updated Action at l
                             ^                    |
Future + Action --> deep l --+                    v
                                      deep l+1: Action <-> Future
```

Existing local/global alternation is unchanged. CA is inserted after EVERY
executed deep block, including the last, not instead of alternate blocks.
The last read affects final Action but cannot retroactively change the last
visual output; earlier reads can affect later Future depth features.

Current/Future do not attend directly. Current never reads Action/Future.
Current is not detached: future/action losses can train its branch and shared
deep weights through Action's CA reads.

All layers SHARE one width-256, 8-head CA (gate initially 0.001), keeping its
parameter count equal to the terminal module. Q is D-dimensional Action;
KV uses the matching layer's local/global concatenated 2D patches, excluding
camera/register tokens. It does not repeatedly read the final Current layer.
The module uses the Predictor LR group.

Implementation computes independent Current first, caches layer patches,
then runs Future with per-layer reads. For that Current pass the dependency
graph is equivalent to advancing streams in lockstep; no Future-to-Current
feedback exists. Cached four DPT levels serve current depth loss without a
third deep pass. Current/future losses are averaged; Stage-2 Stop reads final
refined Action. Intermediate memories/repeated CA increase compute and memory.
Deep blocks and CA support checkpointing. Real GPU cost is not yet measured.

## Run

```bash
RUN_IDS=CA1_HB bash experiments/uavflow_remote_ablation/run_server.sh
```

The launcher sets `model.current_geometry_read_mode=per_layer` and appends
`_layerca` to output names, avoiding old terminal checkpoint/completion
markers. Checkpoints save the mode and reject mismatched transfer/resume.
Old checkpoints default to terminal mode. External inference loaders must
pass the saved mode as well as the existing enabled flag.

CPU regression checks (no GPU or model download):

```bash
PYTHONPATH=src:. python -m unittest \
  experiments.uavflow_remote_ablation.test_current_geometry \
  experiments.uavflow_remote_ablation.test_geometry_architectures -v
```

Tests exercise actual encoder loops with small deterministic blocks: early
visual output is unaffected by that layer's subsequent CA, later visual
features receive gradients from Current, and checkpointed gradients match.
