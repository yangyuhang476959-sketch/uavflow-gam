# VLA--GAM ten-cell matrix

The active release matrix has ten experiments: `G0 G1 C0 C1 Q0 S0 S1 S2
R1 DV`. The former 25-cell depth matrix is superseded and is not used by the
cluster entrypoints.

All cells share the OpenVLA-compatible stratified split, K=5 action/geometry
horizon, GAM `3:1:3` action/feature/depth loss, full-rank DA3 blocks 13+, no
first-frame duplication, and the natural five terminal absorbing windows. No
artificial 20% terminal rebalance is applied.

Stage 1 trains action/feature/depth for 20 epochs at constant LR. During
validation it atomically maintains `best_action.pt` using the lowest H=1
validation Action loss. Stage 2 starts from that checkpoint (not blindly from
the final epoch) and jointly fine-tunes the same policy plus the Stop head for
20 epochs with cosine decay. Only Stop receives its separate normalized pose
branch; this does not alter the main pose ablation. For five-slot policies,
Stop preserves order by concatenating the five refined states before its MLP.

| ID | comparison |
|---|---|
| G0 | frozen T5 GAM, no numeric pose |
| G1 | G0 plus direct numeric pose |
| C0 | frozen Qwen current-image/raw-instruction Condition |
| C1 | C0 plus direct numeric pose |
| Q0 | LoRA Query-VLA, external 5 queries, Semantic + Current Geometry banks |
| S0 | LoRA Slot-VLA, five Qwen slots, no Current branch |
| S1 | S0 plus Current depth supervision; Action cannot read Current |
| S2 | S1 plus per-output-layer reads from `output_current` Geometry Bank |
| R1 | S2, but geometry predicts a zero-initialized residual over VLA action |
| DV | older Current/Future dual-view branch with learned fusion, diagnostic |

Prompt contracts:

- G0/G1: T5 receives the raw instruction.
- C0/C1: frozen Qwen receives current image + raw instruction, with no
  assistant message. Only image-conditioned text states after the image prefix
  are passed to GAM as per-layer language memory.
- Q0: Qwen receives current image + corrected `Current State` + action
  question, with no assistant message. Its five external queries read only the
  image-conditioned text states.
- S0/S1/S2/R1: the same user message as Q0, followed by one assistant message
  containing five action placeholders. Only those five hidden states are used.

`Current State` comes from `preprocessed_logs[:, [0,1,2,4]]`: episode-first
full-rotation coordinates, xyz in centimetres and yaw in degrees. Literal
`In:`/`Out:` markers are not used.

The prepended F0 reference contains visual tokens only from the attention
graph: its rectangular placeholder action slots are fully key-masked and
zeroed. Only the real current Ft block owns control slots. Five-slot VLA--GAM
also adds a learned DA3-width chunk-step embedding before deep refinement, so
action order remains explicit in both the Future Predictor and geometry path.

## One Python command per 8-GPU cluster job

After exporting the paths documented in `docs/REMOTE_TRAINING.md`, enqueue the
matching file. Each command audits data/runtime, resumes safely, runs Stage 1
and Stage 2, and writes only its own result directory.

```bash
python experiments/uavflow_remote_ablation/jobs_v2/01_g0.py
python experiments/uavflow_remote_ablation/jobs_v2/02_g1.py
python experiments/uavflow_remote_ablation/jobs_v2/03_c0.py
python experiments/uavflow_remote_ablation/jobs_v2/04_c1.py
python experiments/uavflow_remote_ablation/jobs_v2/05_q0.py
python experiments/uavflow_remote_ablation/jobs_v2/06_s0.py
python experiments/uavflow_remote_ablation/jobs_v2/07_s1.py
python experiments/uavflow_remote_ablation/jobs_v2/08_s2.py
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py
python experiments/uavflow_remote_ablation/jobs_v2/10_dv.py
```

Defaults are `NPROC=8`, `GLOBAL_BATCH_SIZE=32`, four samples per GPU. Override
only through environment variables when the node differs. For a cheap launch
audit, append `--max-trajectories 20 --stage stage1` and set
`STAGE1_EPOCHS=1`.
