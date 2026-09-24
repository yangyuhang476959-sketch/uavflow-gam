# UAV-Flow remote ablation runner

This directory runs the compact 25-experiment matrix described in
`docs/uavflow_remote_ablation_{ofat,detailed}.md`.

## Temporal contract

For `future=K`, the runner sets:

```yaml
dataset.future_steps: 1
dataset.visual_anchor_stride: K
dataset.chunk_size: K
model.action_chunk_size: K
model.feature_target_offset: 1
```

This predicts K actions and endpoint geometry `F(t+K)`. With one absorbing
extension, the dataset adds exactly K partial/full terminal windows. It does
then deterministically up- or down-samples their union to 20% of the final
training tickets. Both stages repeat the first normal window five extra times.
Stage 1 adds no pure terminal self-pair repeats; Stage 2 adds five per episode
(FN -> FN, all K actions zero). Neither stage repeats the last moving window.
Stage 2 lowers the existing Predictor, deep DA3, and Action-head peak learning
rates to one tenth of Stage 1, while its separate Stop head uses `1e-4`.
The Stop head separately reads the normalized episode-relative current pose;
this does not change the main Action/Predictor pose ablation. Its checkpoint
directory is `stage2_stop_actionpose_clip_lr/`.
The output suffix `_simpleprompt_s1end0_s2end5` prevents resuming older
OpenVLA-prompt or end+5 Stage-1 runs.

The multi-window `M1` row needs four stride-5 contexts and therefore requests
more raw anchors, but explicitly caps terminal absorbing starts at K=5.

## Required environment

```bash
export PROJECT_ROOT=/path/to/Geometric-Action-Model
export UAVFLOW_SIM_ROOT=/path/to/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=/path/to/UAV-Flow-Sim-Depth-Hybrid-All-Final
export DA3_CHECKPOINT=/path/to/track4world_da3.pth
export QWEN_MODEL=/path/to/qwen3.5-2b
export T5_MODEL=/path/to/t5-base
export PYTHON_BIN=/path/to/env/bin/python
export TORCHRUN_BIN=/path/to/env/bin/torchrun
```

`run_server.sh` uses global batch 24 by default: 4 GPUs, 6 samples/GPU, and no
gradient accumulation. This is a practical compromise between OpenVLA-UAV's
global batch 32 and GAM post-training's global batch 12, while also matching
GAM pre-training exactly. The launcher derives the per-GPU batch as
`GLOBAL_BATCH_SIZE / NPROC`, so changing the GPU count does not silently
change the effective batch.

`D1` uses one global metric divisor (100 m by default). Override it with
`DEPTH_FIXED_SCALE_METERS`; this only changes numerical conditioning and does
not independently normalize each episode/window.

If the consolidated depth directory still needs fallback roots, pass an
OmegaConf list:

```bash
export DEPTH_FALLBACKS="['/path/depth-corrected-reverse','/path/depth-original']"
```

## Run

Smoke-test a small subset without Stage 2:

```bash
CUDA_DEVICES=0,1 NPROC=2 GLOBAL_BATCH_SIZE=4 BATCH_SIZE=2 MAX_TRAJECTORIES=20 \
STAGE1_EPOCHS=1 \
OUTPUT_ROOT=/tmp/uavflow_remote_smoke \
RUN_IDS=B0,F3,F10 RUN_STAGE2=0 \
bash experiments/uavflow_remote_ablation/run_remote.sh
```

Run the complete two-stage matrix:

```bash
CUDA_DEVICES=0,1,2,3 NPROC=4 GLOBAL_BATCH_SIZE=24 BATCH_SIZE=6 \
bash experiments/uavflow_remote_ablation/run_remote.sh
```

On a server whose directory layout matches this workstation, the same full
matrix is a single command:

```bash
bash experiments/uavflow_remote_ablation/run_server.sh
```

The command above is sequential. To run the 25 cells concurrently, use the
launcher matching the cluster scheduler; all launchers share the exact same
cell enumeration in `compact_cells.sh`:

```bash
# Slurm (25 array jobs, four GPUs each)
sbatch experiments/uavflow_remote_ablation/submit_slurm_compact.sh

# PBS Pro/OpenPBS (inherit the exported dataset/checkpoint paths)
qsub -V experiments/uavflow_remote_ablation/submit_pbs_compact.sh

# IBM LSF
bsub < experiments/uavflow_remote_ablation/submit_lsf_compact.sh

# One machine with explicitly enumerated four-GPU groups
GPU_GROUPS='0,1,2,3;4,5,6,7' \
bash experiments/uavflow_remote_ablation/run_local_gpu_pool.sh
```

Each array cell invokes `run_compact_cell.sh INDEX`, sets exactly one
`RUN_IDS`, and preserves global batch 24. On a 100-GPU allocation, 25 cells use
100 GPUs in one wave. Cluster-specific queue, account, wall-time and memory
directives may be added to the scheduler header without changing training.

Stage 1 defaults to the earlier UAV runs' constant LR schedule. To retain the
same numerical parameter-group LRs but use released GAM's Stage-1 schedule:

```bash
USE_GAM_SCHEDULE=1 bash experiments/uavflow_remote_ablation/run_server.sh
```

Every stage trains five physical epochs. Stage 1 trains action/feature/depth;
Stage 2 initializes from Stage 1 and jointly trains the same losses plus an
action-hidden Stop head for another five epochs. Stage 1 is constant by
default; Stage 2 always defaults to 500-step warmup plus cosine decay, matching
the earlier CLIP Stop runs. A numbered checkpoint is saved at each epoch
boundary, matching their five checkpoints over five epochs. `last.pt` is a
zero-extra-space hard link to the newest epoch checkpoint and contains model,
optimizer, AMP scaler, scheduler step, epoch, and batch cursor. Mid-epoch
rolling recovery is optional via `CHECKPOINT_EVERY_STEPS=N`; its default is
zero (disabled). Re-running skips `_SUCCESS` stages and resumes from the newest
saved checkpoint.

Run one variable only:

```bash
RUN_IDS=W3 bash experiments/uavflow_remote_ablation/run_remote.sh
```

Inspect progress:

```bash
python experiments/uavflow_remote_ablation/status.py \
  results/remote_ablation
```

## Compute-rich interaction-aware matrix

Run all main effects plus the targeted interactions most likely to change the
conclusion: pose x language, scale x depth target, dynamic weight x depth
target, K x depth target, K x context, scale x dynamic, scale x K, and a hard
K=10 conditioning block. Overlapping configurations are deduplicated. This
produces 64 two-stage pipelines; every Stage 2 uses cosine:

```bash
bash experiments/uavflow_remote_ablation/run_compute_rich_matrix.sh
```

The launcher writes `matrix_manifest.tsv` with a `blocks` provenance column and
is array/shard friendly. For eight workers, launch the same command with
`MATRIX_SHARD_INDEX=0...7`:

```bash
MATRIX_SHARD_COUNT=8 MATRIX_SHARD_INDEX=0 \
bash experiments/uavflow_remote_ablation/run_compute_rich_matrix.sh
```

Run exactly one array cell with `MATRIX_CELL_INDEX`, or only materialize and
inspect the complete manifest with `MATRIX_DRY_RUN=1`.

All shards reuse one locked split manifest. Finished cells are skipped and
partial cells resume from their newest epoch checkpoint. Set
`CHECKPOINT_EVERY_STEPS` only when mid-epoch rolling recovery is desired.

`B0` creates the same split as every other run, but the split is materialized
before any training in `shared_split_seed42.json`; run order therefore cannot
change train/validation membership.
