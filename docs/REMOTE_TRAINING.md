# Remote UAV-Flow training

This document reproduces the compact 21-run ablation and the optional
compute-rich matrix on a clean GPU server. Large datasets and checkpoints are
not stored in Git.

## 1. Clone and create the environment

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git UAVFlow-GAM
cd UAVFlow-GAM

conda env create -f environment-uavflow.yml
conda activate uav-gam
bash scripts/setup_uavflow_remote.sh
```

The validated environment uses Python 3.12, PyTorch 2.5.1+cu124 and
torchvision 0.20.1+cu124. Override `TORCH_INDEX_URL` when the server requires a
different CUDA wheel. The setup installs the pinned DA3 source checkout but
does not install its optional 3D rendering stack.

## 2. Download public data and model weights

The default endpoint uses the mainland-accessible Hugging Face mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=$PWD/data_remote
export MODEL_ROOT=$PWD/checkpoints
bash scripts/download_uavflow_assets.sh
```

This downloads:

- official `wangxiangyu0814/UAV-Flow-Sim` (21 parquet shards);
- `Qwen/Qwen3.5-2B`;
- `google-t5/t5-base`;
- `checkpoints/track4world_da3.pth` from the GAM training assets.

The derived calibrated/hybrid depth repository is hosted on ModelScope. The
download script uses it by default:

```bash
export DEPTH_HUB=modelscope                 # or: huggingface
export DEPTH_DATASET_REPO=acetaffy123/UAV-Flow-Sim-Depth
bash scripts/download_uavflow_assets.sh
```

For an offline transfer, copy the six `tar.zst` shards plus `metadata/` into
`$DATA_ROOT/UAV-Flow-Sim-Depth-Archive`, then run:

```bash
python scripts/extract_uavflow_depth.py \
  --dataset-root "$DATA_ROOT/UAV-Flow-Sim-Depth-Archive" \
  --output "$DATA_ROOT/UAV-Flow-Sim-Depth"
```

The extracted depth directory is approximately 20.6 GiB. It contains 10,109
episodes and seven load-time instruction corrections; never edit the official
parquet files.

## 3. Audit before spending GPU time

```bash
python scripts/verify_uavflow_remote.py \
  --sim-root "$DATA_ROOT/UAV-Flow-Sim" \
  --depth-root "$DATA_ROOT/UAV-Flow-Sim-Depth" \
  --da3-checkpoint "$PWD/checkpoints/track4world_da3.pth" \
  --qwen-model "$MODEL_ROOT/qwen3.5-2b" \
  --t5-model "$MODEL_ROOT/t5-base"
```

The audit requires exactly 21 parquet shards, 10,109 unique depth episodes,
the canonical `hybrid/` and `replay/` trees, all model configs and seven
instruction overrides.

## 4. Export the portable path contract

```bash
export PROJECT_ROOT=$PWD
export UAVFLOW_SIM_ROOT="$DATA_ROOT/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$DATA_ROOT/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$PWD/checkpoints/track4world_da3.pth"
export QWEN_MODEL="$MODEL_ROOT/qwen3.5-2b"
export T5_MODEL="$MODEL_ROOT/t5-base"
export PYTHON_BIN="$(command -v python)"
export TORCHRUN_BIN="$(command -v torchrun)"
export OUTPUT_ROOT=$PWD/results/remote_ablation
```

The launcher recognizes both the published `hybrid/ + replay/` layout and the
older local consolidated layout. Published depth automatically uses hybrid as
the primary source, replay as fallback and the packaged correction file.

## 5. Smoke test

The production effective batch is 24. A two-GPU smoke test deliberately uses
global batch 4 and is not a scientific result:

```bash
CUDA_DEVICES=0,1 NPROC=2 GLOBAL_BATCH_SIZE=4 BATCH_SIZE=2 \
MAX_TRAJECTORIES=20 STAGE1_EPOCHS=1 RUN_STAGE2=0 RUN_IDS=B0,F3 \
OUTPUT_ROOT=/tmp/uavflow_smoke \
bash experiments/uavflow_remote_ablation/run_remote.sh
```

## 6. Compact 21-run experiment

Production defaults are four GPUs, batch 6/GPU, no accumulation and global
batch 24. This lets all 21 compact cells occupy 84 GPUs in one wave. Stage 1
and Stage 2 each train five physical epochs.

```bash
CUDA_DEVICES=0,1,2,3 \
NPROC=4 GLOBAL_BATCH_SIZE=24 BATCH_SIZE=6 \
bash experiments/uavflow_remote_ablation/run_server.sh
```

The command above runs all cells sequentially on one four-GPU worker. On a
cluster, submit one four-GPU array job per cell using the matching scheduler:

```bash
# Slurm: 21 jobs x 4 GPUs = 84 GPUs in one wave
sbatch experiments/uavflow_remote_ablation/submit_slurm_compact.sh

# PBS Pro / OpenPBS
qsub -V experiments/uavflow_remote_ablation/submit_pbs_compact.sh

# IBM LSF
bsub < experiments/uavflow_remote_ablation/submit_lsf_compact.sh
```

On a single machine, enumerate each independent four-GPU group. The example
below runs two cells concurrently and automatically schedules the remaining
cells when a worker becomes free:

```bash
GPU_GROUPS='0,1,2,3;4,5,6,7' \
bash experiments/uavflow_remote_ablation/run_local_gpu_pool.sh
```

All four launch modes use the same ordered list in `compact_cells.sh`; array
index 0 is `B0` and index 20 is `C5_F10HB`. Every cell preserves global batch
24 and writes to a separate output directory.

The compact matrix contains 16 main-effect/control rows and five selected
interactions. See `docs/uavflow_remote_ablation_ofat.md` for the exact list and
`docs/uavflow_remote_ablation_detailed.md` for data/loss semantics.

Each stage writes its resolved configuration, split, logs, epoch checkpoints,
run state and `_SUCCESS`. Re-running the same command skips completed stages
and resumes an interrupted stage from its latest checkpoint.

Inspect progress:

```bash
python experiments/uavflow_remote_ablation/status.py "$OUTPUT_ROOT"
```

Run a selected subset:

```bash
RUN_IDS=B0,C3_W3HB,C5_F10HB \
bash experiments/uavflow_remote_ablation/run_server.sh
```

## 7. Optional compute-rich matrix

The interaction-aware detailed launcher contains 64 deduplicated cells. It is
not a full Cartesian product. One eight-GPU node can run it sequentially:

```bash
MATRIX_ROOT=$PWD/results/uavflow_compute_rich \
bash experiments/uavflow_remote_ablation/run_compute_rich_matrix.sh
```

Across eight separate eight-GPU nodes, assign one shard per node:

```bash
MATRIX_SHARD_COUNT=8 MATRIX_SHARD_INDEX=<0..7> \
MATRIX_ROOT=$PWD/results/uavflow_compute_rich \
bash experiments/uavflow_remote_ablation/run_compute_rich_matrix.sh
```

All nodes must see the same shared `MATRIX_ROOT` and dataset paths. A file lock
creates one immutable split, and each manifest row records the interaction
blocks that selected it.

## 8. What is and is not aligned

- Dataset split: fixed stratified 95/5 split for all ablations.
- Batch: global 24, chosen between OpenVLA-UAV global 32 and GAM post-train
  global 12; it exactly matches GAM pre-training.
- Stage 1: constant schedule except the explicit `S1COS` control.
- Stage 2: 500-step warmup plus cosine, held fixed for every row.
- Qwen/T5/DA3 language and visual backbones are loaded from local paths; no
  training job silently downloads a different revision.
- Final comparison with OpenVLA-UAV should retrain the selected configuration
  on 100% training episodes after ablation selection. The 5% validation split
  exists for controlled model selection and is not claimed as OpenVLA's split.
