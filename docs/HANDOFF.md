# UAVFlow-GAM handoff

This is the shortest path from a clean server to the compact 25-cell
experiment. For rationale and exact ablation semantics, read
`REMOTE_TRAINING.md` and `uavflow_remote_ablation_ofat.md`.

## 1. Clone and create the environment

```bash
git clone ssh://git@ssh.github.com:443/yangyuhang476959-sketch/uavflow-gam.git UAVFlow-GAM
cd UAVFlow-GAM

conda env create -f environment-uavflow.yml
conda activate uav-gam
bash scripts/setup_uavflow_remote.sh
```

The setup installs the pinned PyTorch stack and the pinned DA3 source checkout.
It intentionally omits DA3's optional rendering dependencies.

## 2. Download data and frozen models

```bash
export PROJECT_ROOT="$PWD"
export DATA_ROOT="$PROJECT_ROOT/data_remote"
export MODEL_ROOT="$PWD/checkpoints"
export HF_ENDPOINT=https://hf-mirror.com
export DEPTH_HUB=modelscope
export DEPTH_DATASET_REPO=acetaffy123/UAV-Flow-Sim-Depth

bash scripts/download_uavflow_assets.sh
```

The script downloads the official UAV-Flow-Sim parquet dataset, Qwen3.5-2B,
T5-base and Track4World DA3 weights, then downloads and extracts the derived
depth sidecars from ModelScope. It does not modify official parquet files.

## 3. Export the path contract and audit it

```bash
export UAVFLOW_SIM_ROOT="$DATA_ROOT/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$DATA_ROOT/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$PWD/checkpoints/track4world_da3.pth"
export QWEN_MODEL="$MODEL_ROOT/qwen3.5-2b"
export T5_MODEL="$MODEL_ROOT/t5-base"
export PYTHON_BIN="$(command -v python)"
export TORCHRUN_BIN="$(command -v torchrun)"
export OUTPUT_ROOT="$PROJECT_ROOT/results/remote_ablation"

python scripts/verify_uavflow_remote.py \
  --sim-root "$UAVFLOW_SIM_ROOT" \
  --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" \
  --qwen-model "$QWEN_MODEL" \
  --t5-model "$T5_MODEL"
```

Do not start production training unless the audit prints `remote audit: OK`.

## 4. Smoke test

```bash
CUDA_DEVICES=0,1 NPROC=2 GLOBAL_BATCH_SIZE=4 BATCH_SIZE=2 \
MAX_TRAJECTORIES=20 STAGE1_EPOCHS=1 RUN_STAGE2=0 RUN_IDS=B0 \
OUTPUT_ROOT=/tmp/uavflow_smoke \
bash experiments/uavflow_remote_ablation/run_remote.sh
```

This checks the installation only. It is not a comparable scientific run.

## 5. Production launch

Every compact cell uses four GPUs, batch 6/GPU and global batch 24. Stage 1
trains action/feature/depth for five epochs; Stage 2 resumes Stage 1 and jointly
trains the Stop head for five more epochs.

Use exactly one launcher matching the cluster:

```bash
# Slurm: 25 independent four-GPU jobs
sbatch experiments/uavflow_remote_ablation/submit_slurm_compact.sh

# PBS Pro / OpenPBS
qsub -V experiments/uavflow_remote_ablation/submit_pbs_compact.sh

# IBM LSF
bsub < experiments/uavflow_remote_ablation/submit_lsf_compact.sh

# Single host: semicolon separates independent four-GPU workers
GPU_GROUPS='0,1,2,3;4,5,6,7' \
bash experiments/uavflow_remote_ablation/run_local_gpu_pool.sh
```

Cluster-specific partition/account/wall-time directives belong in the header
of the selected scheduler script. The experiment enumeration must remain in
`compact_cells.sh` so all schedulers use the same 25 cells.

To run one cell manually:

```bash
bash experiments/uavflow_remote_ablation/run_compact_cell.sh 0  # B0
bash experiments/uavflow_remote_ablation/run_compact_cell.sh 20 # C5_F10HB
bash experiments/uavflow_remote_ablation/run_compact_cell.sh 21 # CA1_HB (current geometry action read)
```

## 6. Resume and inspect

Re-submit the same command after interruption. Completed stages contain
`_SUCCESS` and are skipped; interrupted stages resume from their newest epoch
checkpoint, including optimizer, scaler, scheduler, epoch and data cursor.

```bash
python experiments/uavflow_remote_ablation/status.py "$OUTPUT_ROOT"
```

Each cell records `resolved_config.yaml`, split metadata, logs, epoch
checkpoints and `run_state.txt`. The latter includes code revision, dirty-tree
state and dataset-manifest hashes.

## Important files

- `environment-uavflow.yml`: Conda environment shell.
- `requirements-uavflow.txt`: pinned Python dependencies.
- `scripts/setup_uavflow_remote.sh`: PyTorch and DA3 installation.
- `scripts/download_uavflow_assets.sh`: datasets and frozen weights.
- `scripts/verify_uavflow_remote.py`: fail-fast data/model audit.
- `experiments/uavflow_remote_ablation/base.yaml`: shared model/data/loss setup.
- `experiments/uavflow_remote_ablation/compact_matrix.tsv`: human-readable matrix.
- `experiments/uavflow_remote_ablation/compact_cells.sh`: scheduler index order.
- `experiments/uavflow_remote_ablation/run_remote.sh`: restartable two-stage runner.
- `docs/uavflow_remote_ablation_ofat.md`: compact matrix rationale.
- `docs/uavflow_remote_ablation_detailed.md`: depth/horizon/loss definitions.
