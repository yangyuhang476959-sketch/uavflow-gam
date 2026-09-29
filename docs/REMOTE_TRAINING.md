# Remote training handoff

This release runs the ten-cell VLA--GAM matrix. Every experiment is one Python
command intended for one complete 8x64GB GPU node.

## 1. Environment

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git UAVFlow-GAM
cd UAVFlow-GAM
conda env create -f environment-uavflow.yml
conda activate uav-gam
python scripts/setup_uavflow_remote.py
```

The setup pins PyTorch 2.5.1/cu124, NumPy 1.26.4, SciPy 1.15.3 and every DA3
eager-import dependency in one resolver transaction. It checks the NumPy /
SciPy / PyTorch ABI and imports DA3 before returning successfully.

## 2. Data and checkpoints

Mainland-accessible download:

```bash
python scripts/download_uavflow_assets.py \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth
```

The official RGB/parquet data and public model weights use the HF mirror. The
derived depth archive uses ModelScope and is extracted into:

```text
data_remote/UAV-Flow-Sim-Depth/
  replay/<episode>/depth.npy                 # 6,990 episodes
  hybrid/{person,robotic_dog,vehicle}/*.npz # 3,119 episodes
  metadata/instruction_overrides.json
```

The loader gives hybrid labels priority and falls back to replay. These 10,109
episodes are the required calibrated/replayed plus dynamic-object-corrected
depth set; the earlier uncalibrated root is not used.

## 3. Paths and audit

```bash
export PROJECT_ROOT=$PWD
export UAVFLOW_SIM_ROOT=$PWD/data_remote/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=$PWD/data_remote/UAV-Flow-Sim-Depth
export DA3_CHECKPOINT=$PWD/checkpoints/track4world_da3.pth
export QWEN_MODEL=$PWD/checkpoints/qwen3.5-2b
export T5_MODEL=$PWD/checkpoints/t5-base
export OUTPUT_ROOT=$PWD/results/vla_gam_matrix_v2
export NPROC=8
export GLOBAL_BATCH_SIZE=32

python scripts/verify_uavflow_remote.py \
  --sim-root "$UAVFLOW_SIM_ROOT" \
  --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" \
  --qwen-model "$QWEN_MODEL" \
  --t5-model "$T5_MODEL"
```

The audit requires 21 parquet shards, 6,990 replay episodes, 3,119 hybrid
episodes, seven instruction corrections and working pinned imports.

## 4. Submit one command per node

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

Do not submit all ten commands into one node. The cluster should enqueue each
line as an independent 8-GPU job. Every entrypoint performs:

1. runtime and full depth-layout audit;
2. locked creation/reuse of one shared split;
3. Stage 1: 10 epochs action + feature + depth, constant LR;
4. selection of `best_action.pt` by minimum H=1 validation Action loss;
5. Stage 2: 10 epochs joint policy + ordered concat-MLP Stop, cosine LR;
6. exact same-stage optimizer/scheduler/data-cursor resume after interruption.

Completed stages have `_SUCCESS` and are skipped on re-submission. Results are
stored as `results/vla_gam_matrix_v2/<ID>/{stage1,stage2_stop}`.

## 5. Minimal smoke job

```bash
STAGE1_EPOCHS=1 NPROC=2 GLOBAL_BATCH_SIZE=4 CUDA_DEVICES=0,1 \
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py \
  --stage stage1 --max-trajectories 20
```

This checks construction and one tiny training run; it is not a scientific
result and must use a separate `OUTPUT_ROOT` if a production run already
exists.
