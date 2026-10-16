# NVIDIA remote training handoff

This release maintains three metric-depth experiments. Every experiment is
independent and can use one complete eight-device node via the existing DDP trainer.

For Huawei Ascend nodes, use [NPU_SETUP.md](../NPU_SETUP.md). Do not
install this CUDA environment on top of a vendor torch_npu image.

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
export OUTPUT_ROOT=$PWD/results/metric_release
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
python -m torch.distributed.run --standalone --nproc_per_node=8 experiments/metric_jobs/01_gam.py
python -m torch.distributed.run --standalone --nproc_per_node=8 experiments/metric_jobs/02_current_bank.py
python -m torch.distributed.run --standalone --nproc_per_node=8 experiments/metric_jobs/03_rgbwarp.py
```

Submit each line as an independent node job, not all three on one node.
Run the data/runtime audit above first. Default per-device batch4 gives global32
on eight devices; actual memory feasibility remains to be measured. Defaults
are five epochs constant LR, no Stop. Resume explicitly with `--resume` and the
same world size/data/batch. Outputs are `$OUTPUT_ROOT/{gam,current_bank,rgbwarp}`.
Exact loss/pose/readout differences are documented in
[METRIC_EXPERIMENTS.md](METRIC_EXPERIMENTS.md).

## 5. Minimal smoke job

```bash
CUDA_VISIBLE_DEVICES=0,1 OUTPUT_ROOT="$OUTPUT_ROOT/smoke" \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  experiments/metric_jobs/02_current_bank.py --max-trajectories 20 \
  --set training.max_epochs=1 --set training.batch_size=1
```

This checks construction and one tiny training run; it is not a scientific
result and must use a separate `OUTPUT_ROOT` if a production run already
exists.
