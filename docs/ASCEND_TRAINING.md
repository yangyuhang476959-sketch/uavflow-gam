# Ascend NPU training handoff

The same ten matrix entrypoints support NVIDIA CUDA and Huawei Ascend NPU.
The scientific matrix, split, loss weights, stage lengths and checkpoint
selection are unchanged. Only the hardware runtime is selected differently.

## 1. Start from a vendor-matched image

Use an Ascend PyTorch image whose **CANN, driver, firmware, torch,
torchvision and torch_npu are already matched**. Do not create the CUDA conda
environment and do not install the public CUDA PyTorch wheel into this image.
The exact supported version triplet depends on the cluster's CANN release.

Load CANN using the path provided by the cluster, commonly:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Then clone the release and install only model/Python dependencies:

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git UAVFlow-GAM
cd UAVFlow-GAM
python scripts/setup_uavflow_ascend.py
```

The setup refuses to replace torch or torch_npu. It checks NPU tensor
execution, the pinned NumPy/SciPy ABI and DA3 imports.

## 2. Download and configure data

Use the same asset downloader and paths as the CUDA handoff:

```bash
python scripts/download_uavflow_assets.py \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth

export PROJECT_ROOT=$PWD
export UAVFLOW_SIM_ROOT=$PWD/data_remote/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=$PWD/data_remote/UAV-Flow-Sim-Depth
export DA3_CHECKPOINT=$PWD/checkpoints/track4world_da3.pth
export QWEN_MODEL=$PWD/checkpoints/qwen3.5-2b
export T5_MODEL=$PWD/checkpoints/t5-base
export OUTPUT_ROOT=$PWD/results/vla_gam_matrix_v2
```

## 3. Ascend runtime variables

One experiment occupies one eight-NPU node:

```bash
export UAVFLOW_ACCELERATOR=npu
export DEVICE_IDS=0,1,2,3,4,5,6,7
export NPROC=8
export GLOBAL_BATCH_SIZE=32
export AMP_DTYPE=bf16
export QWEN_ATTN_IMPLEMENTATION=eager
export HCCL_ASYNC_ERROR_HANDLING=1
```

BF16 matches the existing NVIDIA matrix and is supported by the target Atlas
910B1 stack: Qwen parameters and outer AMP are BF16, while newly inserted LoRA
parameters remain FP32 on both platforms. Qwen eager attention remains the
conservative backend default. `QWEN_ATTN_IMPLEMENTATION=sdpa` may be tested for
performance after the mandatory smoke, but must not be mixed within one
reported comparison matrix.

Run the full preflight audit before queueing expensive jobs:

```bash
PYTHONPATH=src:. python scripts/verify_uavflow_remote.py \
  --accelerator npu \
  --sim-root "$UAVFLOW_SIM_ROOT" \
  --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" \
  --qwen-model "$QWEN_MODEL" \
  --t5-model "$T5_MODEL"
```

## 4. Smoke test, then submit

Run one small R1 construction/training smoke before launching all ten:

```bash
OUTPUT_ROOT=$PWD/results/ascend_smoke \
STAGE1_EPOCHS=1 NPROC=8 GLOBAL_BATCH_SIZE=8 \
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py \
  --stage stage1 --max-trajectories 20
```

When it succeeds, delete/ignore the smoke output and submit the ten normal
Python commands listed in `experiments/uavflow_remote_ablation/README.md`, one
command per eight-NPU node.

## 5. What the compatibility layer changes

- imports `torch_npu` before Transformers and DA3;
- uses `npu:<local_rank>` and HCCL for DDP;
- uses TorchNPU AMP/GradScaler and reports NPU peak memory;
- disables CUDA/Triton FlexAttention on NPU;
- defaults Qwen to eager attention and NPU AMP to BF16 on Atlas 910B1;
- disables CUDA-oriented DataLoader pinned memory on NPU;
- records accelerator, device IDs, AMP dtype and attention implementation in
  the resolved configuration/run state.

It does **not** use `transfer_to_npu` global monkey migration, does not alter
model weights, and does not change the ten experiment definitions.

Because this repository is developed on NVIDIA hardware, the committed tests
validate routing and CPU/CUDA regression behavior but cannot certify every
operator on the destination CANN/torch_npu version. The mandatory R1 smoke is
the final operator-compatibility gate.

CUDA and Ascend runs implement the same objective and both use BF16, but their
backend kernels are not bitwise equivalent. Keep accelerator, AMP dtype and
attention backend in the experiment metadata, and do not splice checkpoints
from different accelerator stacks into one exact-resume run.
