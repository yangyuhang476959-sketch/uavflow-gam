# Ascend NPU 交接说明

语言：简体中文 | [English](NPU_SETUP.md)

本仓库已在 **Ascend 910B2、aarch64、Driver 25.2.1、CANN 9.0.0、Python 3.11.15、PyTorch 2.7.1、torch_npu 2.7.1.post4** 上验证。这是验证记录，不代表其他 A2/A3 或驱动/CANN 组合可直接照搬。

接收方须先按华为[官方兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)及目标设备安装指南，确认 Driver/Firmware、CANN/ops、Python、PyTorch、torch_npu、torchvision 和 Triton-Ascend 的配套版本。本说明不自动安装或修改系统驱动、固件、CANN。

## 1. Python 环境与平台包

使用运维提供的兼容环境，或在已有匹配架构的 Conda 中创建：

```bash
conda create -n uavflow-npu python=3.11 pip -y
conda activate uavflow-npu
```

### 已验证 910B2 + CANN 9.0.0 参考安装命令

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh

python -m pip install \
  'torch==2.7.1+cpu' 'torchvision==0.22.1' \
  --index-url https://download.pytorch.org/whl/cpu

cat > platform-constraints.txt <<'EOF'
torch==2.7.1+cpu
torch-npu==2.7.1.post4
torchvision==0.22.1
EOF
export PLATFORM_CONSTRAINTS="$PWD/platform-constraints.txt"
python -m pip install -c "$PLATFORM_CONSTRAINTS" \
  'torch-npu==2.7.1.post4' --index-url https://pypi.org/simple
python -c 'import torch, torch_npu, torchvision; print(torch.__version__, torch_npu.__version__, torchvision.__version__)'
```

这组版本只适用于已核实兼容的参考环境。其他硬件或软件版本必须使用对应的官方配套说明；先安装其兼容的 PyTorch/torch_npu/torchvision，再将**实际版本**写入 `platform-constraints.txt`，防止后续 pip 改动平台包。新建 Conda 环境不会继承系统或管理员 Python 环境中的 Torch。

克隆仓库并设置 CANN 与项目路径：

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export PROJECT_ROOT="$PWD"
export CANN_ENV_FILE=/path/to/selected/CANN/set_env.sh
source "$CANN_ENV_FILE"
npu-smi info
python -m pip install --no-deps PyYAML==6.0.2 --index-url https://pypi.org/simple
```

Triton-Ascend 隔离安装、DA3/FLA 固定版本、项目依赖和导入检查按英文版 [NPU_SETUP.md](NPU_SETUP.md) 第 1 节执行。Triton-Ascend 安装在项目 `.ascend/triton`，不覆盖平台依赖。

## 2. 下载数据与模型

目录可由接收方选择：

```bash
export DATA_ROOT=/path/to/datasets
export MODEL_ROOT=/path/to/pretrained-models
export OUTPUT_ROOT=/path/to/training-output
export PLATFORM_CONSTRAINTS="$PROJECT_ROOT/platform-constraints.txt"
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt modelscope==1.38.1
python scripts/download_uavflow_assets.py \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth --workers 2
```

数据包括 21 个原始 Sim parquet 分片；深度目录有 `replay/<episode>/depth.npy` 与 `hybrid/{person,robotic_dog,vehicle>/<episode>.npz`。DA3 权重、Qwen3.5-2B、T5-base 都放入 `$MODEL_ROOT`。原始 parquet 保持不变，loader 会应用七条已复核指令修正。

```bash
export UAVFLOW_SIM_ROOT="$DATA_ROOT/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$DATA_ROOT/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$MODEL_ROOT/track4world_da3.pth"
export QWEN_MODEL="$MODEL_ROOT/qwen3.5-2b"
export T5_MODEL="$MODEL_ROOT/t5-base"
```

## 3. 导入、数据与 NPU smoke test

```bash
export TRITON_ASCEND_TARGET="$PROJECT_ROOT/.ascend/triton"
export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
export PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
python scripts/verify_uavflow_remote.py \
  --sim-root "$UAVFLOW_SIM_ROOT" --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" --qwen-model "$QWEN_MODEL" --t5-model "$T5_MODEL"
```

完整 BF16 前向/反向及 FLA smoke 命令见英文版 NPU_SETUP.md 第 3 节。仅 import 成功不能代替真实训练 smoke。

## 4. 训练命令与实验优先级

四个实验都已提供独立入口。每个作业使用 8 卡、每卡 batch4（global batch32）、10 epoch，学习率全程 constant，不做 warmup：

| ID | 实验设计 | 入口 |
|---|---|---|
| 1 | Pure Qwen + yaw-only numeric Pose Token；5 个 Qwen action slots，不含 GAM/DA3 | `experiments/uavflow_qwen_pure/train.py` |
| 2 | GAM/T5 + metric-depth baseline | `experiments/metric_jobs/01_gam.py` |
| 3 | Bilinear RGB warp；Proposal Action Tokens 注入 Future DA3；DA3 全部 blocks 使用 LoRA | `experiments/metric_jobs/03_rgbwarp.py` |
| 4 | 在 3 上将 Future DA3 的 VLM Action Tokens 换成按 GAM 规则随机初始化的 5 个可学习 Action Tokens（正态标准差 0.02），其余交互及残差头相同 | `experiments/metric_jobs/04_rgbwarp_no_action.py` |

第 4 项的 proposal 仍通过可微 RGB warp 影响 Future 图像；策略推理不读取真实 future image/depth/action。`02_current_bank.py` 保留为额外对照，不属于这四项。优先级为 3、4。

每条命令作为独立作业提交。4 个作业都是 10 epoch、constant LR、warmup 0；每卡 batch4、8 卡共 global batch32：

```bash
export OUTPUT_ROOT="$OUTPUT_ROOT/metric_release"
export UAVFLOW_ACCELERATOR=npu

torchrun --standalone --nproc_per_node=8 experiments/uavflow_qwen_pure/train.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/01_gam.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/03_rgbwarp.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4 \
  --set model.rgbwarp_correction.warp_mode=bilinear

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/04_rgbwarp_no_action.py \
  --set training.max_epochs=10 --set training.lr_schedule=constant \
  --set training.warmup_steps=0 --set training.batch_size=4 \
  --set model.rgbwarp_correction.warp_mode=bilinear
```

仓库中 GAM/校正模型默认仍为 5 epoch constant；命令明确覆盖为 10 epoch，学习率不衰减且无 warmup。历史 R1 benchmark 不等于这些新实验已完成 8 卡性能验证。
