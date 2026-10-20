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

先区分请求的四项和仓库中实际可运行的入口：

| ID | 实验设计 | 当前仓库状态 |
|---|---|---|
| 1 | Pure Qwen + yaw-only numeric Pose Token | 暂不可运行：有 Qwen 模型定义，但缺独立训练入口。旧 Pure-Qwen checkpoint 使用旧 `preprocessed` pose 约定，不是 yaw-only-token 实验。 |
| 2 | GAM/T5 + metric-depth baseline | 可运行：`01_gam.py`；它不是 Pure Qwen。 |
| 3 | Bilinear RGB-warp correction；proposal action tokens 进入 Future DA3；DA3 全部 blocks 使用 LoRA | 可运行：`03_rgbwarp.py`，命令显式覆盖为 bilinear。 |
| 4 | 在 3 上移除 Future DA3 的 proposal action tokens；残差目标来自 warp 与真实 future feature 的差异 | 尚未实现，不能给可执行命令。真实 future feature 只能作训练监督目标，不能成为闭环推理输入；推理时残差必须由当前可用的 current/warp 信息预测。 |

现有 `02_current_bank.py`（Qwen proposal + GAM Future Predictor + Current Bank CA）保留为额外对照，不等同于缺失的 Pure-Qwen 训练器或第 4 项。实验优先级按你的要求为 3、4。

8 卡、每卡 batch4（全局 batch32），10 epoch、cosine、warmup 500。每条命令作为独立作业提交：

```bash
export OUTPUT_ROOT="$OUTPUT_ROOT/metric_release"
export UAVFLOW_ACCELERATOR=npu

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/01_gam.py \
  --set training.max_epochs=10 --set training.lr_schedule=cosine \
  --set training.warmup_steps=500 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/02_current_bank.py \
  --set training.max_epochs=10 --set training.lr_schedule=cosine \
  --set training.warmup_steps=500 --set training.batch_size=4

torchrun --standalone --nproc_per_node=8 experiments/metric_jobs/03_rgbwarp.py \
  --set training.max_epochs=10 --set training.lr_schedule=cosine \
  --set training.warmup_steps=500 --set training.batch_size=4 \
  --set model.rgbwarp_correction.warp_mode=bilinear
```

仓库默认仍为 5 epoch constant；以上命令覆盖为本次交接的 10 epoch cosine。历史 R1 benchmark 不等于这三个新模型已完成 8 卡性能验证。Candidate 4 尚未实现，暂时不能提交其训练命令。
