# UAVFlow-GAM

VLA+Future 和 VLA+GAM 的 UAVFlow-Open、IndoorUAV 训练代码。UAVFlow-Open 训练 **15 轮**，IndoorUAV 训练 **5 轮**。

## 1. 环境准备

下面的版本是之前使用的 Ascend 910B2 参考环境：Python 3.11、CANN 9.0.0、PyTorch 2.7.1、torch-npu 2.7.1.post4。先确认机器的驱动和 CANN 版本与其兼容。其他版本的安装和排查见 [NPU_SETUP.md](NPU_SETUP.md)。

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export PROJECT_ROOT="$PWD"

conda create -n uavflow-npu python=3.11 pip -y
conda activate uavflow-npu
source /usr/local/Ascend/cann-9.0.0/set_env.sh  # 按实际 CANN 路径修改
npu-smi info

python -m pip install --no-deps PyYAML==6.0.2 --index-url https://pypi.org/simple
python -m pip install 'torch==2.7.1+cpu' 'torchvision==0.22.1' \
  --index-url https://download.pytorch.org/whl/cpu
cat > platform-constraints.txt <<'CONSTRAINTS'
torch==2.7.1+cpu
torch-npu==2.7.1.post4
torchvision==0.22.1
CONSTRAINTS
export PLATFORM_CONSTRAINTS="$PROJECT_ROOT/platform-constraints.txt"
python -m pip install -c "$PLATFORM_CONSTRAINTS" 'torch-npu==2.7.1.post4' \
  --index-url https://pypi.org/simple
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt \
  -r requirements-ascend.txt
```

安装 **Triton-Ascend 和 Flash Linear Attention（FLA）**。Qwen 的 FLA 路径使用它们加速。

```bash
export TRITON_ASCEND_TARGET="$PROJECT_ROOT/.ascend/triton"
export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
mkdir -p "$TRITON_ASCEND_TARGET"
python -m pip install --target "$TRITON_ASCEND_TARGET" --ignore-installed --no-deps \
  'triton-ascend==3.2.1' pybind11 --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple

git clone https://github.com/fla-org/flash-linear-attention.git "$FLA_ASCEND_DIR"
git -C "$FLA_ASCEND_DIR" checkout --detach 9f38d24980c46d46bd38614e743cdacd21906578

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
git -C Depth-Anything-3 checkout --detach 2c21ea849ceec7b469a3e62ea0c0e270afc3281a
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt \
  'hatchling>=1.25' 'hatch-vcs>=0.4' editables --index-url https://pypi.org/simple
python -m pip install --no-build-isolation --no-deps -e ./Depth-Anything-3
```

每次打开新终端，激活环境并设置运行变量：

```bash
conda activate uavflow-npu
source /usr/local/Ascend/cann-9.0.0/set_env.sh  # 按实际 CANN 路径修改
cd /path/to/uavflow-gam  # 修改为仓库路径
export PROJECT_ROOT="$PWD"
export TRITON_ASCEND_TARGET="$PROJECT_ROOT/.ascend/triton"
export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
export PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_ACCELERATOR=npu UAVFLOW_QWEN_FLA_NPU=1 UAVFLOW_DISABLE_FLEX_ATTENTION=1

python - <<'PY'
import os
from pathlib import Path
import torch, torch_npu, triton, fla
assert torch.npu.is_available()
assert Path(triton.__file__).resolve().is_relative_to(Path(os.environ['TRITON_ASCEND_TARGET']).resolve())
print('NPU:', torch.npu.device_count(), 'Triton:', triton.__version__, 'FLA:', fla.__file__)
PY
```

环境导入检查通过后，还应在目标机器验证实际 NPU 前向和反向。详细检查步骤见 [NPU_SETUP.md](NPU_SETUP.md)。

## 2. Data Preparation

### UAVFlow-Open

下载数据、深度和预训练权重。数据默认存入 `data/`，权重存入 `checkpoints/`。

```bash
python -m pip install -c "$PROJECT_ROOT/platform-constraints.txt" -c constraints-ascend.txt modelscope==1.38.1
python scripts/download_uavflow_assets.py \
  --data-root "$PROJECT_ROOT/data" --model-root "$PROJECT_ROOT/checkpoints" \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth --workers 2
```

训练使用仓库内固定划分、Open 原始指令映射和动作归一化统计。路径不同时，在 `experiments/uavflow_paper_ablation/configs/` 的对应 YAML 中修改数据及权重路径。权重哈希和数据说明见 [docs/ABLATION_NPU.md](docs/ABLATION_NPU.md)。

### IndoorUAV

准备好 IndoorUAV 数据目录，包含 `train.csv`、`test_seen.csv`、`test_unseen.csv`、`vla_ins/`、`without_screenshot/`、`rgb/` 和 `depth/`。其中 `rgb/` 和 `depth/` 保存轨迹逐帧图像及深度。将下面路径换成实际目录：

```bash
export INDOOR_DATA_ROOT=/path/to/indoor-data
python -m experiments.indooruav_warp.prepare \
  --data-root "$INDOOR_DATA_ROOT" --output prepared/indoor_future --variant vla_future
python -m experiments.indooruav_warp.prepare \
  --data-root "$INDOOR_DATA_ROOT" --output prepared/indoor_gam --variant vla_gam
```

这一步复用已记录的轨迹分组划分、检查来源并生成动作统计和训练配置。

## 3. Run

`NPROC` 是每台机器使用的 NPU 数量。以下以单机 8 卡为例，四条命令分别对应独立实验。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# UAVFlow-Open · 15 轮
NPROC=8 EPOCHS=15 bash scripts/run_ablation_npu.sh experiments/uavflow_paper_ablation/configs/vla_future.yaml
NPROC=8 EPOCHS=15 bash scripts/run_ablation_npu.sh experiments/uavflow_paper_ablation/configs/vla_gam.yaml

# IndoorUAV · 5 轮
NPROC=8 EPOCHS=5 bash scripts/run_ablation_npu.sh prepared/indoor_future/vla_future.yaml
NPROC=8 EPOCHS=5 bash scripts/run_ablation_npu.sh prepared/indoor_gam/vla_gam.yaml
```

支持总计 1、2、3、4、6、8、12、24 卡，全局 batch 为 24。2、4、6、8、12、24 卡会重新分组，损失归约和浮点结果可能与原三卡不同。

三机各 8 卡时，在每台机器设置相同的主节点地址，`NODE_RANK` 分别设为 0、1、2，并在每台运行相同的实验命令：

```bash
export NNODES=3 NPROC=8 MASTER_ADDR=192.0.2.10 MASTER_PORT=29500 # 替换为主节点地址
export NODE_RANK=0  # 另两台分别为 1、2
EPOCHS=15 bash scripts/run_ablation_npu.sh experiments/uavflow_paper_ablation/configs/vla_future.yaml
```

多机需要在各节点的相同路径访问仓库、配置、数据和权重，并共享结果目录。续训时在命令末尾加对应的 `last.pt` 路径，`EPOCHS` 填总目标轮数。每轮、last 和 best 均保存完整训练状态和优化器。

NPU 环境和多卡启动方案沿用已有适配，当前两项消融还未在目标 24 卡 NPU 集群实跑。参数与对齐记录见 [docs/ABLATION_NPU.md](docs/ABLATION_NPU.md)。
