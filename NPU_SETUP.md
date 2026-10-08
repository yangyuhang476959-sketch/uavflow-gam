# Ascend NPU handoff

The code was previously validated on **Ascend 910B2, aarch64, Driver 25.2.1,
CANN 9.0.0, Python 3.11.15, PyTorch 2.7.1 and torch_npu 2.7.1.post4**.
This is a validation record, **not a universal installation prescription**.

The receiving engineer must prepare the hardware-dependent environment according
to Huawei's [official compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
and the target hardware's installation guide: Driver/Firmware, CANN and matching
ops, PyTorch/torch_npu/torchvision, and Triton-Ascend for the Qwen FLA path.
Different A2/A3 hardware or software versions require manual compatibility checks.
This guide does not install or modify those components, or use the automatic
bootstrap scripts. The current matrix audit still enforces the validated
PyTorch/torch_npu pair; a different pair requires project compatibility review.

The steps below install only the project environment, prepare data, and verify
the model. Run them from one Bash session. This document was prepared on an
NVIDIA host; no new NPU validation was performed here.

## 1. Python environment and project packages

Use an existing compatible environment supplied by the engineer, or create one
with an existing architecture-matched Conda installation:

```bash
conda create -n uavflow-npu python=3.11 pip -y
conda activate uavflow-npu
```

Install the engineer-confirmed PyTorch/torch_npu/torchvision and Triton-Ascend
packages into **this environment** following their official instructions. A new
Conda environment does not inherit packages from an administrator's other Python.
Then load the exact existing CANN installation selected by the engineer:

```bash
# Replace the placeholder with the confirmed existing file.
export CANN_ENV_FILE=/path/to/selected/CANN/set_env.sh
source "$CANN_ENV_FILE"
npu-smi info
python -c 'import torch, torch_npu, torchvision, triton; print(torch.__version__, torch_npu.__version__, torchvision.__version__, triton.__version__)'

git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export PROJECT_ROOT="$PWD"

python -m pip install -c constraints-ascend.txt -r requirements-ascend.txt
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
git -C Depth-Anything-3 checkout --detach 2c21ea849ceec7b469a3e62ea0c0e270afc3281a
python -m pip install --no-deps -e Depth-Anything-3

export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
mkdir -p "$PROJECT_ROOT/.ascend"
git clone https://github.com/fla-org/flash-linear-attention.git "$FLA_ASCEND_DIR"
git -C "$FLA_ASCEND_DIR" checkout --detach 9f38d24980c46d46bd38614e743cdacd21906578
export PYTHONPATH="$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
python -m pip check
```

`requirements-ascend.txt` pins the project packages, including NumPy/SciPy and
DA3's import dependencies. Do not install the CUDA environment file or optional
`pycolmap` requirements on this training path. Review pip's proposed changes if
it reports conflicts; do not let it replace the engineer-provided runtime.
Clone dependencies once; for transferred checkouts, verify their pinned commits.
Re-activate the environment, source CANN, and export the variables above in each
new shell/job. CANN's `set_env.sh` provides its library/tool/ops paths; do not
guess `LD_LIBRARY_PATH` or `ASCEND_OPP_PATH`.

## 2. Prepare data and pretrained weights

The existing helper downloads RGB/parquet and weights through the HF mirror,
downloads our depth archive from ModelScope, and verifies/extracts that archive:

```bash
python -m pip install -c constraints-ascend.txt modelscope==1.38.1
python scripts/download_uavflow_assets.py \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth --workers 2
```

For an offline node, transfer the same files/directories instead. If the depth
archive was transferred but not extracted:

```bash
python scripts/extract_uavflow_depth.py \
  --dataset-root "$PROJECT_ROOT/data_remote/UAV-Flow-Sim-Depth-Archive" \
  --output "$PROJECT_ROOT/data_remote/UAV-Flow-Sim-Depth"
```

Required layout:

```text
data_remote/UAV-Flow-Sim/train-*-of-00021.parquet       # 21 unchanged shards
data_remote/UAV-Flow-Sim-Depth/
  replay/<episode>/depth.npy                         # 6,990 episodes
  hybrid/{person,robotic_dog,vehicle}/<episode>.npz    # 3,119 episodes
  metadata/{instruction_overrides.json,episodes.csv,source_summary.json,shards.json}
checkpoints/track4world_da3.pth
checkpoints/qwen3.5-2b/                               # complete model/processor
checkpoints/t5-base/                                 # complete model/tokenizer
```

The helper's sources are `wangxiangyu0814/UAV-Flow-Sim`, `Qwen/Qwen3.5-2B`,
`google-t5/t5-base`, `SeonghuJeon/3da-libero-training-assets` (DA3 checkpoint),
and `acetaffy123/UAV-Flow-Sim-Depth` (ModelScope). Network access and permissions
require manual verification. All three pretrained assets are required by the
current launch audit; trained policy checkpoints are generated by training.

Hybrid depth takes priority over replay. Do not flatten the two formats. Keep
parquet unchanged: the loader applies seven instruction corrections from metadata
and handles RGB/trajectory preprocessing; the runner creates the shared split.

```bash
export UAVFLOW_SIM_ROOT="$PROJECT_ROOT/data_remote/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$PROJECT_ROOT/data_remote/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$PROJECT_ROOT/checkpoints/track4world_da3.pth"
export QWEN_MODEL="$PROJECT_ROOT/checkpoints/qwen3.5-2b"
export T5_MODEL="$PROJECT_ROOT/checkpoints/t5-base"
export OUTPUT_ROOT="$PROJECT_ROOT/results/vla_gam_matrix_v2"
```

## 3. Smoke tests

Run on the receiving NPU node after environment preparation:

```bash
python - <<'PY'
import numpy as np, scipy.special, torch, torch_npu, triton, fla
assert torch.npu.is_available()
print('NPU devices:', torch.npu.device_count())
assert torch.from_numpy(np.zeros(4, dtype=np.float32)).shape == (4,)
scipy.special.expit(np.array([0.0]))
x = torch.randn(64, 64, device='npu', dtype=torch.bfloat16, requires_grad=True)
loss = (x @ x).float().square().mean()
loss.backward()
torch.npu.synchronize()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs()
from depth_anything_3.api import DepthAnything3
print('Imports, NumPy/SciPy and NPU BF16 forward/backward PASS')
PY

python scripts/verify_uavflow_remote.py --accelerator npu \
  --sim-root "$UAVFLOW_SIM_ROOT" --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" --qwen-model "$QWEN_MODEL" --t5-model "$T5_MODEL"

OUTPUT_ROOT="$PROJECT_ROOT/results/npu_smoke" NPROC=1 GLOBAL_BATCH_SIZE=4 \
DEVICE_IDS=0 STAGE1_EPOCHS=1 \
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py \
  --stage stage1 --max-trajectories 20
```

The audit verifies package versions/data counts and randomly loads replay/hybrid
episodes. The final command tests the real R1 forward/backward on 20 trajectories;
imports alone are insufficient. This is a smoke run, not a scientific result.
The existing NPU runner uses BF16; support and actual memory/performance on another
platform must be verified before production. Keep smoke outputs separate.

## 4. Run, resume and validate

One matrix experiment per eight-NPU node; choose the corresponding Python entry
from [the ten-cell list](experiments/uavflow_remote_ablation/README.md). Example R1:

```bash
export NPROC=8 GLOBAL_BATCH_SIZE=32 DEVICE_IDS=0,1,2,3,4,5,6,7
export STAGE1_EPOCHS=10 STAGE2_EPOCHS=10
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py
```

Stage 1: 10 epochs, constant LR; Stage 2: 10 epochs, cosine LR with Stop.
Re-run the **same command and paths/split/batch configuration** to resume:
the runner restores same-stage checkpoint state and skips `_SUCCESS` stages.
Stage 2 initializes from Stage 1's best validation Action-loss checkpoint.
Outputs are `$OUTPUT_ROOT/<ID>/{stage1,stage2_stop}`. Validation normally runs
every 2,000 steps with 100 batches; epoch checkpoint archives are enabled.

Standalone full validation, using the saved configuration:

```bash
RUN_DIR="$OUTPUT_ROOT/R1/stage1"
ASCEND_RT_VISIBLE_DEVICES=0 python experiments/uavflow_predictor_idm/train.py \
  --config "$RUN_DIR/resolved_config.yaml" --resume "$RUN_DIR/best_action.pt" \
  --eval-only --eval-domain mixed \
  --set "training.results_dir=$OUTPUT_ROOT/R1/validation" \
  --set training.eval_max_batches=0
```

Change the ID/stage/checkpoint for other runs; `eval_max_batches=100` restores
bounded validation. Results are `eval_ckpt_<step>.json`. Transferred training
configs/checkpoints retain saved paths, so cross-machine resume needs manual path
review. Closed-loop UE evaluation is **not validated on Ascend in this release**;
use the existing simulation environment on its supported host.
