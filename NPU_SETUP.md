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
bootstrap scripts. The audit checks runtime usability without hard-coding the
historical PyTorch/torch_npu pair; official stack compatibility remains the
receiving engineer's responsibility.

The steps below install only the project environment, prepare data, and verify
the model. Run them from one Bash session. The receiving engineer reported
successful environment, FLA, dataset audit and actual single-card R1 training
on 910B2, including the 2026-10-09 batch-size measurements recorded in
[docs/ascend_cluster.md](docs/ascend_cluster.md). These are not eight-card
training validation; documentation edits/tests here run on an NVIDIA host.

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
# Fresh Python environments may need yaml before torch_npu can be imported.
python -m pip install --no-deps PyYAML==6.0.2 --index-url https://pypi.org/simple
python -c 'import torch, torch_npu, torchvision; print(torch.__version__, torch_npu.__version__, torchvision.__version__)'

git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export PROJECT_ROOT="$PWD"

# EXAMPLE ONLY: previously validated Ascend 910B2 + CANN 9.0.0 stack.
# Before executing, replace these pins with the receiving machine's actual
# officially compatible PyTorch / torch_npu / torchvision versions, including
# local version suffixes. Do not overwrite an existing engineer-provided file.
cat > platform-constraints.txt <<'EOF'
torch==2.7.1+cpu
torch-npu==2.7.1.post4
torchvision==0.22.1
EOF

export PLATFORM_CONSTRAINTS="$PWD/platform-constraints.txt"
test -f "$PLATFORM_CONSTRAINTS"
# Reference ONLY: previously validated 910B2/aarch64, CANN 9.0.0,
# torch_npu 2.7.1.post4. Engineer must approve compatibility before running.
# Isolate the reference package; never resolve/replace base-environment deps.
# Use an empty target for first installation; inspect existing contents before
# reinstalling rather than blindly overwriting a previously working target.
export TRITON_ASCEND_TARGET="$PROJECT_ROOT/.ascend/triton"
mkdir -p "$TRITON_ASCEND_TARGET"
python -m pip install --target "$TRITON_ASCEND_TARGET" --ignore-installed \
  --no-deps 'triton-ascend==3.2.1' pybind11 \
  --only-binary=:all: --index-url https://pypi.org/simple \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt \
  -r requirements-ascend.txt
# Check the base/project stack before DA3 adds its full dependency metadata.
python -m pip check
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
git -C Depth-Anything-3 checkout --detach 2c21ea849ceec7b469a3e62ea0c0e270afc3281a
# Prepare editable-build tools in this environment before disabling isolation.
# Use an accessible index; PyPI is shown here instead of the failing mirror.
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt \
  'hatchling>=1.25' 'hatch-vcs>=0.4' editables \
  --index-url https://pypi.org/simple
python -m pip install --no-build-isolation --no-deps -e ./Depth-Anything-3

export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
mkdir -p "$PROJECT_ROOT/.ascend"
git clone https://github.com/fla-org/flash-linear-attention.git "$FLA_ASCEND_DIR"
git -C "$FLA_ASCEND_DIR" checkout --detach 9f38d24980c46d46bd38614e743cdacd21906578
export PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
python - <<'PY'
import os
from pathlib import Path
import triton
import fla

print("Triton version:", triton.__version__)
print("Triton path:", triton.__file__)
assert Path(triton.__file__).resolve().is_relative_to(
    Path(os.environ["TRITON_ASCEND_TARGET"]).resolve()
), "Wrong Triton loaded: expected the project-local Triton-Ascend target"
print("FLA import OK")
PY
```

The Triton-Ascend command above follows the [official installation guide](https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md),
which supplies an additional package index for 3.2.1; do not assume PyPI alone
hosts that release. `--only-binary=:all:` requires wheels matching this Python,
architecture and platform and fails rather than silently compiling. The
`--target --ignore-installed --no-deps` installation writes only the requested
Triton-Ascend and pybind11 packages into the project-local target, without
resolving or replacing SciPy, Decorator, PyTorch or other environment packages.
The isolated Triton target must be first in `PYTHONPATH` in every new job/shell;
the path assertion above detects accidental use of a different installed Triton.
This deliberately bypasses the full upstream dependency set; verify actual FLA
forward/backward, and do not automatically install missing dependencies or
uninstall a working platform stack. Other
CANN/torch_npu/hardware combinations require a separately approved version, not
blind reuse of this reference. The distribution version and imported
`triton.__version__` may differ; the recorded environment reports 3.2.1 and
3.2.0 respectively. Verify the later FLA forward/backward smoke, not just import.

Both DA3 installation flags are required: `--no-deps` skips its runtime
dependency resolution but **does not disable isolated build dependencies**.
`--no-build-isolation` uses the already-installed build tools, avoiding another
temporary environment trying to download `hatchling`/`editables` from a failing
index. Network access to the chosen index still requires manual verification.

Run `pip check` before installing DA3 to audit the base/project stack. After
the deliberate `--no-deps` DA3 install, a subsequent `pip check` audits DA3's
complete declared dependency set, not just our training path. It may report
missing `e3nn`, `open3d`, `pycolmap`, `xformers`, `fastapi`, or other packages
used outside this path. Do not blindly install them or rerun DA3 installation
without `--no-deps`: review each finding, retain platform constraints, and run
the repository-path import check below and the later runtime/R1 smoke tests.
Conflicts in actual training dependencies still require resolution.
Platform-supplied packages such as `ms-service-profiler` or `te` may also report
their own dependency conflicts. Review those with the platform engineer; such
reports alone neither prove this training path unusable nor prove the environment
safe. Do not force pip to repair the platform packages. The import audit and
real R1 training/FLA smoke below are the required usability checks.

```bash
PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs()
from depth_anything_3.api import DepthAnything3
print("DA3 training-path import OK")
PY
```

`requirements-ascend.txt` pins the project packages, including NumPy/SciPy and
DA3's import dependencies. The engineer must supply `PLATFORM_CONSTRAINTS` with
exact `torch==...`, `torch-npu==...` and `torchvision==...` entries matching the
approved installed stack (including local version suffixes where applicable).
It is a separate local file, not automatically detected or generated. pip must
fail on a conflict rather than replace these versions. Do not install the CUDA environment file or optional
`pycolmap` requirements on this training path. Review pip's proposed changes if
it reports conflicts; do not let it replace the engineer-provided runtime.
Clone dependencies once; for transferred checkouts, verify their pinned commits.
Re-activate the environment, source CANN, and export the variables above in each
new shell/job. CANN's `set_env.sh` provides its library/tool/ops paths; do not
guess `LD_LIBRARY_PATH` or `ASCEND_OPP_PATH`.

## 2. Prepare data and pretrained weights

The existing helper downloads RGB/parquet and weights through the HF mirror,
downloads our depth archive from ModelScope, and verifies/extracts that archive:
Set `DATA_ROOT`, `MODEL_ROOT`, and `OUTPUT_ROOT` to your chosen directories
before this block if needed. Unset variables retain the existing repository-local
defaults; the same selected paths are used for downloads and training.

For example, replace these placeholder paths with your storage directories:

```bash
export DATA_ROOT="/path/to/datasets"
export MODEL_ROOT="/path/to/pretrained-models"
export OUTPUT_ROOT="/path/to/training-output"
```

```bash
python -m pip install -c "$PLATFORM_CONSTRAINTS" -c constraints-ascend.txt modelscope==1.38.1
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data_remote}"
export MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/checkpoints}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/vla_gam_matrix_v2}"
python scripts/download_uavflow_assets.py \
  --data-root "$DATA_ROOT" --model-root "$MODEL_ROOT" \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth --workers 2
```

For an offline node, transfer the same files/directories instead. If the depth
archive was transferred but not extracted:

```bash
python scripts/extract_uavflow_depth.py \
  --dataset-root "$DATA_ROOT/UAV-Flow-Sim-Depth-Archive" \
  --output "$DATA_ROOT/UAV-Flow-Sim-Depth"
```

Required layout (shown with the defaults; subdirectory names are unchanged
under the engineer-selected data/model roots):

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
export UAVFLOW_SIM_ROOT="$DATA_ROOT/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$DATA_ROOT/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$MODEL_ROOT/track4world_da3.pth"
export QWEN_MODEL="$MODEL_ROOT/qwen3.5-2b"
export T5_MODEL="$MODEL_ROOT/t5-base"
```

## 3. Smoke tests

Run on the receiving NPU node after environment preparation:

No compatibility-mode switch is needed. Project package pins, NPU availability,
tensor/ABI checks and data audit remain active; they do not establish full
hardware/software compatibility. A standalone import audit is:

```bash
python scripts/verify_uavflow_remote.py --imports-only --accelerator npu
```

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
