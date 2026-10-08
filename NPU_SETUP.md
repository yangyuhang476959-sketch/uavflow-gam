# Minimal Ascend NPU deployment

Manual handoff for the **ten-cell remote matrix** in this repository. Run commands
from the repository root in the same Bash session. This guide does not install or
change Driver, Firmware or CANN, and does not use the automatic bootstrap scripts.

## 1. Check the machine and select a compatible stack

Previously validated stack (reported deployment, not a new validation here):

| Component | Reference |
| --- | --- |
| Hardware / CPU architecture | Ascend 910B2 / aarch64 |
| Driver | 25.2.1 |
| CANN | 9.0.0, with hardware-matched ops |
| Python | 3.11.15 |
| PyTorch / torch_npu | 2.7.1 (`+cpu` wheel) / 2.7.1.post4 |
| torchvision | 0.22.1 |
| Triton-Ascend | 3.2.1 (imported Triton reports 3.2.0) |

**Before installation, the receiving engineer must verify hardware, OS/CPU,
Driver/Firmware, CANN/ops, Python, PyTorch and torch_npu compatibility.** An A2/A3
or different driver/CANN is not automatically compatible with this reference.
Consult the [official TorchNPU compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
and the hardware-specific Huawei installation requirements. Firmware compatibility
must be confirmed by the administrator; its validated version is not recorded here.

```bash
uname -m
npu-smi info
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export PROJECT_ROOT="$PWD"
```

Ask the administrator for the **exact existing CANN installation's** `set_env.sh`:

```bash
# Replace this placeholder with the administrator-confirmed file; not another CANN.
export CANN_ENV_FILE=/path/to/selected/CANN/set_env.sh
source "$CANN_ENV_FILE"
```

This loads CANN's required library/tool/ops variables (including its configured
`LD_LIBRARY_PATH`, `PATH` and `ASCEND_OPP_PATH`). Do not guess these paths, point
them at a second installation, or install a CUDA PyTorch wheel on this environment.

**Current launcher limitation:** `scripts/verify_uavflow_remote.py` explicitly
requires PyTorch **2.7.1** and torch_npu **2.7.1.post4** on NPU; the matrix also
defaults to BF16. A different officially compatible stack still needs a separate
project compatibility review before these jobs can run. Do not bypass the audit
or assume this document validates that stack.

## 2. Python and project dependencies

If the administrator already supplies a compatible Python environment, activate
it instead. Otherwise, with an existing architecture-matched Conda installation:

```bash
conda create -n uavflow-npu python=3.11.15 pip -y
conda activate uavflow-npu
source "$CANN_ENV_FILE"
```

Only after confirming the reference stack fits the node, install the reference
accelerator packages below. Skip these three commands if they are already supplied
by the administrator. Other hardware-dependent versions must be selected manually.

```bash
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt \
  torch-npu==2.7.1.post4 --index-url https://pypi.org/simple
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt \
  torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Project-level dependencies are pinned in `requirements-ascend.txt`, including
NumPy 1.26.4, SciPy 1.15.3, transformers 5.5.4 and huggingface-hub 1.10.1.
Keep the accelerator constraints during installation so pip cannot replace the
reference runtime. Do not use the CUDA environment file or `requirements-uavflow.txt`
on this node; the latter includes optional `pycolmap`, not required for training.

```bash
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt \
  -r requirements-ascend.txt

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
git -C Depth-Anything-3 checkout --detach 2c21ea849ceec7b469a3e62ea0c0e270afc3281a
python -m pip install --no-deps -e Depth-Anything-3
```

The validated Qwen NPU path additionally uses **hardware-dependent** Triton-Ascend
and pinned FLA. Their compatibility must also be checked on a different platform.
The isolated target below matches the existing installer without replacing a
system Triton package:

```bash
export TRITON_ASCEND_TARGET="$PROJECT_ROOT/.ascend/triton"
export FLA_ASCEND_DIR="$PROJECT_ROOT/.ascend/flash-linear-attention"
mkdir -p "$TRITON_ASCEND_TARGET"
python -m pip install --target "$TRITON_ASCEND_TARGET" --no-deps \
  triton-ascend==3.2.1 pybind11
git clone https://github.com/fla-org/flash-linear-attention.git "$FLA_ASCEND_DIR"
git -C "$FLA_ASCEND_DIR" checkout --detach 9f38d24980c46d46bd38614e743cdacd21906578

export PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UAVFLOW_ACCELERATOR=npu
export UAVFLOW_QWEN_FLA_NPU=1
export UAVFLOW_DISABLE_FLEX_ATTENTION=1
python -m pip check
```

On a fresh checkout, clone each dependency once. If already transferred, verify
its commit rather than cloning over it. Re-export these variables and source the
selected CANN environment in every new shell/job; no `runtime.env` is required.

## 3. Basic verification (run on the receiving NPU node)

```bash
python - <<'PY'
import platform, numpy as np, scipy.special, torch, torch_npu, torchvision
print(platform.machine(), platform.python_version())
print(torch.__version__, torch_npu.__version__, torchvision.__version__)
print('NPU devices:', torch.npu.device_count())
assert torch.npu.is_available()
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
import triton, fla
print('Basic import/BF16 forward/backward PASS; Triton:', triton.__version__)
PY
python scripts/verify_uavflow_remote.py --imports-only --accelerator npu
```

Imports/matmul alone do not validate the full model. Run the small R1 job below
before production. This guide was prepared on an NVIDIA host; no NPU validation
was performed during this documentation task.

## 4. Data and model weights

Download with the existing repository helper (HF mirror + ModelScope). ModelScope
is a download dependency, pinned in the existing NVIDIA requirements file:

```bash
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt \
  modelscope==1.38.1
python scripts/download_uavflow_assets.py \
  --hf-endpoint https://hf-mirror.com \
  --depth-repo acetaffy123/UAV-Flow-Sim-Depth --workers 2
```

This downloads the official `wangxiangyu0814/UAV-Flow-Sim`, `Qwen/Qwen3.5-2B`,
`google-t5/t5-base`, and `checkpoints/track4world_da3.pth` from the dataset
`SeonghuJeon/3da-libero-training-assets`. It downloads depth to
`data_remote/UAV-Flow-Sim-Depth-Archive` and invokes the checksum-verified extractor.
Network access/mirror availability and download permissions require manual
verification. For an offline node, transfer those same directories/files instead.

If the archive was transferred without extraction, run only:

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
checkpoints/qwen3.5-2b/                               # complete model + processor
checkpoints/t5-base/                                 # complete model + tokenizer
```

The loader prioritizes hybrid depth, then replay: do not flatten everything into
`<episode>/depth.npy`. Keep parquet unchanged; seven reviewed instruction fixes
are applied through the metadata overrides. RGB/trajectory preprocessing is done
by the loader; shared train/validation split creation is handled by the job runner.
All three pretrained model assets are required by the current launch audit, even
for T5-only cells. Trained policy checkpoints are outputs, not download prerequisites.

```bash
export UAVFLOW_SIM_ROOT="$PROJECT_ROOT/data_remote/UAV-Flow-Sim"
export UAVFLOW_DEPTH_ROOT="$PROJECT_ROOT/data_remote/UAV-Flow-Sim-Depth"
export DA3_CHECKPOINT="$PROJECT_ROOT/checkpoints/track4world_da3.pth"
export QWEN_MODEL="$PROJECT_ROOT/checkpoints/qwen3.5-2b"
export T5_MODEL="$PROJECT_ROOT/checkpoints/t5-base"
export OUTPUT_ROOT="$PROJECT_ROOT/results/vla_gam_matrix_v2"

python scripts/verify_uavflow_remote.py --accelerator npu \
  --sim-root "$UAVFLOW_SIM_ROOT" --depth-root "$UAVFLOW_DEPTH_ROOT" \
  --da3-checkpoint "$DA3_CHECKPOINT" --qwen-model "$QWEN_MODEL" --t5-model "$T5_MODEL"
```

The audit checks exact package versions, all episode counts, instruction overrides,
and randomly loads three replay plus three hybrid episodes by default.

## 5. Training, resume and validation

First do a separate **one-NPU, 20-trajectory, one-epoch R1 smoke** (real model
forward/backward, not a production result):

```bash
OUTPUT_ROOT="$PROJECT_ROOT/results/npu_smoke" NPROC=1 GLOBAL_BATCH_SIZE=4 \
DEVICE_IDS=0 STAGE1_EPOCHS=1 \
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py \
  --stage stage1 --max-trajectories 20
```

Then submit **one experiment per eight-NPU node**; choose one line, not all ten on
the same node. The reference defaults are batch 4/NPU, total batch 32, Stage 1
10 epochs constant LR, Stage 2 10 epochs cosine LR with Stop. The base LR is
`5e-5`; Stage 2 policy base LR is divided by 10, Stop LR is `5e-4` (the model's
existing parameter-group multipliers still apply).

```bash
export NPROC=8 GLOBAL_BATCH_SIZE=32 DEVICE_IDS=0,1,2,3,4,5,6,7
export STAGE1_EPOCHS=10 STAGE2_EPOCHS=10
python experiments/uavflow_remote_ablation/jobs_v2/01_g0.py  # T5, no numeric pose
python experiments/uavflow_remote_ablation/jobs_v2/02_g1.py  # T5 + numeric pose
python experiments/uavflow_remote_ablation/jobs_v2/03_c0.py  # frozen Qwen condition
python experiments/uavflow_remote_ablation/jobs_v2/04_c1.py  # condition + numeric pose
python experiments/uavflow_remote_ablation/jobs_v2/05_q0.py  # Query-VLA
python experiments/uavflow_remote_ablation/jobs_v2/06_s0.py  # Slot-VLA, no Current
python experiments/uavflow_remote_ablation/jobs_v2/07_s1.py  # add Current depth
python experiments/uavflow_remote_ablation/jobs_v2/08_s2.py  # add Current bank reads
python experiments/uavflow_remote_ablation/jobs_v2/09_r1.py  # geometry action residual
python experiments/uavflow_remote_ablation/jobs_v2/10_dv.py  # dual-view diagnostic
```

`--stage stage1`, `--stage stage2`, or default `both` select the stages. Resume
by re-running the **same command with the same paths, split and batch configuration**:
the runner uses `last.pt` (fallback `ckpt_*.pt`), restores same-stage optimizer,
scheduler/RNG/data cursor, and skips completed `_SUCCESS` stages. Stage 2 starts
from Stage 1's lowest validation Action-loss `best_action.pt`. Do not delete markers
or reuse the smoke output/split for production. Outputs:
`$OUTPUT_ROOT/<ID>/{stage1,stage2_stop}`; logs/configs include `console.log`,
`train.log`, `resolved_config.yaml`. Validation normally runs every 2,000 steps,
limited to 100 batches; epoch checkpoint archives are enabled.

Standalone validation of an existing R1 Stage 1 checkpoint, using its saved
architecture/data configuration and a separate output directory:

```bash
RUN_DIR="$OUTPUT_ROOT/R1/stage1"
ASCEND_RT_VISIBLE_DEVICES=0 python experiments/uavflow_predictor_idm/train.py \
  --config "$RUN_DIR/resolved_config.yaml" --resume "$RUN_DIR/best_action.pt" \
  --eval-only --eval-domain mixed \
  --set "training.results_dir=$OUTPUT_ROOT/R1/validation" \
  --set training.eval_max_batches=0
```

Here `eval_max_batches=0` means the full validation loader; use `100` for the usual
bounded validation. For another ID/stage, change `RUN_DIR` and the checkpoint.
Results are written as `eval_ckpt_<step>.json`. For resume/validation on a different
filesystem, transferred configs/checkpoints contain saved paths and may require
manual compatibility/path review; do not assume cross-machine exact resume.

**Closed-loop UE evaluation is not validated on Ascend in this release.** The
remote release's confirmed evaluation command is the offline validation above.
Use the separate existing simulation evaluation environment on its supported host;
do not invent an NPU UE/policy-server command.
