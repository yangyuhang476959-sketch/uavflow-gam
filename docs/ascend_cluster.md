# Huawei Ascend 910B2 deployment and R1 benchmark

This path preserves the complete R1 method: Qwen3.5-2B LoRA, Future
Predictor, five action slots, trainable Current Geometry Bank with per-layer
reads, DA3 block 13+ full-rank training, Current/Future depth supervision, and
geometry-residual action correction. The depth heads may remain frozen, but
their gradients continue through the decoded depth into DA3 features.

## Validated core stack

- Linux aarch64, Python 3.11.15
- Ascend 910B2 64GB, CANN 9.0.0
- torch 2.7.1+cpu, torch_npu 2.7.1.post4, torchvision 0.22.1
- NumPy 1.26.4, SciPy 1.15.3
- transformers 5.5.4, huggingface-hub 1.10.1
- triton-ascend 3.2.1 (its imported `triton.__version__` reports 3.2.0)
- Flash Linear Attention commit `9f38d24980c46d46bd38614e743cdacd21906578`

Do not replace `torch 2.7.1+cpu` with a CUDA wheel. Ascend device support is
registered by the matching `torch_npu` package.

## One-command new-node setup

On a normal online node whose administrator has already installed compatible
Ascend Driver/Firmware, the reference workflow is:

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
bash scripts/setup_ascend_cluster.sh
```

The script inventories but never modifies Driver/Firmware. It supports
`aarch64` and `x86_64`, accepts any Python 3.11 patch release, and creates an
isolated conda prefix at `.ascend/env`. If `conda` is absent it downloads the
architecture-matched `Miniconda3-latest-Linux-{aarch64,x86_64}.sh` directly
from the official `https://repo.anaconda.com/miniconda/` source and installs it
under `.ascend/miniconda3`. It never modifies system Python. For an offline
node, set `MINICONDA_INSTALLER` to the corresponding official installer.
Before CANN installation, reference bootstrap verifies that host `gcc` and
`g++` are available (it never invokes `sudo`, `apt`, or `yum`) and installs the
CANN Python prerequisites into the isolated prefix: attrs, Cython, NumPy below
2, decorator, SymPy, CFFI, PyYAML, pathlib2, psutil, protobuf 3.20.x, SciPy,
requests, and absl-py. The later project dependency phase applies the pinned
UAVFlow-GAM versions.

The one-command entry point is only an orchestrator. It first bootstraps the
Python 3.11 conda prefix, so even a node without system `python3` can run the
metadata helpers. The phases can also be prepared independently:

```bash
# On a Python-less node, prepare only Miniconda/Python first.
bash scripts/setup_ascend_python.sh --bootstrap-only

# Toolkit/runtime phase; reuses that Python but installs no Python packages.
bash scripts/setup_ascend_cann.sh

# Python/package phase; consumes .ascend/runtime.env from the first phase.
bash scripts/setup_ascend_python.sh

# Full smoke test and environment report.
bash scripts/setup_ascend_cluster.sh
```

It discovers CANN under `/usr/local/Ascend`, `${HOME}/Ascend`, and optional
`ASCEND_SEARCH_ROOT`. Multiple installations are never guessed: set
`CANN_ROOT` to the exact selected installation. CANN version detection reads
Huawei's official `ascend_toolkit_install.info` metadata (`package_name`,
`version`, and `arch`); it does not trust directory names or recursively grep
arbitrary files.

Reference mode looks for an existing, metadata-confirmed CANN 9.0.0. If the
host has only CANN 8.x (or no CANN), that administrator installation is kept
untouched and a separate reference stack is installed under
`.ascend/cann-9.0.0`. If exactly one valid 9.0.0 installation already exists,
it is reused. Symlink aliases are canonicalized before deciding whether there
are multiple installations.

Huawei's official CANN documentation requires both the Toolkit package and
the hardware-specific ops package. For Atlas A2/Ascend 910B, the exact 9.0.0
package names used here are:

```text
Ascend-cann-toolkit_9.0.0_linux-{aarch64,x86_64}.run
Ascend-cann-910b-ops_9.0.0_linux-{aarch64,x86_64}.run
```

Huawei's CANN 9.0 installation guide publishes direct URLs for both packages.
Reference mode downloads the architecture-matched files automatically from
`ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/` when neither
an explicit installer nor a cached copy is available. Offline/manual overrides
remain available:

```bash
export CANN_TOOLKIT_INSTALLER=/packages/Ascend-cann-toolkit_9.0.0_linux-aarch64.run
export CANN_OPS_INSTALLER=/packages/Ascend-cann-910b-ops_9.0.0_linux-aarch64.run
# Optional when authoritative checksums are available:
export CANN_TOOLKIT_SHA256=<official-sha256>
export CANN_OPS_SHA256=<official-sha256>
bash scripts/setup_ascend_cann.sh
```

The script first searches `.ascend/packages` and
`${HOME}/.cache/uavflow-ascend`. An operator may override the authoritative
HTTPS endpoints through `CANN_TOOLKIT_URL` and `CANN_OPS_URL`. Toolkit and ops
are installed into the same selected prefix.
Each downloaded or cached runfile is executed with `--check` before either
installer is allowed to modify the project-local CANN prefix.

An existing reference installation is accepted only when both official files
report version `9.0.0`:

```text
<arch>-linux/ascend_toolkit_install.info
<arch>-linux/ascend_ops_install.info
```

The presence of an `opp/` directory alone is not treated as proof that the
matching 910B ops package is installed.

After selection, `.ascend/runtime.env` records the exact `CANN_ROOT`,
`CANN_ENV_FILE`, Python environment, Triton target, FLA checkout, and DA3
checkout. `source scripts/ascend_env.sh` loads this persisted selection first,
so a second administrator CANN cannot silently change later runs. A stale
persisted path fails clearly instead of falling back to another installation.

Reference torch packages are installed online from the official PyTorch CPU
wheel index and PyPI. The effective commands inside the selected conda prefix
are:

```bash
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt torch-npu==2.7.1.post4 --index-url https://pypi.org/simple
python -m pip install -c constraints-ascend.txt -c constraints-ascend-reference.txt torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Offline nodes may provide all three local wheels:

```bash
export TORCH_WHEEL=/shared/wheels/torch-2.7.1+cpu-cp311-linux_aarch64.whl
export TORCH_NPU_WHEEL=/shared/wheels/torch_npu-2.7.1.post4-cp311-linux_aarch64.whl
export TORCHVISION_WHEEL=/shared/wheels/torchvision-0.22.1-cp311-linux_aarch64.whl
bash scripts/setup_ascend_cluster.sh
```

Wheel Python and architecture tags are checked before installation. Override
paths and caches with `ASCEND_ENV_ROOT`, `ASCEND_SEARCH_ROOT`, `CANN_USER_ROOT`,
`CANN_PACKAGE_CACHE`, or explicit `CANN_ROOT`.

Only when cluster driver/firmware policy makes the exact stack impossible,
the user must select `ASCEND_STACK_MODE=vendor` explicitly; setup never switches
modes automatically. This mode inherits and snapshots
the core triplet, generates exact temporary constraints for it, and then lets
pip resolve ordinary project dependencies without replacing that triplet:

```bash
export ASCEND_STACK_MODE=vendor
export CANN_ROOT=/administrator/selected/cann
export ASCEND_ENV_ROOT=/administrator/python311/environment
bash scripts/setup_ascend_cluster.sh
```

Vendor mode never creates an empty conda prefix. `ASCEND_ENV_ROOT` is required
and must point to an existing Python 3.11 environment where `torch`,
`torch_npu`, and `torchvision` import successfully after the selected
`CANN_ENV_FILE` has been sourced. The initial `--bootstrap-only` phase checks
only the interpreter path and Python version, deliberately delaying the NPU
imports until CANN runtime libraries are active.

If more than one genuinely distinct matching installation remains, setup stops
and asks for explicit `CANN_ROOT`; it never picks lexicographically.

After smoke tests, the complete comparison record is written to
`.ascend/environment-report.json`, including host/NPU inventory, CANN metadata,
Python and package versions, and pinned FLA/DA3 commits.

`pycolmap` is intentionally isolated in
`requirements-optional-geometry.txt`; it is not needed for the training path
and must not block aarch64 installation.

## Smoke test

```bash
source scripts/ascend_env.sh
python - <<'PY'
import numpy, scipy, torch, torch_npu, torchvision
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('torchvision', torchvision.__version__)
print('numpy', numpy.__version__, 'scipy', scipy.__version__)
print('npu available', torch.npu.is_available())
print('device count', torch.npu.device_count())
x = torch.randn(256, 256, dtype=torch.bfloat16, device='npu')
y = x @ x
torch.npu.synchronize()
print('BF16 matmul', tuple(y.shape), bool(y.isfinite().all()))
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs()
from depth_anything_3.api import DepthAnything3
import triton, fla
print('triton', triton.__version__)
PY
```

## Complete R1 50-step benchmark

Set the dataset/checkpoint/model paths, then:

```bash
export UAVFLOW_SIM_ROOT=/data/UAV-Flow-Sim
export UAVFLOW_DEPTH_ROOT=/data/UAV-Flow-Sim-Depth
export DA3_CHECKPOINT=/models/track4world_da3.pth
export QWEN_MODEL=/models/Qwen3.5-2B
export T5_MODEL=/models/t5-base
export OUTPUT_ROOT=$PWD/results/bench_r1_ascend
bash scripts/bench_r1_ascend.sh
```

Expected key output on the validated 910B2 node:

```text
torch 2.7.1+cpu
torch_npu 2.7.1.post4
[QWEN-FLA-NPU] patched 18 linear-attention layers with FLA Triton-Ascend causal-conv + GDR
AMP grad scaler: policy=auto enabled=False dtype=bf16
[BENCH] steps=11-50 count=40 ... mean=1.37...s/step throughput=...
```

The benchmark disables gradient checkpointing, evaluation, and all checkpoint
writes. It writes `_BENCHMARK_COMPLETE`, never a fake `_SUCCESS`. Normal matrix
training still requires real checkpoints, and Stage 2 cannot run without one.

## Performance baseline and rejected optimizations

The full method measured 1.377254 s/step, about 2.904 samples/s, and about
34.99GB peak memory at batch size 4. Depth-disabled and Current-Bank-disabled
numbers are diagnostic floors only and are not production configurations.

The production path deliberately does not enable:

- NpuFusedAdamW: incompatible with BF16 trainable parameters in the tested stack.
- fused RMSNorm: negligible end-to-end gain.
- concatenated four-projection GDN: slower and adds about 578MiB.
- FLA gate/beta fusion: slower forward and backward.
- forced 192KiB UB: slower than the default 64KiB fallback.
- DA3 clone/cat alias rewrite: slower end-to-end.
- Qwen 0.8B substitution: no meaningful end-to-end speed gain.

Do not set `ASCEND_UB_CAPACITY_BITS`; do not replace Triton-Ascend with upstream
NVIDIA Triton merely because FLA prints a recommended-version warning.
