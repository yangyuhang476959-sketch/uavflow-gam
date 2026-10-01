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

## New-node command order

First inventory the node before installing a wheel:

```bash
uname -a
uname -m
python3 --version
python3 -c 'import platform; print(platform.machine()); print(platform.platform())'
ldd --version | head
npu-smi info
find /usr/local/Ascend -maxdepth 3 -type f -name set_env.sh -print
```

Then clone this repository. The default `reference` mode is strict: it requires
the exact Python/CANN/core triplet above and never asks pip to choose a torch
stack. Supply the three architecture-matched wheels when they are not already
present in the isolated environment:

```bash
git clone https://github.com/yangyuhang476959-sketch/uavflow-gam.git
cd uavflow-gam
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest
export CANN_VERSION=9.0.0  # only after verifying this selected installation
export ASCEND_VENV=$PWD/.venv-ascend
export TORCH_WHEEL=/shared/wheels/torch-2.7.1+cpu-cp311-linux_aarch64.whl
export TORCH_NPU_WHEEL=/shared/wheels/torch_npu-2.7.1.post4-cp311-linux_aarch64.whl
export TORCHVISION_WHEEL=/shared/wheels/torchvision-0.22.1-cp311-manylinux_2_28_aarch64.whl
bash scripts/setup_ascend_cluster.sh
```

Only when cluster driver/firmware policy makes the exact stack impossible,
select the administrator runtime explicitly. This mode inherits and snapshots
the core triplet, generates exact temporary constraints for it, and then lets
pip resolve ordinary project dependencies without replacing that triplet:

```bash
export ASCEND_STACK_MODE=vendor
export CANN_ROOT=/administrator/selected/cann
bash scripts/setup_ascend_cluster.sh
```

If more than one CANN installation is discovered, setup/runtime both stop and
list every candidate rather than selecting the lexicographically first one.

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
