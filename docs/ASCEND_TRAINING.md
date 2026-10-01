# Ascend NPU training handoff

The canonical, version-pinned Ascend 910B2 setup, smoke test, complete R1
benchmark, measured performance, and rejected optimizations are documented in
[ascend_cluster.md](ascend_cluster.md).

For a new node, do not use a generic CUDA/conda environment and do not let pip
resolve the torch stack. The supported entry sequence is:

```bash
uname -m
npu-smi info
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest
bash scripts/setup_ascend_cluster.sh
source scripts/ascend_env.sh
```

After the R1 smoke/benchmark succeeds, each existing Python file under
`experiments/uavflow_remote_ablation/jobs_v2/` remains one independent
eight-NPU matrix job. Defaults are `NPROC=8`, `GLOBAL_BATCH_SIZE=32`, hence
batch size 4 per NPU. On NPU the runner selects the validated BF16 profile,
FLA Triton-Ascend, and `model.gradient_checkpointing=false`; it does not alter
the scientific R1 architecture or losses.
