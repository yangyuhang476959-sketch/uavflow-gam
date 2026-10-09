"""Offline checks for benchmark reporting; no claim of NPU validation."""
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.uavflow_predictor_idm import runtime


def test_benchmark_reports_peak_allocated_memory(monkeypatch):
    module = SimpleNamespace(
        synchronize=lambda: None,
        max_memory_allocated=lambda device: 49.52 * 1024**3,
    )
    accelerator = runtime.Accelerator('npu', torch.device('cpu'), 'hccl', module)
    clock = iter([100.0, 174.76])
    monkeypatch.setattr(runtime.time, 'perf_counter', lambda: next(clock))
    benchmark = runtime.StepBenchmark(accelerator, 11, 50, samples_per_step=8)
    benchmark.before_step(11)
    assert benchmark.after_step(49) is None
    line = benchmark.after_step(50)
    assert 'mean=1.869000s/step' in line
    assert 'throughput=4.280 samples/s' in line
    assert 'peak_mem=49.52GB' in line


def test_benchmark_uses_active_environment_and_selectable_batch():
    root = Path(__file__).resolve().parents[1]
    script = (root / 'scripts/bench_r1_ascend.sh').read_text()
    assert 'source "${ROOT}/scripts/ascend_env.sh"' not in script
    assert 'export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"' in script
    assert 'export NPROC=1' in script
    assert 'export UAVFLOW_ACCELERATOR=npu' in script
    assert '${TRITON_ASCEND_TARGET}:${FLA_ASCEND_DIR}' in script
