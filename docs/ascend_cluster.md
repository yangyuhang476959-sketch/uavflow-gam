# Historical Ascend validation record

**Not an installation guide.** The automatic bootstrap handoff is superseded.
For current manual environment/data/smoke instructions use
[NPU_SETUP.md](../NPU_SETUP.md).

The following is the prior 910B2 validation/performance record, not a guarantee
for different hardware or a new validation performed on the current NVIDIA host.

## Validated environment

- Ascend 910B2 64GB, Linux aarch64; Driver 25.2.1, CANN 9.0.0.
- Python 3.11.15; torch 2.7.1+cpu, torch_npu 2.7.1.post4, torchvision 0.22.1.
- NumPy 1.26.4, SciPy 1.15.3, transformers 5.5.4, huggingface-hub 1.10.1.
- triton-ascend 3.2.1; imported `triton.__version__` reports 3.2.0.
- FLA commit `9f38d24980c46d46bd38614e743cdacd21906578`.

## Measured R1 performance

The full method measured 1.377254 s/step, about 2.904 samples/s and about
34.99GB peak memory at batch size 4. Measurement used steps 11–50 with gradient
checkpointing, evaluation and checkpoint writes disabled. These historical
measurements are not expected performance on an arbitrary new node.

### 2026-10-09 single-card batch-size test

Reported by the receiving engineer on the 910B2 deployment, not measured on
the documentation-editing NVIDIA host. Environment/dependencies, FLA, dataset
audit and actual R1 training passed. This does not validate eight-card training.

| Per-card batch | s/step | samples/s | Peak memory |
| --- | --- | --- | --- |
| 2 | 1.609 | 1.243 | 31.01 GB |
| 4 | 1.683 | 2.376 | 36.99 GB |
| 8 | 1.869 | 4.280 | 49.52 GB |
| 12 | OOM | — | — |
| 16 | OOM | — | — |

These single-card results do not change the formal experiment default:
8 cards, global batch 32 (per-card batch 4). For a repeat measurement, first
activate the manually configured environment and source its CANN environment
as in `NPU_SETUP.md`, then run:

```bash
GLOBAL_BATCH_SIZE=8 bash scripts/bench_r1_ascend.sh
```

The helper defaults to batch 4 and uses the active Python, not `.ascend/env`.
`[BENCH]` reports synchronized timing over steps 11–50 and `peak_mem`, the
process peak allocated accelerator memory (bytes / 1024³), not reserved memory
or total device use. Raw logs/configuration remain necessary for comparing nodes.

The production path did not enable NpuFusedAdamW (incompatible with the tested
BF16 trainable parameters), fused RMSNorm (negligible gain), concatenated GDN
projections, FLA gate/beta fusion, forced 192KiB UB, DA3 alias rewrites or a
Qwen 0.8B substitution; they were slower or offered no material end-to-end gain.
Do not substitute upstream NVIDIA Triton for Triton-Ascend to silence a warning.

This page intentionally contains no CANN download/installation commands or
automatic environment bootstrap instructions.
