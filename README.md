# UAVFlow-GAM

Active release: **three metric-depth experiments** — GAM baseline LoRA,
the original VLM/GAM correction with Current Bank CA, and RGB-warp correction.

Minimal research code for transferring Geometric Action Models to
UAV-Flow-Sim. This repository contains only the source, configurations and
launchers needed to reproduce the remote ablation suite:

- GitHub: training code, experiment definitions and setup/launch scripts.
- ModelScope: episode-aligned UAV-Flow depth sidecars.
- Compute-server output directory: checkpoints, logs and generated results.

The original UAV-Flow RGB/instruction trajectories are not duplicated here.
Download them from the official UAV-Flow Hugging Face repository and join the
depth sidecars by `episode_id` and `frame_indices`.

After downloading the official simulation parquet shards, keep them unchanged.
Seven known person/dog instruction mismatches are corrected at load time using
`data/instruction_overrides.json`; the dataset documentation lists this patch
and its application order.

## Repository map

- `experiments/uavflow_predictor_idm/`: model, objectives and training entry point.
- `src/robot/`: DA3/GAM model and UAV-Flow data implementation.
- `scripts/`: data download/audit and training helpers.
- `data/instruction_overrides.json`: seven reviewed instruction corrections.
- `data/uavflow_stats_sim_openvla_yaw4d/`: fixed action normalization statistics.
- `experiments/metric_jobs/`: three independent Python training entry points.
- `experiments/uavflow_remote_ablation/`: shared configuration/split machinery
  and historical benchmark compatibility, not the active ten-cell matrix.

## Remote experiment quick start

**NVIDIA UE simulation handoff: [simulation/README.md](simulation/README.md).**
This is the copied closed-loop simulator client and its separate environment,
not a change to training or Ascend setup.

**Ascend handoff: [NPU_SETUP.md](NPU_SETUP.md).**

The code was previously validated on Ascend 910B2 / aarch64, Driver 25.2.1,
CANN 9.0.0, Python 3.11.15, PyTorch 2.7.1 and torch_npu 2.7.1.post4.
The receiving engineer must prepare a hardware-compatible stack using Huawei's
official compatibility guidance. These versions are a validation record, not a
universal prescription for A2/A3 or other Driver/CANN versions.

Follow the manual Python/package, data preparation and smoke-test commands in
`NPU_SETUP.md`; the previous automatic Ascend bootstrap is no longer the handoff
workflow. This workflow does not install or modify Driver, Firmware or CANN.

Choose one of the [three metric experiments](docs/METRIC_EXPERIMENTS.md).
Defaults are **10 epochs, constant LR, no Stop phase**, per-process batch 4.
One distributed experiment per node is supported through the existing trainer;
use explicit `--resume` with unchanged data/batch/world-size for exact continuation.

For NVIDIA-only environment instructions, see
[`docs/REMOTE_TRAINING.md`](docs/REMOTE_TRAINING.md); do not apply that CUDA setup
to an Ascend node. Architecture details are in
[`docs/vla_gfm_action_tokens.md`](docs/vla_gfm_action_tokens.md).

The derived depth sidecars are published as
[`acetaffy123/UAV-Flow-Sim-Depth`](https://modelscope.cn/datasets/acetaffy123/UAV-Flow-Sim-Depth/files).

Before publishing, follow [`docs/PUBLISH_CHECKLIST.md`](docs/PUBLISH_CHECKLIST.md).

## Data source precedence

For a given episode, training uses the first available source:

1. human-reviewed dynamic-object hybrid depth;
2. corrected reverse replay depth;
3. primary calibrated PnP-to-UE replay depth.

The third source is not the obsolete raw-pose export. Every selected replay
episode has a saved camera-pose reference and per-episode iterative PnP
calibration before depth is rendered in UE.

## Reproducibility status

Machine-specific paths are supplied through the environment variables listed
in `NPU_SETUP.md` (Ascend) or `docs/REMOTE_TRAINING.md` (NVIDIA).
Large checkpoints, depth arrays and all generated
experiment outputs are deliberately excluded from Git.

## Maintained architectures

See [the three-experiment handoff](docs/METRIC_EXPERIMENTS.md) for architecture,
pose conventions, losses and exact commands. Detailed geometry implementation:
[RGB-warp correction](docs/dual_stream_rgbwarp.md).

Unused local launchers, triangle/visibility probes and ten obsolete matrix
job wrappers have been removed. Shared backend modules and regression fixtures
are retained where the three experiments or the working environment benchmark
still import them. Local debug products are ignored, not published. No datasets,
policy checkpoints or evaluation annotations are part of this cleanup.
