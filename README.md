# UAVFlow-GAM

Active matrix: **10 cells** covering GAM/T5, frozen VLM conditioning,
Query-VLA, Slot-VLA, Current Geometry Bank, geometry-action residual, and a
dual-view diagnostic.

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
- `experiments/uavflow_remote_ablation/`: restartable ten-cell remote matrix;
  each cluster job is one Python command on one eight-device node.

## Remote experiment quick start

**Ascend handoff: [NPU_SETUP.md](NPU_SETUP.md).**

The code was previously validated on Ascend 910B2 / aarch64, Driver 25.2.1,
CANN 9.0.0, Python 3.11.15, PyTorch 2.7.1 and torch_npu 2.7.1.post4.
The receiving engineer must prepare a hardware-compatible stack using Huawei's
official compatibility guidance. These versions are a validation record, not a
universal prescription for A2/A3 or other Driver/CANN versions.

Follow the manual Python/package, data preparation and smoke-test commands in
`NPU_SETUP.md`; the previous automatic Ascend bootstrap is no longer the handoff
workflow. This workflow does not install or modify Driver, Firmware or CANN.

Choose one experiment from the [ten-cell matrix](experiments/uavflow_remote_ablation/README.md),
one job per eight-device node. Defaults are global batch 32, Stage 1 **10 epochs
constant LR**, then Stage 2 **10 epochs cosine LR with Stop**. Re-run the same
job command to resume.

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
