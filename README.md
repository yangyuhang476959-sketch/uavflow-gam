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
- `scripts/`: environment setup, data download/audit and one-command handoff.
- `data/instruction_overrides.json`: seven reviewed instruction corrections.
- `data/uavflow_stats_sim_openvla_yaw4d/`: fixed action normalization statistics.
- `experiments/uavflow_remote_ablation/`: restartable ten-cell remote matrix;
  each cluster job is one Python command on one 8-GPU node.

## Remote experiment quick start

The publication-ready setup, download, audit, exact matrix and ten independent
multi-GPU commands are in [`docs/REMOTE_TRAINING.md`](docs/REMOTE_TRAINING.md).
Architecture details are in
[`docs/vla_gfm_action_tokens.md`](docs/vla_gfm_action_tokens.md).
The remote-training document is also the concise handoff and copy-paste guide.

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
in `docs/REMOTE_TRAINING.md`. Large checkpoints, depth arrays and all generated
experiment outputs are deliberately excluded from Git.
