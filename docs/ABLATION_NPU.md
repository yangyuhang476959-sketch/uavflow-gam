# VLA ablations on UAVFlow-Open and IndoorUAV

Run commands from the repository root. Prepare the NPU software stack using `NPU_SETUP.md`. The current code and these ablations have CPU regression coverage. A full training or resume on the target NPU stack must be checked on that machine.

Place the DA3 checkpoint at `checkpoints/track4world_da3.pth` and Qwen3.5-2B at `checkpoints/qwen3.5-2b`. UAVFlow data and depth roots are specified in the two ablation YAML files. Adjust paths with `--set` or edit a copy of the YAML. The DA3 checkpoint SHA-256 shared by the observed local and Indoor runs is `6c589c0f41632a2329920087dcd8cbd6465d4198d574f0d4186eeada78c492c7`. The recorded initial parameter hashes are in `docs/experiment_baselines/warp_20261029/common_init_hashes.json`. Startup checks the shared Qwen, Pose, proposal projection, proposal head and final action head tensors.

VLA+Future uses current RGB and Qwen's five action slots in one DA3 pass and supervises t+5 depth and the final action. VLA+GAM uses the frozen DA3 shallow features and Qwen slots in a GAM predictor, then decodes predicted features and action slots with the future DA3. The configurations retain their recorded slot encoding and Pose choices. Both preserve full epoch, last and best checkpoints including optimizer state.

For UAVFlow, the split is `docs/experiment_baselines/warp_20261029/episode_split.json`. Depth scale is 100 metres. Check the data, model and stats paths, then inspect configurations:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export UAVFLOW_ACCELERATOR=npu
python -m experiments.uavflow_paper_ablation.future_train --print-config
python -m experiments.uavflow_paper_ablation.gam_train --variant vla_gam --print-config
```

For IndoorUAV, supply the official data root with `train.csv`, `test_seen.csv`, `test_unseen.csv`, `vla_ins`, `without_screenshot` and `audit/split_tasks.json`. Prepare both configs using different output directories. The preparation groups a long trajectory and its reverse, excludes the official test sets, checks the stored split IDs and source hashes, computes action statistics, and records the manifest hash.

```bash
python -m experiments.indooruav_warp.prepare --data-root /path/to/indoor-data --output prepared/indoor_future --variant vla_future
python -m experiments.indooruav_warp.prepare --data-root /path/to/indoor-data --output prepared/indoor_gam --variant vla_gam
```

The expected manifest SHA-256 is `11e059df476bfff92da9921f47acec9409c725b41aa1cc706b3587c8abd1d77e`. Indoor actions are five cumulative current-anchor targets, and Habitat depth is converted to metres and trained with depth scale 1. The recorded split has 24,288 train and 1,279 validation tasks. Prepared config paths refer to the location used during preparation, so regenerate if data or checkout moves.

Train from a generated configuration with the intended NPU device allocation. The reference is three devices, batch 8 per device and global batch 24. For example:

```bash
torchrun --standalone --nproc_per_node=3 -m experiments.uavflow_predictor_idm.train --config prepared/indoor_future/vla_future.yaml
torchrun --standalone --nproc_per_node=3 -m experiments.uavflow_predictor_idm.train --config prepared/indoor_gam/vla_gam.yaml
```

Training is not started by preparation. Resume only from a matching full checkpoint using the original configuration and `--resume`.
