# VLA ablations on UAVFlow-Open and IndoorUAV

Run commands from the repository root. Prepare the NPU software stack using `NPU_SETUP.md`. The current code and these ablations have CPU regression coverage. A full training or resume on the target NPU stack must be checked on that machine.

Place the DA3 checkpoint at `checkpoints/track4world_da3.pth` and Qwen3.5-2B at `checkpoints/qwen3.5-2b`. UAVFlow data and depth roots are specified in the two ablation YAML files. Adjust paths with `--set` or edit a copy of the YAML. The DA3 checkpoint SHA-256 shared by the observed local and Indoor runs is `6c589c0f41632a2329920087dcd8cbd6465d4198d574f0d4186eeada78c492c7`. The Qwen3.5-2B `model.safetensors-00001-of-00001.safetensors` SHA-256 shared by both runs is `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`. The recorded initial parameter hashes are in `docs/experiment_baselines/warp_20261029/common_init_hashes.json`. Startup checks the shared Qwen, Pose, proposal projection, proposal head and final action head tensors.

VLA+Future uses current RGB and Qwen's five action slots in one DA3 pass and supervises t+5 depth and the final action. VLA+GAM uses the frozen DA3 shallow features and Qwen slots in a GAM predictor, then decodes predicted features and action slots with the future DA3. The configurations retain their recorded slot encoding and Pose choices. Both preserve full epoch, last and best checkpoints including optimizer state.

For UAVFlow, the split is `docs/experiment_baselines/warp_20261029/episode_split.json`. Both ablations use the original Open instruction map `docs/experiment_baselines/warp_20261029/raw_instructions.json` (SHA-256 `0ec7276977ca233ea4236ea7ddf31859778a8f9c2f1d76006e357df621223c0d`) and the seven reviewed target-noun repairs in `data/instruction_overrides.json`. The recorded split contains 10,008 training and 101 validation episodes. Depth scale is 100 metres. Check the data, model and stats paths, then inspect configurations:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export UAVFLOW_ACCELERATOR=npu
python -m experiments.uavflow_paper_ablation.future_train --print-config
python -m experiments.uavflow_paper_ablation.gam_train --variant vla_gam --print-config
```

For IndoorUAV, supply the official data root with `train.csv`, `test_seen.csv`, `test_unseen.csv`, `vla_ins` and `without_screenshot`. Prepare both configs using different output directories. The preparation groups a long trajectory and its reverse, excludes the official test sets, checks the stored split IDs and source hashes, computes action statistics, and records the manifest hash.

```bash
python -m experiments.indooruav_warp.prepare --data-root /path/to/indoor-data --output prepared/indoor_future --variant vla_future
python -m experiments.indooruav_warp.prepare --data-root /path/to/indoor-data --output prepared/indoor_gam --variant vla_gam
```

The expected manifest SHA-256 is `11e059df476bfff92da9921f47acec9409c725b41aa1cc706b3587c8abd1d77e`. Indoor actions are five cumulative current-anchor targets, and Habitat depth is converted to metres and trained with depth scale 1. The recorded split has 24,288 train and 1,279 validation tasks, with 474,970 and 24,937 windows respectively. Indoor uses its own original instructions and does not apply the UAVFlow instruction map. Prepared config paths refer to the location used during preparation, so regenerate if data or checkout moves.

## Training commands

The default is **15 epochs for UAVFlow-Open** and **5 epochs for IndoorUAV**. Set `NPROC` to 1, 2, 3, 4, 6 or 8 and `EPOCHS` to the total target number of epochs. `NPROC` is the number of NPUs used by this job. Set `ASCEND_RT_VISIBLE_DEVICES` to the desired devices before launching. The launcher maintains global batch 24.

| NPUs | Microbatch per NPU | Gradient accumulation |
| --- | ---: | ---: |
| 1 | 8 | 3 |
| 2 | 12 | 1 |
| 3 | 8 | 1 |
| 4 | 6 | 1 |
| 6 | 4 | 1 |
| 8 | 3 | 1 |

One and three NPUs retain the reference 8-sample microgroups. Other counts repartition each reference global batch of 24 while retaining its sample identities and ordering. Depth-loss reduction groups and floating-point results can differ. The evaluation batch size is selected automatically to give all ranks equal numbers of forward calls without repeating validation windows. Verify memory capacity on the target NPU.

```bash
# Choose the devices available on this machine.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# UAVFlow-Open
NPROC=8 EPOCHS=15 bash scripts/run_ablation_npu.sh experiments/uavflow_paper_ablation/configs/vla_future.yaml
NPROC=8 EPOCHS=15 bash scripts/run_ablation_npu.sh experiments/uavflow_paper_ablation/configs/vla_gam.yaml

# IndoorUAV, after preparing the two configs above
NPROC=8 EPOCHS=5 bash scripts/run_ablation_npu.sh prepared/indoor_future/vla_future.yaml
NPROC=8 EPOCHS=5 bash scripts/run_ablation_npu.sh prepared/indoor_gam/vla_gam.yaml
```

Run each training command as a separate job with its own available devices. To resume, pass the matching full checkpoint as the second argument and keep the same `NPROC` and data configuration. `EPOCHS` is the total target, not the number of extra epochs.

```bash
NPROC=8 EPOCHS=5 bash scripts/run_ablation_npu.sh prepared/indoor_future/vla_future.yaml /path/to/last.pt
```

Preparation and configuration inspection do not start training.

## Alignment and intended ablations

For each dataset, the ablations use that dataset's recorded split, image processing, action representation, normalizer, Qwen/DA3 weights, shared Pose MLP and final action-head initialization. Qwen LoRA learning rate is 1e-5, DA3 LoRA learning rate is 5e-5, and Pose MLP and action heads use 5e-4. AdamW uses constant learning rate and zero warmup. Each epoch, last and best checkpoint includes the full optimizer and training state.

The intentional model differences are VLA+Future's single current-image DA3 pass supervised with future depth, and VLA+GAM's future-feature Predictor with feature MSE and future-depth supervision. VLA+GAM keeps the specified Qwen slot and Pose design. Five-slot Predictor embeddings and an extra Future DA3 Pose token are not enabled in its default configuration. Hardware-dependent grouping on 2, 4, 6 and 8 NPUs is documented above. These are new experiments and the recorded runs retain their historical configurations.
