# Code Map

## Stable training path

```text
experiments/uavflow_predictor_idm/train.py
  -> src/robot/data/uavflow_dataset.py
  -> experiments/uavflow_predictor_idm/model.py
  -> experiments/uavflow_predictor_idm/objectives.py
  -> src/robot/modeling/uav_multimodal_predictor.py
  -> src/robot/modeling/da3_giant_encoder.py
```

The run configuration is loaded with OmegaConf and command-line overrides use
`--set key=value`.  A resolved copy is saved in every output directory.

## Remote execution path

```text
scripts/uavflow_handoff.sh
  -> scripts/setup_uavflow_remote.sh
  -> scripts/download_uavflow_assets.sh
  -> scripts/verify_uavflow_remote.py
  -> experiments/uavflow_remote_ablation/run_server.sh
```

Generated checkpoints, logs and evaluation media are intentionally not part of
this source release. Machine-specific paths are supplied through `server.env`.
