#!/usr/bin/env python3
"""Materialize one immutable UAV-Flow-Sim episode split for all ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from experiments.uavflow_predictor_idm.runtime import apply_overrides
from robot.data.dataset import build_robot_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(OmegaConf.load(args.config), args.set)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg.pop("split_file", None)
    train_set = build_robot_dataset(dict(dataset_cfg), is_eval=False)
    eval_set = build_robot_dataset(dict(dataset_cfg), is_eval=True)
    if not hasattr(train_set, "traj") or not hasattr(eval_set, "traj"):
        raise TypeError("Remote ablation split preparation expects one UAVFlowParquetDataset.")
    train_ids = sorted(str(value) for value in train_set.traj)
    eval_ids = sorted(str(value) for value in eval_set.traj)
    overlap = set(train_ids).intersection(eval_ids)
    if overlap:
        raise RuntimeError(f"Split contains {len(overlap)} leaked episode IDs.")
    payload = {
        "format": "uavflow_stratified_instruction_v1",
        "episode_split_mode": str(cfg.dataset.episode_split_mode),
        "episode_split_seed": int(cfg.dataset.episode_split_seed),
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
        "category_counts": getattr(train_set, "split_category_counts", None),
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"wrote {output}: train={len(train_ids)} val={len(eval_ids)} overlap=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
