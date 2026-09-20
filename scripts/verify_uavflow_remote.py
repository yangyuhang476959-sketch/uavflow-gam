#!/usr/bin/env python3
"""Fail-fast audit for a remote UAV-Flow ablation installation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--da3-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    args = parser.parse_args()

    parquets = sorted(args.sim_root.glob("train-*-of-00021.parquet"))
    if len(parquets) != 21:
        raise RuntimeError(f"Expected 21 UAV-Flow-Sim parquet shards, got {len(parquets)}")

    hybrid = args.depth_root / "hybrid"
    replay = args.depth_root / "replay"
    metadata = args.depth_root / "metadata"
    for path, description in (
        (hybrid, "hybrid depth directory"),
        (replay, "replay depth directory"),
        (metadata / "episodes.csv", "depth episode manifest"),
        (metadata / "instruction_overrides.json", "instruction overrides"),
        (args.da3_checkpoint, "Track4World DA3 checkpoint"),
        (args.qwen_model / "config.json", "Qwen config"),
        (args.t5_model / "config.json", "T5 config"),
    ):
        require(path, description)

    with (metadata / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))
    if len(episode_rows) != 10109:
        raise RuntimeError(f"Expected 10,109 depth episodes, got {len(episode_rows)}")
    if len({row["episode_id"] for row in episode_rows}) != len(episode_rows):
        raise RuntimeError("Depth manifest contains duplicate episode IDs")

    overrides = json.loads((metadata / "instruction_overrides.json").read_text())
    if len(overrides) != 7:
        raise RuntimeError(f"Expected 7 instruction corrections, got {len(overrides)}")

    print("remote audit: OK")
    print(f"  sim parquet shards: {len(parquets)}")
    print(f"  depth episodes:     {len(episode_rows)}")
    print(f"  instruction fixes:  {len(overrides)}")


if __name__ == "__main__":
    main()
