#!/usr/bin/env python3
"""Print a compact status table for remote ablation runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    print(f"{'ID':<6} {'STAGE':<20} {'STATUS':<10} CHECKPOINT")
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        for stage_name in ("stage1", "stage2_stop_clip_lr", "stage2_stop"):
            stage = run / stage_name
            if not stage.exists():
                continue
            success = stage / "_SUCCESS"
            checkpoints = sorted(stage.glob("ckpt_*.pt"))
            if success.exists():
                status = "complete"
                checkpoint = success.read_text().strip()
            elif checkpoints:
                status = "partial"
                checkpoint = str(checkpoints[-1])
            else:
                status = "created"
                checkpoint = "-"
            print(f"{run.name:<6} {stage_name:<20} {status:<10} {checkpoint}")


if __name__ == "__main__":
    main()
