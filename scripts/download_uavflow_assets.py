#!/usr/bin/env python3
"""Download official UAV-Flow/model assets plus the published depth archive."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_remote")
    parser.add_argument("--model-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument(
        "--depth-repo", default="acetaffy123/UAV-Flow-Sim-Depth"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    data = args.data_root.resolve(); models = args.model_root.resolve()
    data.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    # huggingface_hub resolves HF_ENDPOINT while importing its constants, so
    # set the mirror before importing the client. This matters on mainland
    # nodes that have no direct huggingface.co route.
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(
        "wangxiangyu0814/UAV-Flow-Sim", repo_type="dataset",
        local_dir=data / "UAV-Flow-Sim", max_workers=args.workers,
    )
    snapshot_download(
        "Qwen/Qwen3.5-2B", local_dir=models / "qwen3.5-2b",
        max_workers=args.workers,
    )
    snapshot_download(
        "google-t5/t5-base", local_dir=models / "t5-base",
        max_workers=args.workers,
    )
    da3_checkpoint = models / "track4world_da3.pth"
    if not da3_checkpoint.is_file():
        downloaded = hf_hub_download(
            "SeonghuJeon/3da-libero-training-assets",
            "checkpoints/track4world_da3.pth", repo_type="dataset",
            local_dir=models,
        )
        # HF retains the remote checkpoints/ prefix. Keep the public local
        # layout flat beside Qwen/T5, including when --model-root is custom.
        Path(downloaded).replace(da3_checkpoint)
    archive = data / "UAV-Flow-Sim-Depth-Archive"
    subprocess.run([
        "modelscope", "download", args.depth_repo, "--repo-type", "dataset",
        "--local-dir", str(archive),
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/extract_uavflow_depth.py"),
        "--dataset-root", str(archive),
        "--output", str(data / "UAV-Flow-Sim-Depth"),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
