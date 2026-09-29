#!/usr/bin/env python3
"""Create the pinned UAV-Flow training runtime on a clean node."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-index-url",
        default=os.environ.get(
            "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124"
        ),
    )
    parser.add_argument(
        "--da3-commit",
        default="2c21ea849ceec7b469a3e62ea0c0e270afc3281a",
    )
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    run(py, "-m", "pip", "install", "pip==26.1.2", "setuptools==80.10.2", "wheel==0.47.0")
    if not args.skip_torch:
        run(
            py, "-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1",
            "--index-url", args.torch_index_url,
        )
    run(py, "-m", "pip", "install", "-r", str(ROOT / "requirements-uavflow.txt"))
    da3 = ROOT / "Depth-Anything-3"
    if not (da3 / ".git").is_dir():
        run("git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(da3))
    run("git", "-C", str(da3), "fetch", "--all", "--tags")
    run("git", "-C", str(da3), "checkout", args.da3_commit)
    run(py, "-m", "pip", "install", "--no-deps", "-e", str(da3))
    run(py, "-m", "pip", "check")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", "")
    print("+ runtime smoke test", flush=True)
    subprocess.run(
        [py, str(ROOT / "scripts/verify_uavflow_remote.py"), "--imports-only"],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    main()
