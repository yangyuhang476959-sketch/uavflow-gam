#!/usr/bin/env python3
"""Legacy lightweight installer for a pre-provisioned Ascend image.

For a new service node use ``scripts/setup_ascend_cluster.sh``. This helper
remains for vendor images whose torch/torch_npu/torchvision triplet is already
correct; it uses the same constraints and cannot replace that core stack.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def verify_vendor_runtime() -> None:
    torch = importlib.import_module("torch")
    try:
        torch_npu = importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is missing. Use a Huawei Ascend PyTorch/CANN image; "
            "do not install the CUDA environment-uavflow.yml first."
        ) from exc
    if not torch_npu.npu.is_available():
        raise RuntimeError("torch_npu is installed but no NPU is available")
    print(
        f"vendor runtime: torch={torch.__version__} "
        f"npu_devices={torch_npu.npu.device_count()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--da3-commit", default="2c21ea849ceec7b469a3e62ea0c0e270afc3281a"
    )
    parser.add_argument(
        "--skip-python-deps", action="store_true",
        help="Only verify the existing environment and install DA3 editable.",
    )
    args = parser.parse_args()

    verify_vendor_runtime()
    py = sys.executable
    if not args.skip_python_deps:
        # Never pass an index/requirement containing torch here: the vendor
        # image owns the torch/torchvision/torch_npu compatibility triplet.
        run(
            py, "-m", "pip", "install",
            "-c", str(ROOT / "constraints-ascend.txt"),
            "--upgrade-strategy", "only-if-needed",
            "-r", str(ROOT / "requirements-ascend.txt"),
        )

    da3 = ROOT / "Depth-Anything-3"
    if not (da3 / ".git").is_dir():
        run("git", "clone", "https://github.com/ByteDance-Seed/Depth-Anything-3.git", str(da3))
    run("git", "-C", str(da3), "fetch", "--all", "--tags")
    run("git", "-C", str(da3), "checkout", args.da3_commit)
    run(py, "-m", "pip", "install", "--no-deps", "-e", str(da3))
    run(py, "-m", "pip", "check")

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", ""),
        "UAVFLOW_ACCELERATOR": "npu",
        "UAVFLOW_DISABLE_FLEX_ATTENTION": "1",
    })
    run(
        py,
        str(ROOT / "scripts/verify_uavflow_remote.py"),
        "--imports-only",
        "--accelerator",
        "npu",
        env=env,
    )


if __name__ == "__main__":
    main()
