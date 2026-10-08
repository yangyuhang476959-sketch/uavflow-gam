#!/usr/bin/env python3
"""Fail-fast runtime and data audit for a remote UAV-Flow installation."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import random
from collections import OrderedDict
from pathlib import Path


PINNED_RUNTIME = {
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "transformers": "5.5.4",
    "huggingface-hub": "1.10.1",
    "moviepy": "1.0.3",
    "addict": "2.4.0",
    "plyfile": "1.1.3",
    "trimesh": "4.8.3",
    "evo": "1.33.0",
}


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def verify_runtime(accelerator: str = "auto") -> None:
    versions = {}
    for distribution, expected in PINNED_RUNTIME.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"Version drift for {distribution}: expected {expected}, got {actual}"
            )
        versions[distribution] = actual

    np = importlib.import_module("numpy")
    scipy = importlib.import_module("scipy")
    torch = importlib.import_module("torch")
    requested = str(accelerator).strip().lower()
    if requested == "ascend":
        requested = "npu"
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "npu"
    if requested == "npu":
        try:
            torch_npu = importlib.import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError(
                "Ascend audit requires the vendor-matched torch_npu package."
            ) from exc
        if not torch_npu.npu.is_available():
            raise RuntimeError("torch_npu imported but no Ascend NPU is available")
        # Hardware/Driver/CANN/torch compatibility is confirmed by the engineer.
        # Test usability here instead of enforcing one historical version pair.
        device = torch.device("npu:0")
        probe = torch.arange(4, dtype=torch.float32, device=device)
        if float(probe.sum().cpu()) != 6.0:
            raise RuntimeError("Ascend tensor smoke test failed")
        try:
            torch_npu_version = importlib.metadata.version("torch-npu")
        except importlib.metadata.PackageNotFoundError:
            torch_npu_version = getattr(torch_npu, "__version__", "unknown")
        print(
            "  accelerator=npu "
            f"torch_npu={torch_npu_version} "
            f"devices={torch_npu.npu.device_count()}"
        )
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA audit requested but torch.cuda.is_available() is false")
        print(f"  accelerator=cuda devices={torch.cuda.device_count()}")
    else:
        raise ValueError(f"Unsupported accelerator {accelerator!r}")
    # Exercise the exact ABI boundaries that failed after NumPy drifted to 2.x.
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    tensor = torch.from_numpy(array)
    if tuple(tensor.shape) != (3, 4) or tensor.dtype != torch.float32:
        raise RuntimeError("torch.from_numpy interoperability smoke test failed")
    scipy.special.expit(np.asarray([-1.0, 0.0, 1.0], dtype=np.float64))

    # pycolmap is offline-reconstruction-only and has no generally available
    # CPython-3.11 aarch64 wheel. It must not gate the main training runtime.
    for module in ("moviepy.editor", "addict", "plyfile", "trimesh", "evo"):
        importlib.import_module(module)
    from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs

    _install_da3_optional_stubs()
    from depth_anything_3.api import DepthAnything3  # noqa: F401

    print("runtime smoke: OK")
    print("  " + " ".join(f"{name}={version}" for name, version in versions.items()))
    print(f"  torch={torch.__version__} scipy={scipy.__version__}")


def _make_depth_loader(depth_root: Path):
    """Create the smallest real loader instance needed to test `_depth()`."""
    from robot.data.uavflow_dataset import (
        UAVFlowParquetDataset,
        _expand_uavflow_depth_roots,
    )

    loader = UAVFlowParquetDataset.__new__(UAVFlowParquetDataset)
    loader.image_size = (224, 224)
    loader._depth_cache = OrderedDict()
    loader.gt_depth_root = depth_root
    loader.gt_depth_roots = _expand_uavflow_depth_roots([depth_root])
    loader._hybrid_depth_paths = {}
    for root in loader.gt_depth_roots:
        for path in sorted(root.glob("*/*.npz")):
            loader._hybrid_depth_paths.setdefault(path.stem, path)
    loader.gt_depth_required = True
    loader.gt_depth_min_meters = 0.0
    loader.gt_depth_max_meters = 650.0
    return loader


def _check_loaded_depth(loader, episode_id: str, frame_idx: int, expected_source: str) -> None:
    import torch

    source, path = loader._depth_source(episode_id, frame_idx)
    if source != expected_source:
        raise RuntimeError(
            f"Depth routing failed for {episode_id}: expected {expected_source}, "
            f"got {source} ({path})"
        )
    depth, valid, semantic = loader._depth(episode_id, frame_idx)
    if tuple(depth.shape) != (224, 224) or tuple(valid.shape) != (224, 224):
        raise RuntimeError(
            f"Bad depth shape for {episode_id}: depth={tuple(depth.shape)} "
            f"valid={tuple(valid.shape)}"
        )
    if not depth.dtype.is_floating_point or valid.dtype != torch.bool:
        raise RuntimeError(
            f"Bad depth dtype for {episode_id}: depth={depth.dtype} valid={valid.dtype}"
        )
    if semantic is not None and tuple(semantic.shape) != (224, 224):
        raise RuntimeError(f"Bad semantic-mask shape for {episode_id}: {tuple(semantic.shape)}")


def verify_depth_data(depth_root: Path, sample_count: int) -> tuple[int, int]:
    import numpy as np

    hybrid_files = sorted((depth_root / "hybrid").glob("*/*.npz"))
    replay_files = sorted((depth_root / "replay").glob("*/depth.npy"))
    if len(replay_files) != 6990:
        raise RuntimeError(f"Expected 6,990 replay episodes, got {len(replay_files)}")
    if len(hybrid_files) != 3119:
        raise RuntimeError(f"Expected 3,119 hybrid episodes, got {len(hybrid_files)}")

    replay_ids = {path.parent.name for path in replay_files}
    hybrid_ids = {path.stem for path in hybrid_files}
    overlap = replay_ids.intersection(hybrid_ids)
    if overlap:
        raise RuntimeError(f"Replay/hybrid episode overlap: {len(overlap)} IDs")
    if len(replay_ids | hybrid_ids) != 10109:
        raise RuntimeError("Depth trees do not contain 10,109 unique episodes")

    loader = _make_depth_loader(depth_root)
    rng = random.Random(42)
    for path in rng.sample(replay_files, min(sample_count, len(replay_files))):
        array = np.load(path, mmap_mode="r")
        if array.ndim != 3 or array.shape[0] < 1:
            raise RuntimeError(f"Bad replay array shape: {path} -> {array.shape}")
        frame_idx = rng.randrange(int(array.shape[0]))
        _check_loaded_depth(loader, path.parent.name, frame_idx, "replay")

    for path in rng.sample(hybrid_files, min(sample_count, len(hybrid_files))):
        with np.load(path) as payload:
            required = {"hybrid_depth_m", "valid_mask", "frame_indices"}
            missing = required.difference(payload.files)
            if missing:
                raise RuntimeError(f"Hybrid depth {path} is missing {sorted(missing)}")
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
        if frame_indices.size < 1:
            raise RuntimeError(f"Hybrid depth has no frames: {path}")
        frame_idx = int(frame_indices[rng.randrange(int(frame_indices.size))])
        _check_loaded_depth(loader, path.stem, frame_idx, "hybrid")

    print("depth loader smoke: OK")
    print(f"  replay={len(replay_files)} hybrid={len(hybrid_files)} total=10109")
    print(
        f"  sampled {min(sample_count, len(replay_files))} replay + "
        f"{min(sample_count, len(hybrid_files))} hybrid episodes"
    )
    return len(replay_files), len(hybrid_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument(
        "--accelerator", choices=("auto", "cuda", "npu", "ascend"), default="auto"
    )
    parser.add_argument("--sim-root", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--da3-checkpoint", type=Path)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--t5-model", type=Path)
    parser.add_argument("--depth-samples", type=int, default=3)
    args = parser.parse_args()

    verify_runtime(args.accelerator)
    if args.imports_only:
        return

    required_args = {
        "sim_root": args.sim_root,
        "depth_root": args.depth_root,
        "da3_checkpoint": args.da3_checkpoint,
        "qwen_model": args.qwen_model,
        "t5_model": args.t5_model,
    }
    missing_args = [name for name, value in required_args.items() if value is None]
    if missing_args:
        parser.error(
            "full audit requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing_args)
        )

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
    manifest_ids = {row["episode_id"] for row in episode_rows}
    if len(manifest_ids) != len(episode_rows):
        raise RuntimeError("Depth manifest contains duplicate episode IDs")

    replay_count, hybrid_count = verify_depth_data(args.depth_root, args.depth_samples)
    tree_ids = {path.parent.name for path in replay.glob("*/depth.npy")} | {
        path.stem for path in hybrid.glob("*/*.npz")
    }
    if tree_ids != manifest_ids:
        raise RuntimeError(
            f"Depth manifest/tree mismatch: manifest_only={len(manifest_ids - tree_ids)} "
            f"tree_only={len(tree_ids - manifest_ids)}"
        )

    overrides = json.loads((metadata / "instruction_overrides.json").read_text())
    if len(overrides) != 7:
        raise RuntimeError(f"Expected 7 instruction corrections, got {len(overrides)}")

    print("remote audit: OK")
    print(f"  sim parquet shards: {len(parquets)}")
    print(f"  depth episodes:     {replay_count + hybrid_count}")
    print(f"  instruction fixes:  {len(overrides)}")


if __name__ == "__main__":
    main()
