"""Train/evaluate B or scalar C using the existing cursor-aware GAM trainer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

from omegaconf import OmegaConf

from experiments.uavflow_remote_ablation.matrix_v2 import overrides


ROOT = Path(__file__).resolve().parents[2]


def build_config(*, phase="geometry", uncertainty="none", extra=()):
    if phase not in {"geometry", "correction"} or uncertainty not in {"none", "scalar"}:
        raise ValueError("Unknown phase/uncertainty")
    cfg = OmegaConf.load(ROOT / "experiments/uavflow_remote_ablation/base.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides("R1"))))
    depth_root = Path(os.environ.get("UAVFLOW_DEPTH_ROOT", ROOT / "data_remote/UAV-Flow-Sim-Depth"))
    model_root = Path(os.environ.get("MODEL_ROOT", ROOT / "checkpoints"))
    name = "B" if uncertainty == "none" else "C_scalar"
    values = {
        "stage1.da3_checkpoint": os.environ.get("DA3_CHECKPOINT", str(model_root / "track4world_da3.pth")),
        "stage1.qwen_model": os.environ.get("QWEN_MODEL", str(model_root / "qwen3.5-2b")),
        "stage1.action_stats_dir": os.environ.get("ACTION_STATS_DIR", str(ROOT / "data/uavflow_stats_sim_openvla_yaw4d")),
        "dataset.parquet_root": os.environ.get("UAVFLOW_SIM_ROOT", str(ROOT / "data_remote/UAV-Flow-Sim")),
        "dataset.gt_depth_root": str(depth_root / "hybrid"),
        "dataset.gt_depth_fallback_roots": [str(depth_root / "replay")],
        "dataset.instruction_overrides_path": str(depth_root / "metadata/instruction_overrides.json"),
        "model.parallel_current_geometry_read_enabled": False,
        "model.dense_context_supervision": False,
        "model.support_aware_transport.enabled": True,
        "model.support_aware_transport.uncertainty": uncertainty,
        "model.support_aware_transport.uncertainty_tau": 1.,
        "model.support_aware_transport.uncertainty_weight": 1.,
        "model.support_aware_transport.geometry_action_source": "gt" if phase == "geometry" else "proposal",
        "model.support_aware_transport.geometry_translation_gate_meters": .25,
        "model.support_aware_transport.geometry_yaw_gate_degrees": 5.,
        "model.support_aware_transport.warp_depth_scale_meters": 100.,
        "model.support_aware_transport.fov_degrees": 90.,
        "model.support_aware_transport.cleanup_kernel": 5,
        "model.support_aware_transport.min_hole_area": 128,
        "loss.depth_target_mode": "both",
        "loss.depth_scale_mode": "fixed_metric",
        "loss.depth_fixed_scale_meters": 100.,
        "loss.transport_geometry_weight": 1.,
        "training.max_epochs": 5,
        "training.results_dir": str(Path(os.environ.get("OUTPUT_ROOT", ROOT / "results/support_aware_transport")) / name / phase),
    }
    if os.environ.get("SPLIT_FILE"):
        values["dataset.split_file"] = os.environ["SPLIT_FILE"]
    for key, value in values.items():
        OmegaConf.update(cfg, key, value)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(extra)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("geometry", "correction"), default="geometry")
    parser.add_argument("--uncertainty", choices=("none", "scalar"), default="none")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--proposal-checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    extra = list(args.set)
    if args.max_trajectories is not None:
        extra.append(f"dataset.max_trajectories={args.max_trajectories}")
    cfg = build_config(phase=args.phase, uncertainty=args.uncertainty, extra=extra)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if args.phase == "correction" and not (args.resume or args.init_checkpoint):
        raise ValueError("Correction starts from a geometry checkpoint; use --init-checkpoint or --resume")
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    from experiments.uavflow_predictor_idm.train import main as train_main
    # Every DDP child creates its own config file. All semantic overrides are
    # embedded in saved checkpoint config, not tied to this temporary path.
    with tempfile.TemporaryDirectory(prefix="uavflow-support-aware-") as directory:
        path = Path(directory) / "config.yaml"
        OmegaConf.save(cfg, path)
        old_argv = sys.argv
        sys.argv = [old_argv[0], "--config", str(path)]
        for key in ("resume", "init_checkpoint", "proposal_checkpoint"):
            if getattr(args, key):
                sys.argv += ["--"+key.replace("_", "-"), getattr(args, key)]
        if args.eval_only:
            sys.argv += ["--eval-only"]
        try:
            train_main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
