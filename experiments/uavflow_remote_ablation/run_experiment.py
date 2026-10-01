#!/usr/bin/env python3
"""Run one VLA--GAM matrix cell on one 8-GPU node.

The scheduler-facing command is always Python. The process performs the data
audit, creates/reuses one immutable split, resumes interrupted epoch
checkpoints, and runs Stage 1 then joint Stop Stage 2.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from experiments.uavflow_remote_ablation.matrix_v2 import EXPERIMENTS, overrides
from experiments.uavflow_predictor_idm.runtime import env_truthy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/uavflow_remote_ablation/base.yaml"


def python_interpreter(value: str | None = None) -> Path:
    """Return an interpreter path without dereferencing a venv symlink."""
    raw = value if value is not None else os.environ.get("PYTHON_BIN", sys.executable)
    if os.sep not in raw:
        raw = shutil.which(raw) or raw
    return Path(raw).expanduser()


def required_path(env: str, default: Path | None = None) -> Path:
    value = os.environ.get(env)
    path = Path(value).expanduser().resolve() if value else default
    if path is None or not path.exists():
        raise FileNotFoundError(f"Set {env}; missing path: {path}")
    return path


def latest_checkpoint(directory: Path) -> Path | None:
    last = directory / "last.pt"
    if last.is_file() and last.stat().st_size:
        return last
    candidates = sorted(directory.glob("ckpt_*.pt"))
    return candidates[-1] if candidates else None


def tee_run(command: list[str], *, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def set_args(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        result += ["--set", value]
    return result


def benchmark_missing_checkpoint_allowed(
    *, stage: str, requested_stage: str, skip_final_checkpoint: bool
) -> bool:
    """Only an explicit Stage-1-only benchmark may finish without weights."""
    return bool(
        skip_final_checkpoint and stage == "stage1" and requested_stage == "stage1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--stage", choices=("both", "stage1", "stage2"), default="both")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Additional OmegaConf override for diagnostics/benchmarks.",
    )
    args = parser.parse_args()
    skip_final_checkpoint = env_truthy("UAVFLOW_SKIP_FINAL_CHECKPOINT")
    if skip_final_checkpoint and args.stage != "stage1":
        raise ValueError(
            "UAVFLOW_SKIP_FINAL_CHECKPOINT=1 is benchmark-only and requires "
            "--stage stage1; Stage 2 must have a real Stage-1 checkpoint."
        )

    python = python_interpreter()
    sim = required_path("UAVFLOW_SIM_ROOT", ROOT / "data_remote/UAV-Flow-Sim")
    depth = required_path(
        "UAVFLOW_DEPTH_ROOT", ROOT / "data_remote/UAV-Flow-Sim-Depth"
    )
    da3 = required_path("DA3_CHECKPOINT", ROOT / "checkpoints/track4world_da3.pth")
    qwen = required_path("QWEN_MODEL", ROOT / "checkpoints/qwen3.5-2b")
    t5 = required_path("T5_MODEL", ROOT / "checkpoints/t5-base")
    output_root = Path(
        os.environ.get("OUTPUT_ROOT", ROOT / "results/vla_gam_matrix_v2")
    ).resolve()
    split = Path(
        os.environ.get("SPLIT_FILE", output_root / "shared_split_seed42.json")
    ).resolve()
    nproc = int(os.environ.get("NPROC", "8"))
    global_batch = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
    if global_batch % nproc:
        raise ValueError("GLOBAL_BATCH_SIZE must be divisible by NPROC")
    per_gpu_batch = global_batch // nproc
    requested_accelerator = os.environ.get("UAVFLOW_ACCELERATOR", "auto").lower()
    if requested_accelerator == "ascend":
        requested_accelerator = "npu"
    if requested_accelerator == "auto":
        ascend_environment = any(
            os.environ.get(name)
            for name in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
            )
        )
        # Do not import/probe torch here: CUDA/NPU visibility must be fixed
        # before either runtime is initialized in the torchrun children.
        # Ascend vendor images expose one of the CANN variables above; all
        # other remote jobs retain the historical CUDA default.
        requested_accelerator = "npu" if ascend_environment else "cuda"
    if requested_accelerator not in {"cuda", "npu"}:
        raise ValueError(
            "Remote matrix jobs require UAVFLOW_ACCELERATOR=cuda or npu; "
            f"got {requested_accelerator!r}."
        )
    device_ids = os.environ.get(
        "DEVICE_IDS",
        os.environ.get(
            "CUDA_DEVICES", ",".join(str(index) for index in range(nproc))
        ),
    )
    stage1_epochs = int(os.environ.get("STAGE1_EPOCHS", "20"))
    stage2_epochs = int(os.environ.get("STAGE2_EPOCHS", "20"))
    base_lr = float(os.environ.get("BASE_LR", "5e-5"))

    env = os.environ.copy()
    env.update({
        "UAVFLOW_ACCELERATOR": requested_accelerator,
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}:" + env.get("PYTHONPATH", ""),
    })
    if requested_accelerator == "npu":
        env.update({
            "ASCEND_RT_VISIBLE_DEVICES": device_ids,
            "HCCL_ASYNC_ERROR_HANDLING": env.get("HCCL_ASYNC_ERROR_HANDLING", "1"),
            # TorchInductor/Triton FlexAttention is CUDA-specific. The matrix
            # does not require it, so keep the dense portable path on NPU.
            "UAVFLOW_DISABLE_FLEX_ATTENTION": "1",
            "UAVFLOW_QWEN_FLA_NPU": env.get("UAVFLOW_QWEN_FLA_NPU", "1"),
        })
    else:
        env.update({
            "CUDA_VISIBLE_DEVICES": device_ids,
            "PYTORCH_CUDA_ALLOC_CONF": env.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        })
    subprocess.run([
        str(python), str(ROOT / "scripts/verify_uavflow_remote.py"),
        "--sim-root", str(sim), "--depth-root", str(depth),
        "--da3-checkpoint", str(da3), "--qwen-model", str(qwen),
        "--t5-model", str(t5), "--accelerator", requested_accelerator,
    ], cwd=ROOT, env=env, check=True)

    common = [
        f"stage1.da3_checkpoint={da3}",
        f"stage1.idm_checkpoint={ROOT / 'results/robot/unused-idm.pt'}",
        f"stage1.action_stats_dir={ROOT / 'data/uavflow_stats_sim_openvla_yaw4d'}",
        f"stage1.qwen_model={qwen}", f"stage1.t5_model={t5}",
        f"dataset.parquet_root={sim}",
        f"dataset.gt_depth_root={depth / 'hybrid'}",
        f"dataset.gt_depth_fallback_roots=['{depth / 'replay'}']",
        f"dataset.instruction_overrides_path={depth / 'metadata/instruction_overrides.json'}",
        f"training.batch_size={per_gpu_batch}",
        f"training.num_workers={int(os.environ.get('NUM_WORKERS', '4'))}",
        f"training.lr={base_lr}", "training.save_latest_every=0",
        "training.amp_dtype=" + os.environ.get(
            "AMP_DTYPE", "auto"
        ),
        "stage1.qwen_attention_implementation=" + os.environ.get(
            "QWEN_ATTN_IMPLEMENTATION",
            "eager" if requested_accelerator == "npu" else "sdpa",
        ),
        f"dataset.split_file={split}",
    ]
    if requested_accelerator == "npu":
        # Validated 910B2 64GB production profile: BF16 without activation
        # checkpointing fits at batch=4 and is materially faster. R1 method
        # semantics (both depths, trainable Current Bank/read, correction)
        # remain unchanged.
        common += [
            "model.gradient_checkpointing=false",
            "training.amp_dtype=bf16",
        ]
    if args.max_trajectories is not None:
        common.append(f"dataset.max_trajectories={args.max_trajectories}")

    split.parent.mkdir(parents=True, exist_ok=True)
    with (split.with_suffix(split.suffix + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not split.is_file() or not split.stat().st_size:
            split_common = [item for item in common if not item.startswith("dataset.split_file=")]
            subprocess.run([
                str(python),
                str(ROOT / "experiments/uavflow_remote_ablation/prepare_split.py"),
                "--config", str(CONFIG), "--output", str(split),
                *set_args(split_common),
            ], cwd=ROOT, env=env, check=True)

    experiment_root = output_root / args.experiment
    variant = list(overrides(args.experiment)) + list(args.set)
    benchmark_end_step = int(os.environ.get("UAVFLOW_BENCH_END_STEP", "0") or 0)

    def run_stage(stage: str, init: Path | None = None) -> Path | None:
        stage_dir = experiment_root / stage
        success = stage_dir / "_SUCCESS"
        if success.is_file() and success.read_text().strip():
            return Path(success.read_text().strip())
        stage_dir.mkdir(parents=True, exist_ok=True)
        resume = latest_checkpoint(stage_dir)
        stage_values = [f"training.results_dir={stage_dir}"]
        checkpoint: list[str] = []
        if resume is not None:
            checkpoint = ["--resume", str(resume)]
        elif init is not None:
            checkpoint = ["--init-checkpoint", str(init)]
        if stage == "stage1":
            stage_values += [
                "model.stop_head_enabled=false", "loss.stop_weight=0.0",
                "training.lr_schedule=constant", "training.warmup_steps=0",
            ]
            if benchmark_end_step > 0:
                stage_values += [
                    "training.max_epochs=0",
                    f"training.max_steps={benchmark_end_step}",
                    f"training.scheduler_total_steps={benchmark_end_step}",
                    "training.eval_every=0", "training.save_every=0",
                    "training.save_latest_every=0", "training.save_every_epochs=0",
                ]
            else:
                stage_values.append(f"training.max_epochs={stage1_epochs}")
        else:
            stage_values += [
                # Stage 2 alone receives five additional fully terminal
                # FN->FN windows per episode.  Keep Stage 1's action/depth
                # distribution unchanged while raising the raw Stop-positive
                # share for Stop-head training.
                "dataset.endpoint_self_pair_end_count=5",
                "model.stop_head_enabled=true",
                "model.stop_head_mode=action_hidden_pose",
                "loss.stop_weight=1.0", "loss.stop_pos_weight=5.0",
                f"training.lr={base_lr * 0.1}", "training.stop_head_lr=5e-4",
                "training.lr_schedule=cosine", "training.warmup_steps=500",
                "training.min_lr_ratio=0.05",
                f"training.max_epochs={stage2_epochs}",
            ]
        command = [
            str(python), "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={nproc}",
            str(ROOT / "experiments/uavflow_predictor_idm/train.py"),
            "--config", str(CONFIG), *checkpoint,
            *set_args(common + variant + stage_values),
        ]
        state = stage_dir / "run_state.txt"
        state.write_text(
            f"experiment={args.experiment}\nstage={stage}\nstart={datetime.now().isoformat()}\n"
            f"nproc={nproc}\nper_gpu_batch={per_gpu_batch}\nglobal_batch={global_batch}\n"
            f"accelerator={requested_accelerator}\ndevice_ids={device_ids}\n"
            f"split_sha256={hashlib.sha256(split.read_bytes()).hexdigest()}\n"
        )
        tee_run(command, env=env, log=stage_dir / "console.log")
        final = (
            stage_dir / "best_action.pt"
            if stage == "stage1" and (stage_dir / "best_action.pt").is_file()
            else latest_checkpoint(stage_dir)
        )
        if final is None:
            if benchmark_missing_checkpoint_allowed(
                stage=stage,
                requested_stage=args.stage,
                skip_final_checkpoint=skip_final_checkpoint,
            ):
                benchmark_marker = stage_dir / "_BENCHMARK_COMPLETE"
                benchmark_marker.write_text("status=benchmark_complete\n")
                with state.open("a") as stream:
                    stream.write(
                        f"status=benchmark_complete\nend={datetime.now().isoformat()}\n"
                    )
                return None
            raise RuntimeError(f"No checkpoint produced in {stage_dir}")
        success.write_text(str(final) + "\n")
        with state.open("a") as stream:
            stream.write(f"status=complete\ncheckpoint={final}\nend={datetime.now().isoformat()}\n")
        return final

    stage1_checkpoint: Path | None = None
    if args.stage in {"both", "stage1"}:
        stage1_checkpoint = run_stage("stage1")
    if args.stage in {"both", "stage2"}:
        if stage1_checkpoint is None:
            marker = experiment_root / "stage1/_SUCCESS"
            if not marker.is_file():
                raise FileNotFoundError("Stage 2 requires completed Stage 1")
            stage1_checkpoint = Path(marker.read_text().strip())
        run_stage("stage2_stop", init=stage1_checkpoint)


if __name__ == "__main__":
    main()
