#!/usr/bin/env python3
"""Train UAV-Flow GAM feature predictor through a frozen four-frame IDM."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from experiments.uavflow_predictor_idm.runtime import (
    apply_overrides,
    amp_dtype,
    bootstrap_accelerator_plugin,
    configure_accelerator,
    create_npu_profiler,
    create_grad_scaler,
    distributed_info,
    env_truthy,
    grad_scaler_enabled,
    lr_scale,
    manual_seed_all,
    max_memory_allocated_gb,
    profile_phase,
    StepBenchmark,
    step_generator,
)

# TorchNPU must register the NPU backend before Transformers and DA3 modules
# are imported. This is a no-op for the default CUDA runtime.
bootstrap_accelerator_plugin()

from experiments.uavflow_predictor_idm.data import (
    EpisodePoseNormalizer,
    choose_context_length,
    move_batch,
    temporal_color_augment,
)
from experiments.uavflow_predictor_idm.idm import build_frozen_idm
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.objectives import evaluate, forward_batch
from experiments.uavflow_predictor_idm.logging_metrics import compact_train_line, enabled_metric
from experiments.uavflow_predictor_idm.vlm_conditioning import build_stage2_conditioner
from robot.data.dataset import ActionNormalizer, build_robot_dataset
from robot.modeling.da3_giant_encoder import DA3GiantEncoder


class CursorDistributedSampler(DistributedSampler):
    """Distributed sampler that seeks directly to a saved batch cursor.

    Reconstructing a DistributedSampler from ``seed`` and ``epoch`` gives the
    exact original index order.  Slicing that iterator by the number of samples
    already consumed avoids making DataLoader decode every skipped image during
    resume.  The offset applies only to the checkpoint's first resumed epoch.
    """

    def __init__(
        self,
        *args,
        resume_epoch: int = 0,
        resume_batch: int = 0,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resume_epoch = int(resume_epoch)
        self.resume_batch = int(resume_batch)
        self.batch_size = int(batch_size)

    def _cursor_samples(self) -> int:
        if int(self.epoch) != self.resume_epoch:
            return 0
        return min(self.resume_batch * self.batch_size, int(self.num_samples))

    def __iter__(self):
        return itertools.islice(super().__iter__(), self._cursor_samples(), None)

    def __len__(self) -> int:
        return max(0, int(self.num_samples) - self._cursor_samples())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/uavflow_predictor_idm/config.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-checkpoint", default="", help="Load Predictor/action weights only and start a new optimizer/run.")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--eval-domain",
        choices=("all", "mixed", "real", "sim"),
        default="all",
        help="Validation source used with --eval-only.",
    )
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("Use only one of --resume and --init-checkpoint.")
    cfg = apply_overrides(OmegaConf.load(args.config), args.set)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    max_context = max(int(value) for value in cfg.model.context_lengths)
    feature_target_offset = int(cfg.model.get("feature_target_offset", 1))
    action_chunk_size = int(cfg.model.get("action_chunk_size", 1))
    dataset_chunk_size = int(cfg.dataset.get("chunk_size", 1))
    available_frames = int(cfg.dataset.future_steps) + 1
    required_frames = max_context + feature_target_offset
    if available_frames < required_frames:
        raise ValueError(
            "Stage-2 dataset does not contain the long-horizon feature target: "
            f"future_steps+1={available_frames}, but max_context+feature_target_offset="
            f"{required_frames}."
        )
    if dataset_chunk_size != action_chunk_size:
        raise ValueError(
            "Action chunk mismatch: dataset.chunk_size="
            f"{dataset_chunk_size}, model.action_chunk_size={action_chunk_size}."
        )
    is_dist, rank, local_rank, world = distributed_info()
    accelerator = configure_accelerator(local_rank=local_rank, distributed=is_dist)
    if rank == 0:
        print(
            "runtime "
            f"accelerator={accelerator.kind} "
            f"backend={accelerator.distributed_backend} "
            f"world_size={world} "
            f"amp={bool(cfg.training.amp)} "
            f"amp_dtype={cfg.training.get('amp_dtype', 'auto')}",
            flush=True,
        )
    if is_dist and accelerator.kind == "cuda":
        # ``flex_attention`` is a higher-order operator.  TorchDynamo's DDP
        # graph/bucket optimizer cannot partition graphs containing it and
        # otherwise fails on the first forward with
        # ``DDPOptimizer ... Found a higher order op``.  Disabling only that
        # graph rewrite preserves compiled flex-attention and normal DDP
        # gradient synchronization.
        torch._dynamo.config.optimize_ddp = False
    device = accelerator.device
    seed = int(cfg.seed) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    manual_seed_all(accelerator, seed)
    if accelerator.kind == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    split_file = dataset_cfg.pop("split_file", None)
    train_dataset_cfg = dict(dataset_cfg)
    eval_dataset_cfg = dict(dataset_cfg)
    if split_file:
        split_payload = json.loads(Path(str(split_file)).expanduser().read_text())
        split_format = split_payload.get("format")
        if split_format == "uavflow_stratified_instruction_v1":
            train_ids = [str(value) for value in split_payload["train_episode_ids"]]
            val_ids = [str(value) for value in split_payload["eval_episode_ids"]]
            overlap = set(train_ids).intersection(val_ids)
            if overlap:
                raise ValueError(f"Explicit split leaks {len(overlap)} episode IDs.")
            train_dataset_cfg["include_episodes"] = train_ids
            eval_dataset_cfg["include_episodes"] = val_ids
        elif split_format == "uavflow_episode_split_v1":
            split_source = Path(str(split_payload.get("source_checkpoint", ""))).expanduser().resolve()
            configured_idm = Path(str(cfg.stage1.idm_checkpoint)).expanduser().resolve()
            if split_source != configured_idm and rank == 0:
                print(
                    "Using fixed episode manifest with a derived IDM checkpoint: "
                    f"split_source={split_source}, configured_idm={configured_idm}",
                    flush=True,
                )
            domains = split_payload["domains"]
            for domain in ("real", "sim"):
                train_ids = [str(value) for value in domains[domain]["train"]]
                val_ids = [str(value) for value in domains[domain]["val"]]
                overlap = set(train_ids).intersection(val_ids)
                if overlap:
                    raise ValueError(
                        f"Explicit {domain} split leaks {len(overlap)} episode IDs."
                    )
                train_dataset_cfg[f"{domain}_include_episodes"] = train_ids
                eval_dataset_cfg[f"{domain}_include_episodes"] = val_ids
        else:
            raise ValueError(f"Unsupported UAV split format in {split_file}: {split_format!r}")
        if rank == 0:
            print(f"Using explicit IDM episode split: {split_file}", flush=True)
    train_set = build_robot_dataset(train_dataset_cfg, is_eval=False)
    eval_set = build_robot_dataset(eval_dataset_cfg, is_eval=True)
    eval_domain_sets: dict[str, torch.utils.data.Dataset] = {}
    if hasattr(eval_set, "datasets") and isinstance(eval_set.datasets, dict):
        eval_items = list(eval_set.datasets.items())
        eval_domain_sets = {str(name): dataset for name, dataset in eval_items}
        if rank == 0:
            print(
                "Physical IDM-split eval: "
                + " + ".join(f"{name}={len(dataset)}" for name, dataset in eval_items)
                + f" total={sum(len(dataset) for _, dataset in eval_items)}",
                flush=True,
            )
        eval_set = ConcatDataset([dataset for _, dataset in eval_items])
    # The frozen IDM predicts actions in its original normalized space. Never
    # recompute/overwrite these statistics from the Stage-2 seven-frame
    # sampling view, even though trajectory-level values should be equivalent.
    normalizer = ActionNormalizer.from_stats_dir(
        str(cfg.stage1.action_stats_dir), [str(dataset_cfg.get("action_stats_key", "uavflow_joint"))]
    )
    pose_normalizer = None
    needs_pose_targets = (
        bool(cfg.model.get("use_pose_history", False))
        or (
            bool(cfg.model.get("stop_head_enabled", False))
            and str(cfg.model.get("stop_head_mode", "")) == "action_hidden_pose"
        )
        or bool(cfg.model.get("relative_pose_head_enabled", False))
        or float(cfg.loss.get("action_rollout_pose_weight", 0.0)) > 0.0
    )
    if needs_pose_targets:
        if not hasattr(train_set, "compute_episode_pose_statistics"):
            raise TypeError("Pose supervision requires UAV episode-pose statistics.")
        pose_stats = train_set.compute_episode_pose_statistics(
            int(cfg.model.get("pose_stats_samples", 200000))
        )
        pose_normalizer = EpisodePoseNormalizer(pose_stats)
        if rank == 0:
            print(
                "Episode-pose target stats (train physical frames): "
                f"q01={pose_normalizer.q01.tolist()} q99={pose_normalizer.q99.tolist()} "
                f"mask={pose_normalizer.mask.tolist()}",
                flush=True,
            )

    da3 = DA3GiantEncoder(
        ckpt_path=str(cfg.stage1.da3_checkpoint), encoder_input_size=224,
        freeze_backbone=True, n_action_steps=0, views_per_timestep=1,
        use_temporal_embed=False,
    ).to(device).eval()
    for parameter in da3.parameters():
        parameter.requires_grad = False
    train_deep_backbone = bool(cfg.model.get("train_deep_backbone", False))
    deep_lora_cfg = cfg.model.get("deep_lora", {}) or {}
    deep_lora_enabled = bool(deep_lora_cfg.get("enabled", False))
    if train_deep_backbone and deep_lora_enabled:
        raise ValueError("Choose either full deep-backbone tuning or deep LoRA, not both.")
    if train_deep_backbone:
        deep_start = int(cfg.model.get("deep_train_start_block", da3.out_layers[0]))
        da3.freeze_blocks_before(deep_start)
        # DPT is a fixed readout just as in GAM; depth gradients optimize the
        # post-boundary transformer blocks and Predictor, not the DPT weights.
        da3.dpt_head.eval()
        for parameter in da3.dpt_head.parameters():
            parameter.requires_grad = False
    elif deep_lora_enabled:
        deep_start = int(deep_lora_cfg.get("start_block", da3.out_layers[0]))
        lora_stats = da3.configure_deep_lora(
            block_idx=deep_start,
            rank=int(deep_lora_cfg.get("rank", 8)),
            alpha=float(deep_lora_cfg.get("alpha", 16.0)),
            dropout=float(deep_lora_cfg.get("dropout", 0.0)),
            target_modules=deep_lora_cfg.get("target_modules"),
        )
        if rank == 0:
            print(
                "DA3 deeper LoRA: "
                f"start={deep_start} rank={int(deep_lora_cfg.get('rank', 8))} "
                f"alpha={float(deep_lora_cfg.get('alpha', 16.0)):.1f} "
                f"wrapped={int(lora_stats['wrapped_modules'])} "
                f"trainable={int(lora_stats['trainable_parameters'])/1e6:.2f}M",
                flush=True,
            )
    train_deep_parameters = train_deep_backbone or deep_lora_enabled
    compute_idm_branch = bool(cfg.model.get("compute_idm_branch", True))
    idm = None
    idm_meta = {"loaded": False, "reason": "compute_idm_branch=false"}
    if compute_idm_branch:
        idm, idm_meta = build_frozen_idm(
            str(cfg.stage1.idm_checkpoint), da3=da3, action_dim=4
        )
        idm = idm.to(device)
    # New closed-loop mode predicts one next frame and left-pads/slides it into
    # an arbitrary multi-frame IDM window. Only legacy open-loop rollout must
    # make the number of generated frames equal IDM.window_size - 1.
    if (
        compute_idm_branch
        and int(cfg.model.rollout_steps) > 1
        and int(idm.window_size) != int(cfg.model.rollout_steps) + 1
    ):
        raise ValueError(
            "Legacy multi-step rollout requires IDM window_size=rollout_steps+1; "
            f"got window={idm.window_size}, rollout={cfg.model.rollout_steps}."
        )
    text = build_stage2_conditioner(cfg.stage1, cfg.model).to(device).eval()
    # Text backends instantiate different numbers of parameters before loading
    # their pretrained states, so they advance PyTorch's RNG by different
    # amounts. Reset the Stage-2 initialization stream here: common Predictor
    # and action-head parameters then start identically in CLIP/T5 ablations.
    model_init_seed = int(cfg.get("model_init_seed", cfg.seed))
    torch.manual_seed(model_init_seed)
    manual_seed_all(accelerator, model_init_seed)
    from experiments.uavflow_predictor_idm.current_geometry import load_current_geometry_read
    from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state, load_architecture_state
    model = UAVFlowPredictorIDM(
        da3=da3, idm=idm, action_dim=4,
        action_chunk_size=int(cfg.model.get("action_chunk_size", 1)),
        feature_target_offset=int(cfg.model.get("feature_target_offset", 1)),
        rollout_steps=int(cfg.model.rollout_steps), d_model=int(cfg.model.d_model),
        depth=int(cfg.model.depth), num_heads=int(cfg.model.num_heads),
        ffn_ratio=float(cfg.model.ffn_ratio), dropout=float(cfg.model.dropout),
        language_dim=int(text.hidden_size),
        language_len=int(cfg.model.get("language_len", 77)),
        variable_language_tokens=bool(
            getattr(text, "variable_length_tokens", False)
        ),
        condition_mode=str(cfg.model.condition_mode),
        use_action_history=bool(cfg.model.get("use_action_history", False)),
        use_pose_history=bool(cfg.model.get("use_pose_history", False)),
        action_history_keep_prob=float(cfg.model.get("action_history_keep_prob", 1.0)),
        pose_history_keep_prob=float(cfg.model.get("pose_history_keep_prob", 1.0)),
        pose_history_xyz_noise_meters=float(
            cfg.model.get("pose_history_xyz_noise_meters", 0.0)
        ),
        pose_history_yaw_noise_degrees=float(
            cfg.model.get("pose_history_yaw_noise_degrees", 0.0)
        ),
        use_fixed_first_frame=bool(cfg.model.get("use_fixed_first_frame", False)),
        use_reference_type_embedding=bool(cfg.model.get("use_reference_type_embedding", False)),
        dense_context_supervision=bool(cfg.model.get("dense_context_supervision", False)),
        direct_action_enabled=bool(cfg.model.get("direct_action_enabled", False)),
        compute_idm_branch=bool(cfg.model.get("compute_idm_branch", True)),
        deep_action_enabled=bool(cfg.model.get("deep_action_enabled", False)),
        current_geometry_action_enabled=bool(cfg.model.get("current_geometry_action_enabled", False)),
        current_geometry_read_mode=str(cfg.model.get("current_geometry_read_mode", "terminal")),
        current_geometry_bank_mode=str(
            cfg.model.get("current_geometry_bank_mode", "output_current")
        ),
        geometry_architecture=str(cfg.model.get("geometry_architecture", "legacy")),
        vlm_action_seed_enabled=bool(
            cfg.model.get("vlm_action_seed_enabled", False)
        ),
        causal_action_decoder_enabled=bool(
            cfg.model.get("causal_action_decoder_enabled", False)
        ),
        causal_action_bins=int(cfg.model.get("causal_action_bins", 256)),
        causal_action_model_dim=int(
            cfg.model.get("causal_action_model_dim", 512)
        ),
        causal_action_num_heads=int(
            cfg.model.get("causal_action_num_heads", 8)
        ),
        causal_action_num_layers=int(
            cfg.model.get("causal_action_num_layers", 2)
        ),
        parallel_vla_gfm_enabled=bool(
            cfg.model.get("parallel_vla_gfm_enabled", False)
        ),
        parallel_vla_gfm_width=int(
            cfg.model.get("parallel_vla_gfm_width", 512)
        ),
        parallel_vla_gfm_heads=int(
            cfg.model.get("parallel_vla_gfm_heads", 8)
        ),
        parallel_vla_gfm_mode=str(
            cfg.model.get("parallel_vla_gfm_mode", "external_query")
        ),
        parallel_action_post_bidir_layers=int(
            cfg.model.get("parallel_action_post_bidir_layers", 0)
        ),
        parallel_current_depth_enabled=bool(
            cfg.model.get("parallel_current_depth_enabled", True)
        ),
        parallel_current_geometry_read_enabled=bool(
            cfg.model.get("parallel_current_geometry_read_enabled", True)
        ),
        parallel_action_decode_mode=str(
            cfg.model.get("parallel_action_decode_mode", "full")
        ),
        train_deep_backbone=train_deep_parameters,
        deep_train_start_block=int(
            deep_lora_cfg.get("start_block", cfg.model.get("deep_train_start_block", da3.out_layers[0]))
        ),
        depth_decode_enabled=float(cfg.loss.get("depth_weight", 0.0)) > 0.0,
        depth_scale_head_enabled=str(
            cfg.loss.get("depth_scale_mode", "per_frame_median")
        ) == "scale_separated",
        depth_scale_init_meters=float(cfg.loss.get("depth_fixed_scale_meters", 100.0)),
        relative_pose_head_enabled=bool(
            cfg.model.get("relative_pose_head_enabled", False)
        ),
        stop_head_enabled=bool(cfg.model.get("stop_head_enabled", False)),
        stop_head_mode=str(cfg.model.get("stop_head_mode", "legacy_action_token")),
        stop_model_dim=int(cfg.model.get("stop_model_dim", 512)),
        stop_num_heads=int(cfg.model.get("stop_num_heads", 8)),
        stop_num_queries=int(cfg.model.get("stop_num_queries", 4)),
        stop_decoder_layers=int(cfg.model.get("stop_decoder_layers", 2)),
        stop_future_gate_init=float(cfg.model.get("stop_future_gate_init", 0.1)),
        residual_prediction=bool(cfg.model.residual_prediction),
        residual_gate_init=float(cfg.model.residual_gate_init),
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
    ).to(device)
    raw_model = model
    trainable_conditioner_names = {
        name for name, parameter in text.named_parameters() if parameter.requires_grad
    }
    if trainable_conditioner_names:
        # Register the conditioner under the policy before DDP wrapping so its
        # LoRA gradients participate in the same reducer and optimizer. The
        # encoding call still happens before policy.forward, but its graph is
        # connected through lang_feats to the returned policy losses.
        raw_model.add_module("trainable_conditioner", text)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if trainable_conditioner_names:
            state = init_ckpt.get("conditioner_trainable")
            if state is not None:
                text.load_state_dict(state, strict=False)
        load_current_geometry_read(
            raw_model, init_ckpt,
            required=(raw_model.current_geometry_action_enabled or raw_model.parallel_vla_gfm_enabled),
        )
        load_architecture_state(raw_model, init_ckpt)
        raw_model.predictor.load_state_dict(init_ckpt["predictor"], strict=True)
        if "direct_action_head" in init_ckpt:
            raw_model.direct_action_head.load_state_dict(init_ckpt["direct_action_head"], strict=True)
        if raw_model.vlm_action_seed_enabled and init_ckpt.get("vlm_action_seed") is not None:
            raw_model.vlm_action_seed.load_state_dict(init_ckpt["vlm_action_seed"], strict=True)
        if raw_model.causal_action_decoder_enabled and init_ckpt.get("causal_action_decoder") is not None:
            raw_model.causal_action_decoder.load_state_dict(
                init_ckpt["causal_action_decoder"], strict=True
            )
        if raw_model.parallel_vla_gfm_enabled:
            bridge_state = init_ckpt.get("semantic_geometry_action")
            oft_state = init_ckpt.get("oft_action_tokenizer")
            head_state = init_ckpt.get("parallel_action_head")
            correction_state = init_ckpt.get("parallel_action_correction_head")
            if bridge_state is not None and raw_model.semantic_geometry_action is not None:
                raw_model.semantic_geometry_action.load_state_dict(bridge_state, strict=True)
            if oft_state is not None and raw_model.oft_action_tokenizer is not None:
                raw_model.oft_action_tokenizer.load_state_dict(oft_state, strict=True)
            if head_state is not None:
                raw_model.parallel_action_head.load_state_dict(head_state, strict=True)
            if raw_model.parallel_action_correction_head is not None:
                if correction_state is None:
                    raise KeyError(
                        "Geometry-residual init checkpoint has no correction-head state."
                    )
                raw_model.parallel_action_correction_head.load_state_dict(
                    correction_state, strict=True
                )
        if raw_model.relative_pose_head_enabled and init_ckpt.get("relative_pose_head") is not None:
            raw_model.relative_pose_head.load_state_dict(
                init_ckpt["relative_pose_head"], strict=True
            )
        if raw_model.depth_scale_head_enabled and init_ckpt.get("depth_scale_head") is not None:
            raw_model.depth_scale_head.load_state_dict(
                init_ckpt["depth_scale_head"], strict=True
            )
        for name in (
            "missing_action_embed", "missing_pose_embed", "reference_step_embed",
            "residual_gate_logit",
        ):
            if name in init_ckpt:
                getattr(raw_model, name).data.copy_(init_ckpt[name])
        if raw_model.stop_head_enabled and init_ckpt.get("stop_head") is not None:
            init_cfg = init_ckpt.get("config", {})
            if OmegaConf.is_config(init_cfg):
                init_cfg = OmegaConf.to_container(init_cfg, resolve=True)
            old_stop_mode = (
                init_cfg.get("model", {}).get("stop_head_mode", "legacy_action_token")
                if isinstance(init_cfg, dict) else "legacy_action_token"
            )
            if str(old_stop_mode) == raw_model.stop_head_mode:
                raw_model.stop_head.load_state_dict(init_ckpt["stop_head"], strict=True)
            elif rank == 0:
                print(
                    f"Not loading incompatible Stop head: checkpoint={old_stop_mode}, "
                    f"current={raw_model.stop_head_mode}; new Stop head starts fresh.",
                    flush=True,
                )
        if train_deep_parameters and init_ckpt.get("da3_trainable") is not None:
            current = raw_model.da3.state_dict()
            current.update(init_ckpt["da3_trainable"])
            raw_model.da3.load_state_dict(current, strict=True)
        if rank == 0:
            print(
                f"Initialized Stage-2 weights from {args.init_checkpoint}; "
                "optimizer, scheduler and data cursor start fresh.",
                flush=True,
            )
    trainable = [p for p in model.parameters() if p.requires_grad]
    direct_head_ids = {
        id(parameter) for parameter in raw_model.direct_action_head.parameters()
        if parameter.requires_grad
    }
    if raw_model.causal_action_decoder is not None:
        direct_head_ids.update(
            id(parameter) for parameter in raw_model.causal_action_decoder.parameters()
            if parameter.requires_grad
        )
    if raw_model.parallel_action_head is not None:
        direct_head_ids.update(
            id(parameter) for parameter in raw_model.parallel_action_head.parameters()
            if parameter.requires_grad
        )
    if raw_model.parallel_action_correction_head is not None:
        direct_head_ids.update(
            id(parameter)
            for parameter in raw_model.parallel_action_correction_head.parameters()
            if parameter.requires_grad
        )
    separate_stop_lr = cfg.training.get("stop_head_lr") is not None
    stop_head_ids = {
        id(parameter) for parameter in raw_model.stop_head.parameters()
        if parameter.requires_grad
    }
    if not separate_stop_lr:
        direct_head_ids.update(stop_head_ids)
    direct_head_ids.update(
        id(parameter) for parameter in raw_model.relative_pose_head.parameters()
        if parameter.requires_grad
    )
    direct_head_ids.update(
        id(parameter) for parameter in raw_model.depth_scale_head.parameters()
        if parameter.requires_grad
    )
    backbone_ids = {
        id(parameter) for parameter in raw_model.da3.parameters()
        if parameter.requires_grad
    }
    predictor_trainable = [
        p for p in trainable
        if id(p) not in direct_head_ids
        and id(p) not in backbone_ids
        and id(p) not in stop_head_ids
    ]
    direct_head_trainable = [p for p in trainable if id(p) in direct_head_ids]
    backbone_trainable = [p for p in trainable if id(p) in backbone_ids]
    stop_head_trainable = [p for p in trainable if id(p) in stop_head_ids]
    predictor_lr_mult = float(cfg.training.get("predictor_lr_mult", 1.0))
    head_lr_mult = float(cfg.training.get("direct_action_head_lr_mult", 1.0))
    optimizer_groups = [{
        "params": predictor_trainable,
        "lr": float(cfg.training.lr) * predictor_lr_mult,
        "lr_multiplier": predictor_lr_mult,
    }]
    if backbone_trainable:
        optimizer_groups.append({
            "params": backbone_trainable,
            "lr": float(cfg.training.lr),
            "lr_multiplier": 1.0,
        })
    if direct_head_trainable:
        optimizer_groups.append({
            "params": direct_head_trainable,
            "lr": float(cfg.training.lr) * head_lr_mult,
            "lr_multiplier": head_lr_mult,
        })
    if separate_stop_lr and stop_head_trainable:
        base_lr = float(cfg.training.lr)
        stop_head_lr = float(cfg.training.stop_head_lr)
        if base_lr <= 0 or stop_head_lr <= 0:
            raise ValueError("training.lr and training.stop_head_lr must be positive")
        optimizer_groups.append({
            "params": stop_head_trainable,
            "lr": stop_head_lr,
            "lr_multiplier": stop_head_lr / base_lr,
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(cfg.training.weight_decay)
    )
    scaler_policy = os.environ.get("UAVFLOW_GRAD_SCALER", "auto").strip().lower()
    scaler = create_grad_scaler(
        device.type,
        enabled=bool(cfg.training.amp and device.type in {"cuda", "npu"}),
        dtype_name=str(cfg.training.get("amp_dtype", "auto")),
        policy=scaler_policy,
    )
    if rank == 0:
        resolved_amp_dtype = (
            str(cfg.training.get("amp_dtype", "auto"))
            if not bool(cfg.training.amp)
            else str(
                {
                    torch.float16: "fp16",
                    torch.bfloat16: "bf16",
                }.get(
                    amp_dtype(str(cfg.training.get("amp_dtype", "auto")), device.type),
                    "unknown",
                )
            )
        )
        print(
            "AMP grad scaler: "
            f"policy={scaler_policy} "
            f"enabled={grad_scaler_enabled(device.type, amp_enabled=bool(cfg.training.amp), dtype_name=str(cfg.training.get('amp_dtype', 'auto')), policy=scaler_policy)} "
            f"dtype={resolved_amp_dtype}",
            flush=True,
        )
    start_step = 0; start_epoch = 0; start_batch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_cfg = ckpt.get("config", {})
        saved_model_cfg = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}
        saved_dataset_cfg = saved_cfg.get("dataset", {}) if isinstance(saved_cfg, dict) else {}
        saved_rollout = int(saved_model_cfg.get("rollout_steps", -1))
        saved_future = int(saved_dataset_cfg.get("future_steps", -1))
        current_rollout = int(cfg.model.rollout_steps)
        current_future = int(cfg.dataset.future_steps)
        if (saved_rollout, saved_future) != (current_rollout, current_future):
            raise ValueError(
                "Cannot resume across Stage-2 temporal protocols: checkpoint has "
                f"rollout_steps={saved_rollout}, future_steps={saved_future}; current config has "
                f"rollout_steps={current_rollout}, future_steps={current_future}. "
                "Start a new run for the one-step sliding-IDM protocol."
            )
        # The cursor reconstructed below is exact only when the data ordering
        # and architecture-defining settings match.  Loss weights and the
        # final training budget may intentionally change on continuation, so
        # they are not part of this compatibility contract.
        compatibility_paths = (
            "stage1.da3_checkpoint",
            "stage1.idm_checkpoint",
            "stage1.action_stats_dir",
            "stage1.text_encoder_type",
            "stage1.t5_model",
            "stage1.qwen_model",
            "stage1.qwen_layers",
            "stage1.qwen_attention_implementation",
            "stage1.qwen_use_reference_image",
            "stage1.qwen_prompt_mode",
            "stage1.qwen_token_selection",
            "stage1.qwen_lora_enabled",
            "stage1.qwen_action_attention_mode",
            "stage1.qwen_lora_rank",
            "stage1.qwen_lora_alpha",
            "stage1.qwen_lora_dropout",
            "stage1.qwen_action_placeholder_count",
            "dataset.openvla_prompt_pose_mode",
            "dataset.split_file",
            "dataset.real_root",
            "dataset.sim_root",
            "dataset.mixture_weights",
            "dataset.chunk_size",
            "dataset.visual_anchor_stride",
            "dataset.action_convention",
            "dataset.action_delta_mode",
            "dataset.endpoint_repeat_count",
            "dataset.endpoint_self_pair_start_count",
            "dataset.endpoint_self_pair_end_count",
            "dataset.endpoint_terminal_window_repeat_count",
            "dataset.endpoint_absorbing_window_count",
            "dataset.endpoint_absorbing_max_starts",
            "dataset.episode_start_repeat_count",
            "dataset.episode_start_train_fraction",
            "model.d_model",
            "model.depth",
            "model.num_heads",
            "model.ffn_ratio",
            "model.condition_mode",
            "model.use_action_history",
            "model.use_pose_history",
            "model.action_history_keep_prob",
            "model.pose_history_keep_prob",
            "model.pose_history_xyz_noise_meters",
            "model.pose_history_yaw_noise_degrees",
            "model.use_fixed_first_frame",
            "model.use_reference_type_embedding",
            "model.dense_context_supervision",
            "model.action_chunk_size",
            "model.feature_target_offset",
            "model.direct_action_enabled",
            "model.compute_idm_branch",
            "model.deep_action_enabled",
            "model.current_geometry_action_enabled",
            "model.current_geometry_read_mode",
            "model.current_geometry_bank_mode",
            "model.geometry_architecture",
            "model.vlm_action_seed_enabled",
            "model.causal_action_decoder_enabled",
            "model.causal_action_bins",
            "model.causal_action_model_dim",
            "model.causal_action_num_heads",
            "model.causal_action_num_layers",
            "model.parallel_vla_gfm_enabled",
            "model.parallel_vla_gfm_width",
            "model.parallel_vla_gfm_heads",
            "model.parallel_vla_gfm_mode",
            "model.parallel_action_post_bidir_layers",
            "model.parallel_current_depth_enabled",
            "model.parallel_current_geometry_read_enabled",
            "model.relative_pose_head_enabled",
            "loss.depth_scale_mode",
            "model.train_deep_backbone",
            "model.deep_train_start_block",
            "model.deep_lora.enabled",
            "model.deep_lora.start_block",
            "model.deep_lora.rank",
            "model.deep_lora.alpha",
            "model.deep_lora.dropout",
            "model.stop_head_enabled",
            "model.stop_head_mode",
            "model.stop_model_dim",
            "model.stop_num_heads",
            "model.stop_num_queries",
            "model.stop_decoder_layers",
            "model.stop_future_gate_init",
            "model.language_len",
            "model.residual_prediction",
            "loss.depth_target_mode",
            "loss.depth_fixed_scale_meters",
            "loss.depth_scale_loss_weight",
            "training.batch_size",
            "training.amp_dtype",
            "training.direct_action_head_lr_mult",
            "training.predictor_lr_mult",
            "training.stop_head_lr",
        )

        def nested_get(mapping, path):
            value = mapping
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    return None
                value = value[part]
            return value

        current_cfg = cfg_dict if isinstance(cfg_dict, dict) else {}
        mismatches = []
        for path in compatibility_paths:
            saved_value = nested_get(saved_cfg, path)
            current_value = nested_get(current_cfg, path)
            if path == "model.current_geometry_action_enabled":
                # Checkpoints predating CA1_HB omit this disabled-by-default flag.
                saved_value, current_value = bool(saved_value), bool(current_value)
            if path == "model.geometry_architecture":
                saved_value, current_value = saved_value or "legacy", current_value or "legacy"
            if path == "model.current_geometry_read_mode":
                saved_value, current_value = saved_value or "terminal", current_value or "terminal"
            if path == "model.current_geometry_bank_mode":
                saved_value = saved_value or "every_layer_concat"
                current_value = current_value or "every_layer_concat"
            if saved_value != current_value:
                mismatches.append(f"{path}: checkpoint={saved_value!r}, current={current_value!r}")
        saved_world = ckpt.get("world_size")
        if saved_world is not None and int(saved_world) != int(world):
            mismatches.append(
                f"world_size: checkpoint={int(saved_world)!r}, current={int(world)!r}"
            )
        saved_train_size = ckpt.get("train_size")
        if saved_train_size is not None and int(saved_train_size) != int(len(train_set)):
            mismatches.append(
                f"train_size: checkpoint={int(saved_train_size)!r}, current={int(len(train_set))!r}"
            )
        if mismatches:
            raise ValueError(
                "Cannot exactly resume with a changed architecture/data cursor:\n  "
                + "\n  ".join(mismatches)
            )
        raw_model.predictor.load_state_dict(ckpt["predictor"])
        if trainable_conditioner_names:
            state = ckpt.get("conditioner_trainable")
            if state is None:
                raise KeyError("Qwen-LoRA resume checkpoint has no conditioner_trainable state")
            text.load_state_dict(state, strict=False)
        if raw_model.vlm_action_seed_enabled:
            if ckpt.get("vlm_action_seed") is None:
                raise KeyError("VLM-action-seed resume checkpoint has no projector state.")
            raw_model.vlm_action_seed.load_state_dict(ckpt["vlm_action_seed"], strict=True)
        if raw_model.causal_action_decoder_enabled:
            if ckpt.get("causal_action_decoder") is None:
                raise KeyError("Causal-action resume checkpoint has no decoder state.")
            raw_model.causal_action_decoder.load_state_dict(
                ckpt["causal_action_decoder"], strict=True
            )
        if raw_model.parallel_vla_gfm_enabled:
            active_initializer = (
                raw_model.oft_action_tokenizer
                if raw_model.oft_action_tokenizer is not None
                else raw_model.semantic_geometry_action
            )
            state_key = (
                "oft_action_tokenizer"
                if raw_model.oft_action_tokenizer is not None
                else "semantic_geometry_action"
            )
            if ckpt.get(state_key) is None:
                raise KeyError(f"dual_vla_gfm checkpoint has no {state_key} state.")
            if ckpt.get("parallel_action_head") is None:
                raise KeyError("dual_vla_gfm checkpoint has no parallel action-head state.")
            active_initializer.load_state_dict(ckpt[state_key], strict=True)
            raw_model.parallel_action_head.load_state_dict(
                ckpt["parallel_action_head"], strict=True
            )
            if raw_model.parallel_action_correction_head is not None:
                state = ckpt.get("parallel_action_correction_head")
                if state is None:
                    raise KeyError(
                        "Geometry-residual resume checkpoint has no correction-head state."
                    )
                raw_model.parallel_action_correction_head.load_state_dict(
                    state, strict=True
                )
        load_current_geometry_read(raw_model, ckpt, required=True)
        load_architecture_state(raw_model, ckpt)
        raw_model.residual_gate_logit.data.copy_(ckpt["residual_gate_logit"])
        if "missing_action_embed" in ckpt:
            raw_model.missing_action_embed.data.copy_(ckpt["missing_action_embed"])
        if "missing_pose_embed" in ckpt:
            raw_model.missing_pose_embed.data.copy_(ckpt["missing_pose_embed"])
        if "reference_step_embed" in ckpt:
            raw_model.reference_step_embed.data.copy_(ckpt["reference_step_embed"])
        if "direct_action_head" in ckpt:
            raw_model.direct_action_head.load_state_dict(ckpt["direct_action_head"], strict=True)
        if raw_model.relative_pose_head_enabled:
            state = ckpt.get("relative_pose_head")
            if state is None:
                raise KeyError("Relative-pose-enabled resume checkpoint has no head state.")
            raw_model.relative_pose_head.load_state_dict(state, strict=True)
        if train_deep_parameters:
            if "da3_trainable" not in ckpt:
                raise KeyError("Deep-backbone resume checkpoint has no da3_trainable state.")
            current = raw_model.da3.state_dict()
            current.update(ckpt["da3_trainable"])
            raw_model.da3.load_state_dict(current, strict=True)
        if raw_model.stop_head_enabled:
            if "stop_head" not in ckpt:
                raise KeyError("Stop-enabled resume checkpoint has no stop_head state.")
            raw_model.stop_head.load_state_dict(ckpt["stop_head"], strict=True)
        if raw_model.depth_scale_head_enabled:
            if ckpt.get("depth_scale_head") is None:
                raise KeyError("Scale-separated resume checkpoint has no depth_scale_head state.")
            raw_model.depth_scale_head.load_state_dict(
                ckpt["depth_scale_head"], strict=True
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]); start_epoch = int(ckpt.get("epoch", 0)); start_batch = int(ckpt.get("batch", 0))
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )

    train_sampler = (
        CursorDistributedSampler(
            train_set,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=int(cfg.seed),
            resume_epoch=start_epoch,
            resume_batch=start_batch,
            batch_size=int(cfg.training.batch_size),
        )
        if is_dist else None
    )
    # A bounded evaluation must not consume the Real-first ConcatDataset prefix
    # (which previously contained zero Sim windows).  A fixed shuffled sampler
    # gives every evaluation the same deterministic subset over all physical
    # validation windows and both domains.
    eval_sampler = (
        DistributedSampler(
            eval_set, num_replicas=world, rank=rank, shuffle=True,
            seed=int(cfg.seed), drop_last=False,
        )
        if is_dist else None
    )
    train_loader = DataLoader(
        train_set, batch_size=int(cfg.training.batch_size), sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=int(cfg.training.num_workers),
        pin_memory=accelerator.kind == "cuda", drop_last=True,
        persistent_workers=int(cfg.training.num_workers) > 0,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=int(cfg.training.batch_size), sampler=eval_sampler,
        shuffle=False, num_workers=max(1, int(cfg.training.num_workers) // 2),
        pin_memory=accelerator.kind == "cuda", drop_last=False,
    )
    # Keep domain-specific validation loaders as well as the historical mixed
    # loader.  In particular, ``sim_H*_action/raw`` uses the same normalized
    # and physical action units as the Stage-1 IDM Sim evaluation, so Stage-2
    # degradation can be measured without Real/Sim mixture bias.
    eval_domain_loaders: dict[str, DataLoader] = {}
    for domain_index, (domain_name, domain_set) in enumerate(eval_domain_sets.items()):
        domain_sampler = (
            DistributedSampler(
                domain_set, num_replicas=world, rank=rank, shuffle=True,
                seed=int(cfg.seed) + 1000 + domain_index, drop_last=False,
            )
            if is_dist else None
        )
        eval_domain_loaders[domain_name] = DataLoader(
            domain_set,
            batch_size=int(cfg.training.batch_size),
            sampler=domain_sampler,
            shuffle=False,
            num_workers=max(1, int(cfg.training.num_workers) // 2),
            pin_memory=accelerator.kind == "cuda",
            drop_last=False,
        )
    output = Path(str(cfg.training.results_dir)); output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
        if hasattr(train_set, "traj") and hasattr(eval_set, "traj"):
            split_manifest = {
                "format": "uavflow_stratified_instruction_v1",
                "episode_split_mode": str(cfg.dataset.get("episode_split_mode", "sorted")),
                "episode_split_seed": int(cfg.dataset.get("episode_split_seed", 42)),
                "train_episode_ids": sorted(str(key) for key in train_set.traj),
                "eval_episode_ids": sorted(str(key) for key in eval_set.traj),
                "category_counts": getattr(train_set, "split_category_counts", None),
            }
            (output / "episode_split.json").write_text(
                json.dumps(split_manifest, indent=2, ensure_ascii=False)
            )
        (output / "meta.json").write_text(json.dumps({
            "idm_meta": idm_meta, "world_size": world,
            "trainable_params": sum(p.numel() for p in trainable),
        }, indent=2, default=str))
        print(
            f"train={len(train_set)} eval={len(eval_set)} "
            f"trainable={sum(p.numel() for p in trainable)/1e6:.1f}M "
            f"use_action_history={raw_model.use_action_history} "
            f"use_pose_history={raw_model.use_pose_history}"
            f" use_fixed_first_frame={raw_model.use_fixed_first_frame}"
            f" reference_type={raw_model.use_reference_type_embedding}"
            f" variable_language_tokens={raw_model.predictor.variable_language_tokens}"
            f" dense_context={raw_model.dense_context_supervision}"
            f" direct_action={raw_model.direct_action_enabled}"
            f" vlm_action_seed={raw_model.vlm_action_seed_enabled}"
            f" causal_action_decoder={raw_model.causal_action_decoder_enabled}"
            f" geometry_architecture={raw_model.geometry_architecture}"
            f" idm_branch={raw_model.compute_idm_branch}"
            f" action_chunk={raw_model.action_chunk_size}"
            f" feature_target_offset={raw_model.feature_target_offset}"
        )

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume CHECKPOINT.")
        selected_loaders: dict[str, DataLoader] = {}
        if args.eval_domain in {"all", "mixed"}:
            selected_loaders["mixed"] = eval_loader
        if args.eval_domain in {"all", "real", "sim"}:
            for domain_name, domain_loader in eval_domain_loaders.items():
                short_name = (
                    "sim" if "sim" in domain_name.lower()
                    else "real" if "real" in domain_name.lower()
                    else domain_name
                )
                if args.eval_domain in {"all", short_name}:
                    selected_loaders[short_name] = domain_loader
        evaluation: dict[str, dict[str, float]] = {}
        for eval_name, selected_loader in selected_loaders.items():
            evaluation[eval_name] = evaluate(
                model=model, da3=da3, text=text, normalizer=normalizer,
                pose_normalizer=pose_normalizer,
                loader=selected_loader, device=device,
                contexts=sorted({int(value) for value in cfg.model.context_lengths}),
                rollout=int(cfg.model.rollout_steps),
                horizon_weights=list(cfg.loss.feature_horizon_weights),
                feature_patch_weight=float(cfg.loss.feature_patch_weight),
                feature_cls_weight=float(cfg.loss.feature_cls_weight),
                feature_register_weight=float(cfg.loss.feature_register_weight),
                deep_feature_enabled=float(cfg.loss.get("deep_feature_weight", 0.0)) > 0.0,
                deep_feature_patch_weight=float(cfg.loss.get("deep_feature_patch_weight", 1.0)),
                deep_feature_cls_weight=float(cfg.loss.get("deep_feature_cls_weight", 0.25)),
                stop_pos_weight=float(cfg.loss.get("stop_pos_weight", 1.0)),
                action_direct_weight=float(cfg.loss.get("action_direct_weight", 1.0)),
                action_refine_weight=float(cfg.loss.get("action_refine_weight", 1.0)),
                dual_branch_action_aux_weight=float(
                    cfg.loss.get("dual_branch_action_aux_weight", 0.5)
                ),
                max_batches=int(cfg.training.eval_max_batches),
                amp=bool(cfg.training.amp),
                amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
                depth_target_source=str(cfg.loss.get("depth_target_source", "ue_gt")),
                depth_scale_mode=str(cfg.loss.get("depth_scale_mode", "per_frame_median")),
                depth_target_mode=str(cfg.loss.get("depth_target_mode", "future")),
                depth_fixed_scale_meters=float(cfg.loss.get("depth_fixed_scale_meters", 100.0)),
                depth_scale_loss_weight=float(cfg.loss.get("depth_scale_loss_weight", 1.0)),
                depth_linear_weight=float(cfg.loss.get("depth_linear_weight", 0.5)),
                depth_log_weight=float(cfg.loss.get("depth_log_weight", 1.0)),
                depth_log_epsilon_normalized=float(
                    cfg.loss.get("depth_log_epsilon_normalized", 1e-6)
                ),
                depth_grad_weight=float(cfg.loss.get("depth_grad_weight", 1.0)),
                depth_semantic_weight=float(cfg.loss.get("depth_semantic_weight", 3.0)),
                depth_semantic_separate_weight=float(
                    cfg.loss.get("depth_semantic_separate_weight", 0.0)
                ),
                depth_semantic_linear_weight=float(
                    cfg.loss.get("depth_semantic_linear_weight", cfg.loss.get("depth_linear_weight", 0.5))
                ),
                depth_semantic_log_weight=float(
                    cfg.loss.get("depth_semantic_log_weight", cfg.loss.get("depth_log_weight", 1.0))
                ),
                depth_semantic_grad_weight=float(
                    cfg.loss.get("depth_semantic_grad_weight", cfg.loss.get("depth_grad_weight", 1.0))
                ),
                depth_gradient_mode=str(
                    cfg.loss.get("depth_gradient_mode", "tolerant_log")
                ),
                depth_gradient_tolerance_pixels=int(
                    cfg.loss.get("depth_gradient_tolerance_pixels", 2)
                ),
            )
        if rank == 0:
            payload = {
                "checkpoint": str(args.resume),
                "step": int(start_step),
                "protocol": {
                    "rollout_steps": int(cfg.model.rollout_steps),
                    "future_steps": int(cfg.dataset.future_steps),
                    "visual_anchor_stride": int(cfg.dataset.get("visual_anchor_stride", cfg.dataset.chunk_size)),
                    "action_chunk_size": int(cfg.model.get("action_chunk_size", 1)),
                    "feature_target_offset": int(cfg.model.get("feature_target_offset", 1)),
                },
                "evaluation": evaluation,
            }
            destination = output / f"eval_ckpt_{start_step:07d}.json"
            destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            for eval_name, values in evaluation.items():
                print(
                    f"[eval_only step={start_step:07d} domain={eval_name}] "
                    + " ".join(f"{key}={value:.5f}" for key, value in values.items()),
                    flush=True,
                )
            print(f"Saved evaluation: {destination}", flush=True)
        if is_dist:
            dist.destroy_process_group()
        return

    configured_epochs = int(cfg.training.get("max_epochs", 0))
    samples_per_rank = (
        int(math.ceil(len(train_set) / float(world))) if is_dist else len(train_set)
    )
    steps_per_epoch = samples_per_rank // int(cfg.training.batch_size)
    max_steps = (
        configured_epochs * steps_per_epoch
        if configured_epochs > 0
        else int(cfg.training.max_steps)
    )
    scheduler_total_steps = int(
        cfg.training.get("scheduler_total_steps", 0) or max_steps
    )
    if scheduler_total_steps < max_steps:
        raise ValueError(
            f"training.scheduler_total_steps={scheduler_total_steps} must be >= max_steps={max_steps}."
        )
    if rank == 0:
        print(
            f"training_budget steps_per_epoch={steps_per_epoch} "
            f"max_epochs={configured_epochs} max_steps={max_steps}",
            flush=True,
        )
    def checkpoint_payload(saved_step: int, saved_epoch: int, saved_batch: int) -> dict:
        return {
            "step": saved_step, "epoch": saved_epoch, "batch": saved_batch,
            "world_size": world, "train_size": len(train_set),
            "predictor": raw_model.predictor.state_dict(),
            "geometry_architecture_state": architecture_state(raw_model),
            "current_geometry_read_mode": raw_model.current_geometry_read_mode,
            "current_geometry_bank_mode": raw_model.current_geometry_bank_mode,
            "current_geometry_read": (
                raw_model.current_geometry_read.state_dict()
                if raw_model.current_geometry_read is not None else None
            ),
            "residual_gate_logit": raw_model.residual_gate_logit.detach().cpu(),
            "missing_action_embed": raw_model.missing_action_embed.detach().cpu(),
            "missing_pose_embed": raw_model.missing_pose_embed.detach().cpu(),
            "reference_step_embed": raw_model.reference_step_embed.detach().cpu(),
            "direct_action_head": raw_model.direct_action_head.state_dict(),
            "vlm_action_seed": (
                raw_model.vlm_action_seed.state_dict()
                if raw_model.vlm_action_seed is not None else None
            ),
            "causal_action_decoder": (
                raw_model.causal_action_decoder.state_dict()
                if raw_model.causal_action_decoder is not None else None
            ),
            "semantic_geometry_action": (
                raw_model.semantic_geometry_action.state_dict()
                if raw_model.semantic_geometry_action is not None else None
            ),
            "oft_action_tokenizer": (
                raw_model.oft_action_tokenizer.state_dict()
                if raw_model.oft_action_tokenizer is not None else None
            ),
            "parallel_action_head": (
                raw_model.parallel_action_head.state_dict()
                if raw_model.parallel_action_head is not None else None
            ),
            "parallel_action_correction_head": (
                raw_model.parallel_action_correction_head.state_dict()
                if raw_model.parallel_action_correction_head is not None else None
            ),
            "conditioner_trainable": (
                {
                    name: value.detach().cpu()
                    for name, value in text.state_dict().items()
                    if name in trainable_conditioner_names
                }
                if trainable_conditioner_names else None
            ),
            "relative_pose_head": (
                raw_model.relative_pose_head.state_dict()
                if raw_model.relative_pose_head_enabled else None
            ),
            "depth_scale_head": (
                raw_model.depth_scale_head.state_dict()
                if raw_model.depth_scale_head_enabled else None
            ),
            "da3_trainable": {
                name: value.detach().cpu()
                for name, value in raw_model.da3.named_parameters()
                if value.requires_grad
            },
            "stop_head": raw_model.stop_head.state_dict() if raw_model.stop_head_enabled else None,
            "pose_stats": pose_normalizer.state_dict() if pose_normalizer is not None else None,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "config": cfg_dict,
        }

    def atomic_save_checkpoint(payload: dict, destination: Path) -> None:
        """Write a checkpoint atomically so interruption cannot corrupt resume."""
        temporary = destination.with_name(
            f".{destination.name}.tmp-rank0-{os.getpid()}"
        )
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def point_last_at(numbered_checkpoint: Path) -> None:
        """Make last.pt an atomic, zero-extra-space alias of an epoch checkpoint."""
        destination = output / "last.pt"
        temporary = output / f".last.pt.tmp-rank0-{os.getpid()}"
        try:
            temporary.unlink(missing_ok=True)
            os.link(numbered_checkpoint, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_numbered_checkpoint(
        saved_step: int, saved_epoch: int, saved_batch: int
    ) -> Path:
        payload = checkpoint_payload(saved_step, saved_epoch, saved_batch)
        destination = output / f"ckpt_{saved_step:07d}.pt"
        atomic_save_checkpoint(payload, destination)
        point_last_at(destination)
        return destination

    best_action_path = output / "best_action.pt"
    best_action_meta_path = output / "best_action.json"
    best_action_value = float("inf")
    if best_action_meta_path.is_file():
        try:
            best_action_value = float(
                json.loads(best_action_meta_path.read_text())["value"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            if rank == 0:
                print(
                    f"Ignoring invalid best-action metadata: {best_action_meta_path}",
                    flush=True,
                )

    step = start_step; epoch = start_epoch
    benchmark = StepBenchmark.from_environment(accelerator)
    profiler = create_npu_profiler(accelerator)
    if profiler is not None:
        profiler.start()
    last_saved_step = start_step if args.resume else -1
    pbar = tqdm(total=max_steps, initial=step, disable=rank != 0, desc="uav-predictor-idm")
    model.train()
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        # With DDP, CursorDistributedSampler has already sliced away consumed
        # samples. Keep a logical (pre-slice) batch index for future checkpoints.
        resumed_batch_offset = start_batch if (is_dist and epoch == start_epoch) else 0
        for batch_idx, batch in enumerate(train_loader):
            if not is_dist and epoch == start_epoch and batch_idx < start_batch:
                continue
            logical_batch_idx = batch_idx + resumed_batch_offset
            if step >= max_steps:
                break
            step += 1
            if benchmark is not None:
                benchmark.before_step(step)
            lr_schedule = str(cfg.training.get("lr_schedule", "cosine")).lower()
            if lr_schedule == "constant":
                lr = float(cfg.training.lr)
            elif lr_schedule == "cosine":
                lr = float(cfg.training.lr) * lr_scale(
                    step, warmup=int(cfg.training.warmup_steps), total=scheduler_total_steps,
                    min_ratio=float(cfg.training.min_lr_ratio),
                )
            else:
                raise ValueError(f"Unsupported training.lr_schedule={lr_schedule!r}.")
            for group in optimizer.param_groups:
                group["lr"] = lr * float(group.get("lr_multiplier", 1.0))
            batch = move_batch(batch, device)
            # Step-keyed randomness makes context sampling and augmentation
            # bitwise resume-stable without serializing global CUDA RNG state.
            batch_generator = step_generator(
                accelerator, int(cfg.seed) + step * world + rank
            )
            augmented_sequence = temporal_color_augment(
                torch.cat([
                    batch["episode_first_image"][:, None],
                    batch["all_view_images"],
                ], dim=1),
                OmegaConf.to_container(cfg.augmentation, resolve=True),
                generator=batch_generator,
            )
            batch["episode_first_image"] = augmented_sequence[:, 0]
            batch["all_view_images"] = augmented_sequence[:, 1:]
            h = choose_context_length(
                OmegaConf.to_container(cfg.model, resolve=True), device,
                generator=batch_generator,
            )
            values = forward_batch(
                model=model, da3=da3, text=text, normalizer=normalizer, batch=batch,
                pose_normalizer=pose_normalizer,
                context_len=h, rollout_steps=int(cfg.model.rollout_steps),
                feature_horizon_weights=list(cfg.loss.feature_horizon_weights),
                feature_patch_weight=float(cfg.loss.feature_patch_weight),
                feature_cls_weight=float(cfg.loss.feature_cls_weight),
                feature_register_weight=float(cfg.loss.feature_register_weight),
                deep_feature_enabled=float(cfg.loss.get("deep_feature_weight", 0.0)) > 0.0,
                deep_feature_patch_weight=float(cfg.loss.get("deep_feature_patch_weight", 1.0)),
                deep_feature_cls_weight=float(cfg.loss.get("deep_feature_cls_weight", 0.25)),
                stop_pos_weight=float(cfg.loss.get("stop_pos_weight", 1.0)),
                action_direct_weight=float(cfg.loss.get("action_direct_weight", 1.0)),
                action_refine_weight=float(cfg.loss.get("action_refine_weight", 1.0)),
                dual_branch_action_aux_weight=float(
                    cfg.loss.get("dual_branch_action_aux_weight", 0.5)
                ),
                amp=bool(cfg.training.amp),
                amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
                depth_target_source=str(cfg.loss.get("depth_target_source", "ue_gt")),
                depth_scale_mode=str(cfg.loss.get("depth_scale_mode", "per_frame_median")),
                depth_target_mode=str(cfg.loss.get("depth_target_mode", "future")),
                depth_fixed_scale_meters=float(cfg.loss.get("depth_fixed_scale_meters", 100.0)),
                depth_scale_loss_weight=float(cfg.loss.get("depth_scale_loss_weight", 1.0)),
                depth_linear_weight=float(cfg.loss.get("depth_linear_weight", 0.5)),
                depth_log_weight=float(cfg.loss.get("depth_log_weight", 1.0)),
                depth_log_epsilon_normalized=float(
                    cfg.loss.get("depth_log_epsilon_normalized", 1e-6)
                ),
                depth_grad_weight=float(cfg.loss.get("depth_grad_weight", 1.0)),
                depth_semantic_weight=float(cfg.loss.get("depth_semantic_weight", 3.0)),
                depth_semantic_separate_weight=float(
                    cfg.loss.get("depth_semantic_separate_weight", 0.0)
                ),
                depth_semantic_linear_weight=float(
                    cfg.loss.get("depth_semantic_linear_weight", cfg.loss.get("depth_linear_weight", 0.5))
                ),
                depth_semantic_log_weight=float(
                    cfg.loss.get("depth_semantic_log_weight", cfg.loss.get("depth_log_weight", 1.0))
                ),
                depth_semantic_grad_weight=float(
                    cfg.loss.get("depth_semantic_grad_weight", cfg.loss.get("depth_grad_weight", 1.0))
                ),
                depth_gradient_mode=str(
                    cfg.loss.get("depth_gradient_mode", "tolerant_log")
                ),
                depth_gradient_tolerance_pixels=int(
                    cfg.loss.get("depth_gradient_tolerance_pixels", 2)
                ),
                conditioning_generator=batch_generator,
            )
            weighted_action = float(cfg.loss.action_weight) * values["action"]
            weighted_feature = float(cfg.loss.feature_weight) * values["feature"]
            if "depth_global_weight" in cfg.loss or "depth_dynamic_weight" in cfg.loss:
                weighted_depth_global = float(
                    cfg.loss.get("depth_global_weight", 0.0)
                ) * values["depth_global"]
                weighted_depth_dynamic = float(
                    cfg.loss.get("depth_dynamic_weight", 0.0)
                ) * values["depth_semantic_total"]
            else:
                weighted_depth_global = float(
                    cfg.loss.get("depth_weight", 0.0)
                ) * values["depth"]
                weighted_depth_dynamic = values["depth"].new_zeros(())
            loss = (
                weighted_action
                + weighted_feature
                + float(cfg.loss.get("deep_feature_weight", 0.0)) * values["deep_feature"]
                + float(cfg.loss.get("stop_weight", 0.0)) * values["stop"]
                + weighted_depth_global
                + weighted_depth_dynamic
                + float(cfg.loss.get("relative_pose_weight", 0.0))
                * values["relative_pose"]
                + float(cfg.loss.get("action_rollout_pose_weight", 0.0))
                * values["action_rollout_pose"]
                + float(cfg.loss.get("pose_consistency_weight", 0.0))
                * values["pose_consistency"]
            )
            optimizer.zero_grad(set_to_none=True)
            with profile_phase("R1/backward"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(
                    trainable, float(cfg.training.grad_clip)
                )
            with profile_phase("R1/optimizer"):
                scaler.step(optimizer); scaler.update()
            if profiler is not None:
                profiler.step()
            if benchmark is not None:
                benchmark_line = benchmark.after_step(step)
                if benchmark_line is not None and rank == 0:
                    print("\n" + benchmark_line, flush=True)
                    with (output / "train.log").open("a") as handle:
                        handle.write(benchmark_line + "\n")
            if rank == 0 and step % int(cfg.training.log_every) == 0:
                feat_steps = "/".join(
                    f"{float(value):.4f}"
                    for value in values["feature_per_horizon"].detach().float().cpu()
                )
                cos_steps = "/".join(
                    f"{float(value):.4f}"
                    for value in values["feature_cos_per_horizon"].detach().float().cpu()
                )
                memory_gb = max_memory_allocated_gb(accelerator)
                line = (
                    f"[step={step:07d}] total={loss.item():.4f} H={h} "
                    f"action={values['action'].item():.4f} "
                    f"direct={values['direct_action'].item():.4f} "
                    f"refine={values['refine_action'].item():.4f} "
                    f"dual_cur={values['action_dual_current'].item():.4f} "
                    f"dual_fut={values['action_dual_future'].item():.4f} "
                    f"dual_fgate={values['action_dual_future_gate'].item():.3f} "
                    f"raw={values['raw_l1'].item():.5f} "
                    f"zero_slots={values['target_zero_slot_fraction'].item():.3f} "
                    f"stop={values['stop'].item():.4f} "
                    f"stop_acc={values['stop_accuracy'].item():.3f} "
                    f"stop_rec={values['stop_recall'].item():.3f} "
                    f"stop_rate={values['stop_rate'].item():.3f} "
                    f"stop_p+={values['stop_positive_probability'].item():.3f} "
                    f"stop_p-={values['stop_negative_probability'].item():.3f} "
                    f"stop_fgate={values['stop_future_gate'].item():.3f} "
                    f"feat={values['feature'].item():.4f} "
                    f"patch={values['feature_patch'].item():.4f} "
                    f"cls={values['feature_cls'].item():.4f} "
                    f"reg={values['feature_register'].item():.4f} "
                    f"cos={values['feature_cos'].item():.4f} "
                    f"deep={values['deep_feature'].item():.4f} "
                    f"deep_patch={values['deep_feature_patch'].item():.4f} "
                    f"deep_cls={values['deep_feature_cls'].item():.4f} "
                    f"deep_cos={values['deep_feature_cos'].item():.4f} "
                    f"depth={values['depth'].item():.4f} "
                    f"depth_linear={values['depth_linear_l1'].item():.4f} "
                    f"depth_log={values['depth_log_l1'].item():.4f} "
                    f"depth_grad={values['depth_grad'].item():.4f} "
                    f"depth_scale={values['depth_scale_l1'].item():.4f} "
                    f"scene_scale={values['depth_scene_scale_mean'].item():.3f} "
                    f"depth_valid={values['depth_valid_ratio'].item():.3f} "
                    f"depth_sem={values['depth_semantic_ratio'].item():.3f} "
                    f"depth_sem_loss={values['depth_semantic_total'].item():.4f} "
                    f"depth_sem_linear={values['depth_semantic_linear_l1'].item():.4f} "
                    f"depth_sem_log={values['depth_semantic_log_l1'].item():.4f} "
                    f"depth_sem_grad={values['depth_semantic_grad'].item():.4f} "
                    f"pose5={values['relative_pose'].item():.4f} "
                    f"pose5_xyz={values['relative_pose_translation'].item():.4f} "
                    f"pose5_xyz_axes={values['relative_pose_translation_axes'].tolist()} "
                    f"pose5_yaw={values['relative_pose_yaw'].item():.4f} "
                    f"action_pose5={values['action_rollout_pose'].item():.4f} "
                    f"action_pose5_xyz={values['action_rollout_pose_translation'].item():.4f} "
                    f"action_pose5_xyz_axes={values['action_rollout_pose_translation_axes'].tolist()} "
                    f"action_pose5_xyz_m={values['action_rollout_pose_translation_axes_meters'].tolist()} "
                    f"action_pose5_yaw={values['action_rollout_pose_yaw'].item():.4f} "
                    f"pose_cons={values['pose_consistency'].item():.4f} "
                    f"feat_steps={feat_steps} cos_steps={cos_steps} "
                    f"weighted_action={weighted_action.item():.4f} "
                    f"weighted_feat={weighted_feature.item():.4f} "
                    f"weighted_depth_global={weighted_depth_global.item():.4f} "
                    f"weighted_depth_dynamic={weighted_depth_dynamic.item():.4f} "
                    f"gate={values['gate'].item():.4f} grad={float(grad):.3f} "
                    f"lr_pred/backbone/head={lr * predictor_lr_mult:.2e}/"
                    f"{lr:.2e}/{lr * head_lr_mult:.2e} "
                    f"peak_mem={memory_gb:.2f}GB"
                )
                line = compact_train_line(line, values, cfg)
                print("\n" + line, flush=True)
                with (output / "train.log").open("a") as handle: handle.write(line + "\n")
            if int(cfg.training.eval_every) > 0 and step % int(cfg.training.eval_every) == 0:
                ev = evaluate(
                    model=model, da3=da3, text=text, normalizer=normalizer,
                    pose_normalizer=pose_normalizer,
                    loader=eval_loader, device=device,
                    contexts=sorted({int(value) for value in cfg.model.context_lengths}),
                    rollout=int(cfg.model.rollout_steps),
                    horizon_weights=list(cfg.loss.feature_horizon_weights),
                    feature_patch_weight=float(cfg.loss.feature_patch_weight),
                    feature_cls_weight=float(cfg.loss.feature_cls_weight),
                    feature_register_weight=float(cfg.loss.feature_register_weight),
                    deep_feature_enabled=float(cfg.loss.get("deep_feature_weight", 0.0)) > 0.0,
                    deep_feature_patch_weight=float(cfg.loss.get("deep_feature_patch_weight", 1.0)),
                    deep_feature_cls_weight=float(cfg.loss.get("deep_feature_cls_weight", 0.25)),
                    stop_pos_weight=float(cfg.loss.get("stop_pos_weight", 1.0)),
                    action_direct_weight=float(cfg.loss.get("action_direct_weight", 1.0)),
                    action_refine_weight=float(cfg.loss.get("action_refine_weight", 1.0)),
                    dual_branch_action_aux_weight=float(
                        cfg.loss.get("dual_branch_action_aux_weight", 0.5)
                    ),
                    max_batches=int(cfg.training.eval_max_batches), amp=bool(cfg.training.amp),
                    amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
                    depth_target_source=str(cfg.loss.get("depth_target_source", "ue_gt")),
                depth_scale_mode=str(cfg.loss.get("depth_scale_mode", "per_frame_median")),
                depth_target_mode=str(cfg.loss.get("depth_target_mode", "future")),
                depth_fixed_scale_meters=float(cfg.loss.get("depth_fixed_scale_meters", 100.0)),
                depth_scale_loss_weight=float(cfg.loss.get("depth_scale_loss_weight", 1.0)),
                    depth_linear_weight=float(cfg.loss.get("depth_linear_weight", 0.5)),
                    depth_log_weight=float(cfg.loss.get("depth_log_weight", 1.0)),
                    depth_log_epsilon_normalized=float(
                        cfg.loss.get("depth_log_epsilon_normalized", 1e-6)
                    ),
                    depth_grad_weight=float(cfg.loss.get("depth_grad_weight", 1.0)),
                    depth_semantic_weight=float(cfg.loss.get("depth_semantic_weight", 3.0)),
                    depth_semantic_separate_weight=float(
                        cfg.loss.get("depth_semantic_separate_weight", 0.0)
                    ),
                    depth_semantic_linear_weight=float(
                        cfg.loss.get("depth_semantic_linear_weight", cfg.loss.get("depth_linear_weight", 0.5))
                    ),
                    depth_semantic_log_weight=float(
                        cfg.loss.get("depth_semantic_log_weight", cfg.loss.get("depth_log_weight", 1.0))
                    ),
                    depth_semantic_grad_weight=float(
                        cfg.loss.get("depth_semantic_grad_weight", cfg.loss.get("depth_grad_weight", 1.0))
                    ),
                    depth_gradient_mode=str(
                        cfg.loss.get("depth_gradient_mode", "tolerant_log")
                    ),
                    depth_gradient_tolerance_pixels=int(
                        cfg.loss.get("depth_gradient_tolerance_pixels", 2)
                    ),
                )
                domain_evals: dict[str, dict[str, float]] = {}
                for domain_name, domain_loader in eval_domain_loaders.items():
                    domain_evals[domain_name] = evaluate(
                        model=model, da3=da3, text=text, normalizer=normalizer,
                        pose_normalizer=pose_normalizer,
                        loader=domain_loader, device=device,
                        contexts=sorted({int(value) for value in cfg.model.context_lengths}),
                        rollout=int(cfg.model.rollout_steps),
                        horizon_weights=list(cfg.loss.feature_horizon_weights),
                        feature_patch_weight=float(cfg.loss.feature_patch_weight),
                        feature_cls_weight=float(cfg.loss.feature_cls_weight),
                        feature_register_weight=float(cfg.loss.feature_register_weight),
                        deep_feature_enabled=float(cfg.loss.get("deep_feature_weight", 0.0)) > 0.0,
                        deep_feature_patch_weight=float(cfg.loss.get("deep_feature_patch_weight", 1.0)),
                        deep_feature_cls_weight=float(cfg.loss.get("deep_feature_cls_weight", 0.25)),
                        stop_pos_weight=float(cfg.loss.get("stop_pos_weight", 1.0)),
                        action_direct_weight=float(cfg.loss.get("action_direct_weight", 1.0)),
                        action_refine_weight=float(cfg.loss.get("action_refine_weight", 1.0)),
                        dual_branch_action_aux_weight=float(
                            cfg.loss.get("dual_branch_action_aux_weight", 0.5)
                        ),
                        max_batches=int(cfg.training.eval_max_batches),
                        amp=bool(cfg.training.amp),
                        amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
                        depth_target_source=str(cfg.loss.get("depth_target_source", "ue_gt")),
                depth_scale_mode=str(cfg.loss.get("depth_scale_mode", "per_frame_median")),
                depth_target_mode=str(cfg.loss.get("depth_target_mode", "future")),
                depth_fixed_scale_meters=float(cfg.loss.get("depth_fixed_scale_meters", 100.0)),
                depth_scale_loss_weight=float(cfg.loss.get("depth_scale_loss_weight", 1.0)),
                        depth_linear_weight=float(cfg.loss.get("depth_linear_weight", 0.5)),
                        depth_log_weight=float(cfg.loss.get("depth_log_weight", 1.0)),
                        depth_log_epsilon_normalized=float(
                            cfg.loss.get("depth_log_epsilon_normalized", 1e-6)
                        ),
                        depth_grad_weight=float(cfg.loss.get("depth_grad_weight", 1.0)),
                        depth_semantic_weight=float(cfg.loss.get("depth_semantic_weight", 3.0)),
                        depth_semantic_separate_weight=float(
                            cfg.loss.get("depth_semantic_separate_weight", 0.0)
                        ),
                        depth_semantic_linear_weight=float(
                            cfg.loss.get("depth_semantic_linear_weight", cfg.loss.get("depth_linear_weight", 0.5))
                        ),
                        depth_semantic_log_weight=float(
                            cfg.loss.get("depth_semantic_log_weight", cfg.loss.get("depth_log_weight", 1.0))
                        ),
                        depth_semantic_grad_weight=float(
                            cfg.loss.get("depth_semantic_grad_weight", cfg.loss.get("depth_grad_weight", 1.0))
                        ),
                        depth_gradient_mode=str(
                            cfg.loss.get("depth_gradient_mode", "tolerant_log")
                        ),
                        depth_gradient_tolerance_pixels=int(
                            cfg.loss.get("depth_gradient_tolerance_pixels", 2)
                        ),
                    )
                if rank == 0:
                    line = f"[step={step:07d}] eval " + " ".join(
                        f"{k}={v:.6f}" for k, v in ev.items() if enabled_metric(k, cfg)
                    )
                    print("\n" + line, flush=True)
                    with (output / "train.log").open("a") as handle: handle.write(line + "\n")
                    selection_context = min(
                        int(value) for value in cfg.model.context_lengths
                    )
                    selection_metric = f"H{selection_context}_action"
                    selection_value = float(ev[selection_metric])
                    if selection_value < best_action_value:
                        best_action_value = selection_value
                        atomic_save_checkpoint(
                            checkpoint_payload(step, epoch, logical_batch_idx + 1),
                            best_action_path,
                        )
                        best_action_meta_path.write_text(json.dumps({
                            "metric": selection_metric,
                            "value": selection_value,
                            "step": int(step),
                            "epoch": int(epoch),
                            "checkpoint": str(best_action_path),
                        }, indent=2) + "\n")
                        print(
                            f"[best_action] {selection_metric}={selection_value:.6f} "
                            f"checkpoint={best_action_path}",
                            flush=True,
                        )
                    for domain_name, domain_values in domain_evals.items():
                        short_name = (
                            "sim" if "sim" in domain_name.lower()
                            else "real" if "real" in domain_name.lower()
                            else domain_name
                        )
                        domain_line = (
                            f"[step={step:07d}] eval_{short_name} "
                            + " ".join(
                                f"{short_name}_{key}={value:.5f}"
                                for key, value in domain_values.items()
                            )
                        )
                        print(domain_line, flush=True)
                        with (output / "train.log").open("a") as handle:
                            handle.write(domain_line + "\n")
            if rank == 0 and int(cfg.training.save_every) > 0 and step % int(cfg.training.save_every) == 0:
                save_numbered_checkpoint(step, epoch, logical_batch_idx + 1)
                last_saved_step = step
            save_latest_every = int(cfg.training.get("save_latest_every", 0))
            if (
                rank == 0
                and save_latest_every > 0
                and step % save_latest_every == 0
                and step != last_saved_step
            ):
                atomic_save_checkpoint(
                    checkpoint_payload(step, epoch, logical_batch_idx + 1),
                    output / "last.pt",
                )
            pbar.update(1)
        epoch += 1; start_batch = 0
        save_every_epochs = int(cfg.training.get("save_every_epochs", 0))
        if (
            rank == 0
            and save_every_epochs > 0
            and epoch % save_every_epochs == 0
            and step != last_saved_step
        ):
            save_numbered_checkpoint(step, epoch, 0)
            last_saved_step = step
    pbar.close()
    if profiler is not None:
        profiler.stop()
    # Always persist the exact requested endpoint even when it is not aligned
    # to save_every (for example one physical epoch = 10,886 steps).
    if (
        rank == 0
        and step != last_saved_step
        and not env_truthy("UAVFLOW_SKIP_FINAL_CHECKPOINT")
    ):
        save_numbered_checkpoint(step, epoch, 0)
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
