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

from experiments.uavflow_predictor_idm.data import (
    EpisodePoseNormalizer,
    choose_context_length,
    move_batch,
    temporal_color_augment,
)
from experiments.uavflow_predictor_idm.idm import build_frozen_idm
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.objectives import evaluate, forward_batch
from experiments.uavflow_predictor_idm.runtime import (
    apply_overrides,
    distributed_info,
    lr_scale,
)
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
    if is_dist:
        # ``flex_attention`` is a higher-order operator.  TorchDynamo's DDP
        # graph/bucket optimizer cannot partition graphs containing it and
        # otherwise fails on the first forward with
        # ``DDPOptimizer ... Found a higher order op``.  Disabling only that
        # graph rewrite preserves compiled flex-attention and normal DDP
        # gradient synchronization.
        torch._dynamo.config.optimize_ddp = False
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.seed) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_init_seed)
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
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        raw_model.predictor.load_state_dict(init_ckpt["predictor"], strict=True)
        if "direct_action_head" in init_ckpt:
            raw_model.direct_action_head.load_state_dict(init_ckpt["direct_action_head"], strict=True)
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
    direct_head_ids.update(
        id(parameter) for parameter in raw_model.stop_head.parameters()
        if parameter.requires_grad
    )
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
        if id(p) not in direct_head_ids and id(p) not in backbone_ids
    ]
    direct_head_trainable = [p for p in trainable if id(p) in direct_head_ids]
    backbone_trainable = [p for p in trainable if id(p) in backbone_ids]
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
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(cfg.training.weight_decay)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.training.amp and device.type == "cuda"))
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
            "training.direct_action_head_lr_mult",
            "training.predictor_lr_mult",
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
        pin_memory=True, drop_last=True, persistent_workers=int(cfg.training.num_workers) > 0,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=int(cfg.training.batch_size), sampler=eval_sampler,
        shuffle=False, num_workers=max(1, int(cfg.training.num_workers) // 2),
        pin_memory=True, drop_last=False,
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
            pin_memory=True,
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
                max_batches=int(cfg.training.eval_max_batches),
                amp=bool(cfg.training.amp),
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
            "residual_gate_logit": raw_model.residual_gate_logit.detach().cpu(),
            "missing_action_embed": raw_model.missing_action_embed.detach().cpu(),
            "missing_pose_embed": raw_model.missing_pose_embed.detach().cpu(),
            "reference_step_embed": raw_model.reference_step_embed.detach().cpu(),
            "direct_action_head": raw_model.direct_action_head.state_dict(),
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

    step = start_step; epoch = start_epoch
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
            step_generator = torch.Generator(device=device)
            step_generator.manual_seed(int(cfg.seed) + step * world + rank)
            augmented_sequence = temporal_color_augment(
                torch.cat([
                    batch["episode_first_image"][:, None],
                    batch["all_view_images"],
                ], dim=1),
                OmegaConf.to_container(cfg.augmentation, resolve=True),
                generator=step_generator,
            )
            batch["episode_first_image"] = augmented_sequence[:, 0]
            batch["all_view_images"] = augmented_sequence[:, 1:]
            h = choose_context_length(
                OmegaConf.to_container(cfg.model, resolve=True), device,
                generator=step_generator,
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
                amp=bool(cfg.training.amp),
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
                conditioning_generator=step_generator,
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(trainable, float(cfg.training.grad_clip))
            scaler.step(optimizer); scaler.update()
            if rank == 0 and step % int(cfg.training.log_every) == 0:
                feat_steps = "/".join(
                    f"{float(value):.4f}"
                    for value in values["feature_per_horizon"].detach().float().cpu()
                )
                cos_steps = "/".join(
                    f"{float(value):.4f}"
                    for value in values["feature_cos_per_horizon"].detach().float().cpu()
                )
                memory_gb = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda" else 0.0
                )
                line = (
                    f"[step={step:07d}] total={loss.item():.4f} H={h} "
                    f"action={values['action'].item():.4f} "
                    f"direct={values['direct_action'].item():.4f} "
                    f"refine={values['refine_action'].item():.4f} "
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
                    max_batches=int(cfg.training.eval_max_batches), amp=bool(cfg.training.amp),
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
                        max_batches=int(cfg.training.eval_max_batches),
                        amp=bool(cfg.training.amp),
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
                    line = f"[step={step:07d}] eval " + " ".join(f"{k}={v:.5f}" for k, v in ev.items())
                    print("\n" + line, flush=True)
                    with (output / "train.log").open("a") as handle: handle.write(line + "\n")
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
    # Always persist the exact requested endpoint even when it is not aligned
    # to save_every (for example one physical epoch = 10,886 steps).
    if rank == 0 and step != last_saved_step:
        save_numbered_checkpoint(step, epoch, 0)
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
