"""Canonical ten-cell VLA--GAM matrix.

This module is deliberately data-only so cluster launchers, documentation and
tests all consume one source of truth. Values are OmegaConf dotlist overrides.
"""
from __future__ import annotations


COMMON = (
    "dataset.episode_start_repeat_count=0",
    "dataset.episode_start_train_fraction=0.0",
    "dataset.endpoint_absorbing_window_count=1",
    "dataset.endpoint_absorbing_max_starts=5",
    "dataset.endpoint_absorbing_train_fraction=0.0",
    "dataset.endpoint_self_pair_start_count=0",
    "dataset.endpoint_self_pair_end_count=0",
    "dataset.openvla_prompt_pose_mode=preprocessed",
    "model.action_chunk_size=5",
    "model.rollout_steps=1",
    "model.feature_target_offset=1",
    "model.train_deep_backbone=true",
    "model.deep_train_start_block=13",
    "model.deep_lora.enabled=false",
    "loss.action_weight=3.0",
    "loss.feature_weight=1.0",
    "loss.depth_weight=3.0",
    "loss.depth_linear_weight=1.0",
    "loss.depth_log_weight=0.0",
    "loss.depth_grad_weight=1.0",
    "loss.depth_gradient_mode=gam_l1",
    "loss.depth_semantic_weight=1.0",
)


def _legacy(language: str, pose: bool) -> tuple[str, ...]:
    values = [
        "model.geometry_architecture=legacy",
        "model.parallel_vla_gfm_enabled=false",
        "model.parallel_action_decode_mode=full",
        f"model.use_pose_history={'true' if pose else 'false'}",
        "model.use_action_history=false",
        "loss.depth_target_mode=future",
        "loss.action_direct_weight=0.0",
        "loss.action_refine_weight=1.0",
    ]
    if language == "t5":
        values += ["stage1.text_encoder_type=t5", "model.language_len=77"]
    else:
        values += [
            "stage1.text_encoder_type=qwen3_5",
            "stage1.qwen_prompt_mode=current_image_instruction",
            "stage1.qwen_token_selection=text_after_image",
            "stage1.qwen_lora_enabled=false",
        ]
    return tuple(values)


def _vla(mode: str, *, current_depth: bool, current_read: bool,
         residual: bool = False) -> tuple[str, ...]:
    placeholders = mode == "qwen_tokens"
    return (
        "stage1.text_encoder_type=qwen3_5",
        "stage1.qwen_layers=[23]",
        "stage1.qwen_use_reference_image=false",
        "stage1.qwen_prompt_mode=current_image_action_question",
        f"stage1.qwen_token_selection={'action_placeholders' if placeholders else 'text_after_image'}",
        f"stage1.qwen_action_placeholder_count={5 if placeholders else 0}",
        "stage1.qwen_lora_enabled=true",
        "stage1.qwen_lora_rank=32",
        "stage1.qwen_lora_alpha=16",
        "stage1.qwen_lora_dropout=0.0",
        "model.geometry_architecture=dual_vla_gfm",
        "model.parallel_vla_gfm_enabled=true",
        f"model.parallel_vla_gfm_mode={mode}",
        "model.parallel_vla_gfm_width=512",
        "model.parallel_vla_gfm_heads=8",
        "model.parallel_action_post_bidir_layers=0",
        f"model.parallel_current_depth_enabled={'true' if current_depth else 'false'}",
        f"model.parallel_current_geometry_read_enabled={'true' if current_read else 'false'}",
        "model.current_geometry_bank_mode=output_current",
        f"model.parallel_action_decode_mode={'geometry_residual' if residual else 'full'}",
        "model.use_pose_history=false",
        "model.use_action_history=false",
        "model.direct_action_enabled=true",
        "model.compute_idm_branch=false",
        "model.deep_action_enabled=true",
        f"loss.depth_target_mode={'both' if current_depth else 'future'}",
        f"loss.action_direct_weight={1.0 / 3.0 if current_read else 0.0}",
        f"loss.action_refine_weight={2.0 / 3.0 if current_read else 1.0}",
    )


EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "G0": _legacy("t5", False),
    "G1": _legacy("t5", True),
    "C0": _legacy("qwen", False),
    "C1": _legacy("qwen", True),
    "Q0": _vla("external_query", current_depth=True, current_read=True),
    "S0": _vla("qwen_tokens", current_depth=False, current_read=False),
    "S1": _vla("qwen_tokens", current_depth=True, current_read=False),
    "S2": _vla("qwen_tokens", current_depth=True, current_read=True),
    "R1": _vla(
        "qwen_tokens", current_depth=True, current_read=True, residual=True
    ),
    "DV": (
        "stage1.text_encoder_type=qwen3_5",
        "stage1.qwen_layers=[23]",
        "stage1.qwen_use_reference_image=false",
        "stage1.qwen_prompt_mode=current_image_action_question",
        "stage1.qwen_token_selection=text_after_image",
        "stage1.qwen_action_placeholder_count=0",
        "stage1.qwen_lora_enabled=true",
        "stage1.qwen_lora_rank=32",
        "stage1.qwen_lora_alpha=16",
        "model.geometry_architecture=dual_observed",
        "model.parallel_vla_gfm_enabled=false",
        "model.parallel_action_decode_mode=full",
        "model.use_pose_history=false",
        "model.use_action_history=false",
        "model.direct_action_enabled=true",
        "model.compute_idm_branch=false",
        "model.deep_action_enabled=true",
        "loss.depth_target_mode=both",
        "loss.action_direct_weight=0.0",
        "loss.action_refine_weight=1.0",
        "loss.dual_branch_action_aux_weight=0.5",
    ),
}


def overrides(experiment_id: str) -> tuple[str, ...]:
    try:
        return COMMON + EXPERIMENTS[experiment_id.upper()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment {experiment_id!r}; choose {','.join(EXPERIMENTS)}"
        ) from exc
