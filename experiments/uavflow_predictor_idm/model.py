"""Composable one-step predictor -> DA3-deep -> sliding-IDM UAV policy.

The implementation intentionally reuses:
  * GAMFuturePredictor for block-causal attention and 4-D axial RoPE;
  * DA3GiantEncoder for the shallow boundary and frozen deeper backbone;
  * DA3HierarchicalWindowInversePerceiver trained by the Stage-1 IDM script.
"""

from __future__ import annotations

import torch
from torch import nn

from robot.modeling.action_head_v2 import ActionHeadV2
from robot.modeling.action_head_causal_token import CausalTokenActionHead
from robot.modeling.future_predictor import GAMFuturePredictor
from .idm import build_frozen_idm  # Backward-compatible re-export.
from .current_geometry import CurrentGeometryRead
from .semantic_geometry_action import (
    OFTInternalActionProjector,
    ParallelContinuousActionHead,
    QwenInternalActionProjector,
    SemanticActionInitializer,
)
from .geometry_architectures import (
    GEOMETRY_ARCHITECTURES, DUAL_ARCHITECTURES, DirectCurrentActionSeed,
    DualActionFusion, select_dual_view,
)


class PatchMotionStopHead(nn.Module):
    """Predict Stop from unpooled F0/current patches, pose and language.

    Current visual tokens first cross-attend to the fixed-reference tokens so
    the branch can represent patch-level motion/change.  A learned Stop query
    then attends to every fused motion token, the current episode-relative pose
    token and every valid language token.  In particular, there is no spatial
    mean/max pooling before the Stop decision.
    """

    def __init__(
        self,
        *,
        visual_dim: int,
        language_dim: int,
        pose_dim: int = 5,
        model_dim: int = 512,
        num_heads: int = 8,
        visual_prefix_tokens: int = 0,
        include_plan_tokens: bool = False,
        initial_stop_probability: float | None = None,
    ) -> None:
        super().__init__()
        self.visual_prefix_tokens = int(visual_prefix_tokens)
        self.include_plan_tokens = bool(include_plan_tokens)
        if model_dim % num_heads:
            raise ValueError(f"stop model_dim={model_dim} must divide num_heads={num_heads}.")
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, model_dim),
        )
        self.language_proj = nn.Sequential(
            nn.LayerNorm(language_dim), nn.Linear(language_dim, model_dim),
        )
        if self.include_plan_tokens:
            # Keep action and predicted-future projections separate: they have
            # different semantics even though both originate at DA3 width.
            self.action_proj = nn.Sequential(
                nn.LayerNorm(visual_dim), nn.Linear(visual_dim, model_dim),
            )
            self.future_proj = nn.Sequential(
                nn.LayerNorm(visual_dim), nn.Linear(visual_dim, model_dim),
            )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_dim), nn.Linear(pose_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.reference_type = nn.Parameter(torch.empty(model_dim))
        self.current_type = nn.Parameter(torch.empty(model_dim))
        self.pose_type = nn.Parameter(torch.empty(model_dim))
        self.language_type = nn.Parameter(torch.empty(model_dim))
        if self.include_plan_tokens:
            self.action_type = nn.Parameter(torch.empty(model_dim))
            self.future_type = nn.Parameter(torch.empty(model_dim))
        self.stop_query = nn.Parameter(torch.empty(1, 1, model_dim))
        typed_parameters = [
            self.reference_type, self.current_type, self.pose_type,
            self.language_type, self.stop_query,
        ]
        if self.include_plan_tokens:
            typed_parameters.extend([self.action_type, self.future_type])
        for parameter in typed_parameters:
            nn.init.normal_(parameter, std=0.02)
        self.current_to_reference = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.0, batch_first=True,
        )
        self.motion_fuse = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.stop_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.0, batch_first=True,
        )
        self.stop_norm = nn.LayerNorm(model_dim)
        self.stop_ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim), nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.output = nn.Linear(model_dim, 1)
        # A tiny non-zero initialization preserves the conservative initial
        # Stop probability while allowing patch/language/pose layers to receive
        # gradients on the very first optimizer step.
        nn.init.normal_(self.output.weight, std=1e-3)
        if initial_stop_probability is None:
            initial_bias = -3.0
        else:
            probability = min(max(float(initial_stop_probability), 1e-4), 1.0 - 1e-4)
            initial_bias = float(torch.logit(torch.tensor(probability)).item())
        nn.init.constant_(self.output.bias, initial_bias)

    def forward(
        self,
        *,
        reference_visual: torch.Tensor,
        current_visual: torch.Tensor,
        current_pose: torch.Tensor | None,
        language: torch.Tensor,
        language_padding_mask: torch.Tensor | None,
        predicted_future: torch.Tensor | None = None,
        deep_action_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # visual: (B,1,V,N,D) reference and (B,H,V,N,D) current
        b, h, views, tokens, _ = current_visual.shape
        if self.visual_prefix_tokens:
            if tokens <= self.visual_prefix_tokens:
                raise ValueError(
                    f"Visual sequence has {tokens} tokens but Stop must drop "
                    f"{self.visual_prefix_tokens} CLS/register tokens."
                )
            reference_visual = reference_visual[:, :, :, self.visual_prefix_tokens:]
            current_visual = current_visual[:, :, :, self.visual_prefix_tokens:]
            tokens -= self.visual_prefix_tokens
        ref = reference_visual.expand(-1, h, -1, -1, -1).reshape(b * h, views * tokens, -1)
        cur = current_visual.reshape(b * h, views * tokens, -1)
        ref = self.visual_proj(ref) + self.reference_type
        cur = self.visual_proj(cur) + self.current_type
        aligned_ref, _ = self.current_to_reference(cur, ref, ref, need_weights=False)
        motion = self.motion_fuse(torch.cat([cur, aligned_ref, cur - aligned_ref], dim=-1))

        if current_pose is None:
            # A learned modality/type token still marks this memory slot, but
            # no ground-truth pose value is exposed to the Stop decoder.
            pose_value = current_visual.new_zeros(b * h, self.pose_proj.in_features)
        else:
            pose_value = current_pose.reshape(b * h, -1)
        pose = self.pose_proj(pose_value).unsqueeze(1) + self.pose_type
        lang = self.language_proj(language)
        lang = lang[:, None].expand(-1, h, -1, -1).reshape(b * h, lang.shape[1], -1)
        lang = lang + self.language_type
        memory_parts = [motion]
        if self.include_plan_tokens:
            if predicted_future is None or deep_action_tokens is None:
                raise RuntimeError(
                    "Hybrid Stop requires predicted_future and deep_action_tokens."
                )
            future = predicted_future.reshape(b * h, views * predicted_future.shape[3], -1)
            action = deep_action_tokens.reshape(b * h, views, -1)
            memory_parts.extend([
                self.future_proj(future) + self.future_type,
                self.action_proj(action) + self.action_type,
            ])
        memory_parts.extend([pose, lang])
        memory = torch.cat(memory_parts, dim=1)

        key_padding_mask = None
        if language_padding_mask is not None:
            lang_invalid = ~language_padding_mask.to(device=memory.device, dtype=torch.bool)
            lang_invalid = lang_invalid[:, None].expand(-1, h, -1).reshape(b * h, -1)
            key_padding_mask = torch.cat([
                torch.zeros(
                    b * h, sum(part.shape[1] for part in memory_parts[:-1]),
                    device=memory.device, dtype=torch.bool,
                ),
                lang_invalid,
            ], dim=1)
        query = self.stop_query.expand(b * h, -1, -1)
        attended, _ = self.stop_attention(
            query, memory, memory, key_padding_mask=key_padding_mask, need_weights=False,
        )
        hidden = self.stop_norm(query + attended)
        hidden = self.stop_norm(hidden + self.stop_ffn(hidden))
        return self.output(hidden).reshape(b, h)


class HybridPatchActionStopHead(nn.Module):
    """Stop decoder with local motion, global progress and policy intent.

    The old action-token and patch-motion heads remain unchanged for exact
    checkpoint compatibility.  This head is used only by the new hybrid mode:

    * F0/current patch correspondence describes observed local motion;
    * current/predicted-future correspondence describes planned local motion;
    * F0/current/future CLS tokens preserve global scene/progress semantics;
    * the DA3-deep action token exposes the policy's intended control;
    * pose and every valid language token remain directly readable.

    Predicted-future evidence is gated from a conservative initial value
    because it is learned rather than observed.  The non-zero gate still lets
    feature/action losses train that path from the first optimizer step.
    """

    def __init__(
        self,
        *,
        visual_dim: int,
        language_dim: int,
        pose_dim: int = 5,
        model_dim: int = 512,
        num_heads: int = 8,
        num_queries: int = 4,
        decoder_layers: int = 2,
        visual_prefix_tokens: int = 0,
        future_gate_init: float = 0.1,
        initial_stop_probability: float = 0.192,
    ) -> None:
        super().__init__()
        self.visual_prefix_tokens = int(visual_prefix_tokens)
        if self.visual_prefix_tokens < 1:
            raise ValueError("Hybrid Stop requires the DA3 CLS token in its visual prefix.")
        if model_dim % num_heads:
            raise ValueError(f"stop model_dim={model_dim} must divide num_heads={num_heads}.")
        if int(num_queries) <= 0 or int(decoder_layers) <= 0:
            raise ValueError("Hybrid Stop num_queries and decoder_layers must be positive.")
        self.num_queries = int(num_queries)
        self.decoder_layers_count = int(decoder_layers)
        if not 0.0 < float(future_gate_init) < 1.0:
            raise ValueError("future_gate_init must be strictly between 0 and 1.")

        def projection(input_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim))

        self.patch_proj = projection(visual_dim)
        self.future_patch_proj = projection(visual_dim)
        self.cls_proj = projection(visual_dim)
        self.action_proj = projection(visual_dim)
        self.language_proj = projection(language_dim)
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_dim), nn.Linear(pose_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        type_names = (
            "reference_patch_type", "current_patch_type", "future_patch_type",
            "reference_cls_type", "current_cls_type", "future_cls_type",
            "observed_progress_type", "future_progress_type", "action_type",
            "pose_type", "language_type",
        )
        for name in type_names:
            parameter = nn.Parameter(torch.empty(model_dim))
            nn.init.normal_(parameter, std=0.02)
            setattr(self, name, parameter)
        # Multiple latent slots can specialize in goal, observed progress,
        # predicted progress and action intent. Their roles are learned rather
        # than hard-wired, avoiding brittle modality-specific supervision.
        self.stop_queries = nn.Parameter(torch.empty(1, self.num_queries, model_dim))
        self.decision_query = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.stop_queries, std=0.02)
        nn.init.normal_(self.decision_query, std=0.02)

        self.current_to_reference = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.0, batch_first=True,
        )
        self.future_to_current = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.0, batch_first=True,
        )
        self.observed_motion_fuse = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.future_motion_fuse = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.observed_progress_fuse = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.future_progress_fuse = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.future_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(future_gate_init)))
        )
        self.stop_decoder = nn.ModuleList()
        for _ in range(self.decoder_layers_count):
            self.stop_decoder.append(nn.ModuleDict({
                "cross_attention": nn.MultiheadAttention(
                    model_dim, num_heads, dropout=0.0, batch_first=True,
                ),
                "cross_norm": nn.LayerNorm(model_dim),
                "self_attention": nn.MultiheadAttention(
                    model_dim, num_heads, dropout=0.0, batch_first=True,
                ),
                "self_norm": nn.LayerNorm(model_dim),
                "ffn": nn.Sequential(
                    nn.Linear(model_dim, 4 * model_dim), nn.SiLU(),
                    nn.Linear(4 * model_dim, model_dim),
                ),
                "ffn_norm": nn.LayerNorm(model_dim),
            }))
        self.decision_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=0.0, batch_first=True,
        )
        self.decision_cross_norm = nn.LayerNorm(model_dim)
        self.decision_ffn = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim), nn.SiLU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.decision_ffn_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        probability = min(max(float(initial_stop_probability), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(
            self.output.bias, float(torch.logit(torch.tensor(probability)).item())
        )

    @property
    def future_gate(self) -> torch.Tensor:
        return self.future_gate_logit.sigmoid()

    def forward(
        self,
        *,
        reference_visual: torch.Tensor,
        current_visual: torch.Tensor,
        current_pose: torch.Tensor,
        language: torch.Tensor,
        language_padding_mask: torch.Tensor | None,
        predicted_future: torch.Tensor,
        deep_action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # visual tensors: reference=(B,1,V,N,D), current/future=(B,H,V,N,D)
        b, h, views, tokens, _ = current_visual.shape
        if reference_visual.shape[1] != 1:
            raise ValueError("Hybrid Stop expects one fixed F0 reference timestep.")
        if predicted_future.shape[:4] != current_visual.shape[:4]:
            raise ValueError(
                "Hybrid Stop future/current shapes disagree: "
                f"future={tuple(predicted_future.shape)}, current={tuple(current_visual.shape)}."
            )
        if tokens <= self.visual_prefix_tokens:
            raise ValueError(
                f"Visual sequence has {tokens} tokens but prefix has "
                f"{self.visual_prefix_tokens}."
            )
        patch_count = tokens - self.visual_prefix_tokens
        ref_full = reference_visual.expand(-1, h, -1, -1, -1)

        ref_patch = ref_full[:, :, :, self.visual_prefix_tokens:].reshape(
            b * h, views * patch_count, -1
        )
        cur_patch = current_visual[:, :, :, self.visual_prefix_tokens:].reshape(
            b * h, views * patch_count, -1
        )
        fut_patch = predicted_future[:, :, :, self.visual_prefix_tokens:].reshape(
            b * h, views * patch_count, -1
        )
        ref_patch = self.patch_proj(ref_patch) + self.reference_patch_type
        cur_patch = self.patch_proj(cur_patch) + self.current_patch_type
        fut_patch = self.future_patch_proj(fut_patch) + self.future_patch_type

        aligned_ref, _ = self.current_to_reference(
            cur_patch, ref_patch, ref_patch, need_weights=False,
        )
        observed_motion = self.observed_motion_fuse(
            torch.cat([cur_patch, aligned_ref, cur_patch - aligned_ref], dim=-1)
        )
        aligned_current, _ = self.future_to_current(
            fut_patch, cur_patch, cur_patch, need_weights=False,
        )
        future_motion = self.future_motion_fuse(
            torch.cat([fut_patch, aligned_current, fut_patch - aligned_current], dim=-1)
        )

        # CLS tokens are never pooled together with patches.  Each view keeps
        # its own global token; the learned Stop query performs the final read.
        ref_cls = self.cls_proj(ref_full[:, :, :, 0].reshape(b * h, views, -1))
        cur_cls = self.cls_proj(current_visual[:, :, :, 0].reshape(b * h, views, -1))
        fut_cls = self.cls_proj(predicted_future[:, :, :, 0].reshape(b * h, views, -1))
        observed_progress = self.observed_progress_fuse(
            torch.cat([ref_cls, cur_cls, cur_cls - ref_cls], dim=-1)
        ) + self.observed_progress_type
        future_progress = self.future_progress_fuse(
            torch.cat([cur_cls, fut_cls, fut_cls - cur_cls], dim=-1)
        ) + self.future_progress_type

        gate = self.future_gate.to(dtype=future_motion.dtype)
        action = self.action_proj(deep_action_tokens.reshape(b * h, views, -1))
        action = action + self.action_type
        pose = self.pose_proj(current_pose.reshape(b * h, -1)).unsqueeze(1) + self.pose_type
        lang = self.language_proj(language)
        lang = lang[:, None].expand(-1, h, -1, -1).reshape(b * h, lang.shape[1], -1)
        lang = lang + self.language_type

        memory_parts = [
            observed_motion,
            gate * future_motion,
            ref_cls + self.reference_cls_type,
            cur_cls + self.current_cls_type,
            gate * (fut_cls + self.future_cls_type),
            observed_progress,
            gate * future_progress,
            action,
            pose,
            lang,
        ]
        memory = torch.cat(memory_parts, dim=1)
        key_padding_mask = None
        if language_padding_mask is not None:
            lang_invalid = ~language_padding_mask.to(device=memory.device, dtype=torch.bool)
            lang_invalid = lang_invalid[:, None].expand(-1, h, -1).reshape(b * h, -1)
            key_padding_mask = torch.cat([
                torch.zeros(
                    b * h, sum(part.shape[1] for part in memory_parts[:-1]),
                    device=memory.device, dtype=torch.bool,
                ),
                lang_invalid,
            ], dim=1)
        queries = self.stop_queries.expand(b * h, -1, -1)
        for layer in self.stop_decoder:
            attended, _ = layer["cross_attention"](
                queries, memory, memory,
                key_padding_mask=key_padding_mask, need_weights=False,
            )
            queries = layer["cross_norm"](queries + attended)
            communicated, _ = layer["self_attention"](
                queries, queries, queries, need_weights=False,
            )
            queries = layer["self_norm"](queries + communicated)
            queries = layer["ffn_norm"](queries + layer["ffn"](queries))

        decision = self.decision_query.expand(b * h, -1, -1)
        pooled, _ = self.decision_attention(
            decision, queries, queries, need_weights=False,
        )
        hidden = self.decision_cross_norm(decision + pooled)
        hidden = self.decision_ffn_norm(hidden + self.decision_ffn(hidden))
        return self.output(hidden).reshape(b, h)


class ActionHiddenPoseStopHead(nn.Module):
    """CLIP-style Stop readout from refined action state and stop-only pose."""

    def __init__(self, action_dim: int, model_dim: int, pose_dim: int = 5) -> None:
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, model_dim), nn.GELU(),
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim),
            nn.GELU(), nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.output[-1].weight, std=1e-3)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, action_hidden: torch.Tensor, stop_pose: torch.Tensor) -> torch.Tensor:
        expected = (*action_hidden.shape[:-1], 5)
        if stop_pose.shape != expected:
            raise ValueError(
                f"Action-hidden Stop requires normalized pose {expected}, "
                f"got {tuple(stop_pose.shape)}."
            )
        hidden = self.action_proj(action_hidden)
        hidden = hidden + self.pose_proj(stop_pose.to(hidden.dtype))
        return self.output(hidden).squeeze(-1)


class UAVFlowPredictorIDM(nn.Module):
    """Predict one future shallow frame and decode its transition with an IDM.

    The predictor sees only real observed shallow tokens and already-executed
    actions. Ground-truth future tokens are used solely as detached loss
    targets. The frozen four-frame IDM receives the three most recent observed
    frames plus the one predicted next frame. At startup, missing history is
    left-padded by repeating the first observation::

        H=1: [F0, F0, F0, pred F1]
        H=2: [F0, F0, F1, pred F2]
        H>=3: [F{t-2}, F{t-1}, Ft, pred F{t+1}]

    The IDM predicts the three adjacent transitions in that window; only its
    final output is the current control action. After execution, deployment
    replaces the predicted frame with the real observation before replanning.
    """

    def __init__(
        self,
        *,
        da3: nn.Module,
        idm: nn.Module | None,
        action_dim: int,
        action_chunk_size: int = 1,
        feature_target_offset: int = 1,
        rollout_steps: int = 1,
        d_model: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        language_dim: int = 768,
        language_len: int = 77,
        variable_language_tokens: bool = False,
        condition_mode: str = "cross_attn",
        use_action_history: bool = False,
        use_pose_history: bool = False,
        action_history_keep_prob: float = 1.0,
        pose_history_keep_prob: float = 1.0,
        pose_history_xyz_noise_meters: float = 0.0,
        pose_history_yaw_noise_degrees: float = 0.0,
        use_fixed_first_frame: bool = False,
        use_reference_type_embedding: bool = False,
        dense_context_supervision: bool = False,
        direct_action_enabled: bool = False,
        compute_idm_branch: bool = True,
        deep_action_enabled: bool = False,
        train_deep_backbone: bool = False,
        deep_train_start_block: int = 19,
        depth_decode_enabled: bool = False,
        depth_scale_head_enabled: bool = False,
        depth_scale_init_meters: float = 100.0,
        relative_pose_head_enabled: bool = False,
        stop_head_enabled: bool = False,
        stop_head_mode: str = "legacy_action_token",
        stop_model_dim: int = 512,
        stop_num_heads: int = 8,
        stop_num_queries: int = 4,
        stop_decoder_layers: int = 2,
        stop_future_gate_init: float = 0.1,
        residual_prediction: bool = True,
        residual_gate_init: float = 0.10,
        gradient_checkpointing: bool = True,
        current_geometry_action_enabled: bool = False,
        current_geometry_read_mode: str = "terminal",
        current_geometry_bank_mode: str = "output_current",
        geometry_architecture: str = "legacy",
        vlm_action_seed_enabled: bool = False,
        causal_action_decoder_enabled: bool = False,
        causal_action_bins: int = 256,
        causal_action_model_dim: int = 512,
        causal_action_num_heads: int = 8,
        causal_action_num_layers: int = 2,
        parallel_vla_gfm_enabled: bool = False,
        parallel_vla_gfm_width: int = 512,
        parallel_vla_gfm_heads: int = 8,
        parallel_vla_gfm_mode: str = "external_query",
        parallel_action_post_bidir_layers: int = 0,
        parallel_current_depth_enabled: bool = True,
        parallel_current_geometry_read_enabled: bool = True,
        parallel_action_decode_mode: str = "full",
    ) -> None:
        super().__init__()
        self.da3 = da3
        self.idm = idm
        self.rollout_steps = int(rollout_steps)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        self.feature_target_offset = int(feature_target_offset)
        if self.action_chunk_size <= 0 or self.feature_target_offset <= 0:
            raise ValueError(
                "action_chunk_size and feature_target_offset must be positive, got "
                f"{self.action_chunk_size} and {self.feature_target_offset}."
            )
        self.deep_gradient_checkpointing = bool(gradient_checkpointing)
        self.use_action_history = bool(use_action_history)
        self.use_pose_history = bool(use_pose_history)
        self.action_history_keep_prob = float(action_history_keep_prob)
        self.pose_history_keep_prob = float(pose_history_keep_prob)
        self.pose_history_xyz_noise_meters = float(pose_history_xyz_noise_meters)
        self.pose_history_yaw_noise_degrees = float(pose_history_yaw_noise_degrees)
        if not 0.0 <= self.action_history_keep_prob <= 1.0:
            raise ValueError("action_history_keep_prob must be in [0,1].")
        if not 0.0 <= self.pose_history_keep_prob <= 1.0:
            raise ValueError("pose_history_keep_prob must be in [0,1].")
        if self.pose_history_xyz_noise_meters < 0.0 or self.pose_history_yaw_noise_degrees < 0.0:
            raise ValueError("Pose noise standard deviations must be non-negative.")
        self.use_fixed_first_frame = bool(use_fixed_first_frame)
        self.use_reference_type_embedding = bool(use_reference_type_embedding)
        if self.use_reference_type_embedding and not self.use_fixed_first_frame:
            raise ValueError("Reference type embedding requires use_fixed_first_frame=true.")
        self.dense_context_supervision = bool(dense_context_supervision)
        self.direct_action_enabled = bool(direct_action_enabled)
        self.compute_idm_branch = bool(compute_idm_branch)
        self.deep_action_enabled = bool(deep_action_enabled)
        self.current_geometry_action_enabled = bool(current_geometry_action_enabled)
        self.current_geometry_read_mode = str(current_geometry_read_mode)
        if self.current_geometry_read_mode not in {"terminal", "per_layer"}:
            raise ValueError("current_geometry_read_mode must be terminal or per_layer")
        self.current_geometry_bank_mode = str(current_geometry_bank_mode).lower()
        valid_bank_modes = {
            "every_layer_concat", "output_current", "output_concat", "global_current",
        }
        if self.current_geometry_bank_mode not in valid_bank_modes:
            raise ValueError(
                f"current_geometry_bank_mode={self.current_geometry_bank_mode!r} not in "
                f"{sorted(valid_bank_modes)}"
            )
        self.geometry_architecture = str(geometry_architecture)
        self.vlm_action_seed_enabled = bool(vlm_action_seed_enabled)
        self.causal_action_decoder_enabled = bool(causal_action_decoder_enabled)
        self.parallel_vla_gfm_enabled = bool(parallel_vla_gfm_enabled)
        self.parallel_vla_gfm_mode = str(parallel_vla_gfm_mode).lower()
        self.parallel_action_post_bidir_layers = int(parallel_action_post_bidir_layers)
        self.parallel_current_depth_enabled = bool(parallel_current_depth_enabled)
        self.parallel_current_geometry_read_enabled = bool(
            parallel_current_geometry_read_enabled
        )
        self.parallel_action_decode_mode = str(parallel_action_decode_mode).lower()
        if self.parallel_action_decode_mode not in {"full", "geometry_residual"}:
            raise ValueError(
                "parallel_action_decode_mode must be full or geometry_residual"
            )
        if self.parallel_action_post_bidir_layers < 0:
            raise ValueError("parallel_action_post_bidir_layers must be non-negative")
        valid_parallel_modes = {"external_query", "qwen_tokens", "oft_gfm", "oft_direct"}
        if self.parallel_vla_gfm_enabled:
            if self.geometry_architecture != "dual_vla_gfm":
                raise ValueError(
                    "parallel_vla_gfm_enabled=true requires geometry_architecture=dual_vla_gfm"
                )
            if self.vlm_action_seed_enabled or self.causal_action_decoder_enabled:
                raise ValueError(
                    "dual_vla_gfm replaces the pooled VLM seed and causal action decoder"
                )
            if self.rollout_steps != 1 or self.action_chunk_size <= 1:
                raise ValueError("dual_vla_gfm requires rollout_steps=1 and chunk_size>1")
            if self.parallel_vla_gfm_mode not in valid_parallel_modes:
                raise ValueError(
                    f"parallel_vla_gfm_mode={self.parallel_vla_gfm_mode!r} not in "
                    f"{sorted(valid_parallel_modes)}"
                )
            # Geometry Bank is always consumed inside every DA3 refine layer,
            # never as the legacy one-shot terminal read.
            self.current_geometry_read_mode = "per_layer"
        elif self.parallel_current_depth_enabled or self.parallel_current_geometry_read_enabled:
            # These switches only describe the parallel VLA-GFM family.
            self.parallel_current_depth_enabled = False
            self.parallel_current_geometry_read_enabled = False
        if self.geometry_architecture not in GEOMETRY_ARCHITECTURES:
            raise ValueError(f"Unknown geometry_architecture={self.geometry_architecture}")
        if self.geometry_architecture != "legacy":
            if not self.deep_action_enabled or self.rollout_steps != 1 or self.compute_idm_branch:
                raise ValueError("Geometry architectures require deep action, rollout=1, no legacy IDM")
            if self.current_geometry_action_enabled:
                raise ValueError("Do not combine joint geometry architectures with the terminal CA control")
        if self.current_geometry_action_enabled and (not self.deep_action_enabled or self.rollout_steps != 1):
            raise ValueError("Current geometry action read requires deep_action_enabled and rollout_steps=1")
        self.train_deep_backbone = bool(train_deep_backbone)
        self.deep_train_start_block = int(deep_train_start_block)
        self.depth_decode_enabled = bool(depth_decode_enabled)
        self.depth_scale_head_enabled = bool(depth_scale_head_enabled)
        self.relative_pose_head_enabled = bool(relative_pose_head_enabled)
        self.stop_head_enabled = bool(stop_head_enabled)
        self.stop_head_mode = str(stop_head_mode)
        if (self.deep_action_enabled or self.stop_head_enabled) and not self.direct_action_enabled:
            raise ValueError("Deep action/stop heads require direct_action_enabled=true.")
        if self.dense_context_supervision and self.rollout_steps != 1:
            raise ValueError("Dense context supervision requires rollout_steps=1.")
        if self.dense_context_supervision and self.compute_idm_branch:
            raise ValueError(
                "Dense context supervision currently supports the direct action branch only; "
                "set compute_idm_branch=false."
            )
        self.residual_prediction = bool(residual_prediction)
        if self.compute_idm_branch:
            if idm is None:
                raise ValueError("compute_idm_branch=true requires a frozen IDM.")
            if int(getattr(idm, "window_size")) < 2:
                raise ValueError(f"IDM window_size must be >=2, got {getattr(idm, 'window_size')}.")
            if self.rollout_steps > 1 and int(getattr(idm, "window_size")) != self.rollout_steps + 1:
                raise ValueError(
                    "Legacy multi-step evaluation requires IDM window_size=rollout_steps+1; "
                    f"got window={getattr(idm, 'window_size')}, rollout={self.rollout_steps}."
                )
        n_visual = 1 + int(getattr(da3, "num_register_tokens", 0)) + 256
        self.predictor = GAMFuturePredictor(
            d_da3=int(da3.embed_dim),
            d_model=int(d_model),
            depth=int(depth),
            num_heads=int(num_heads),
            ffn_ratio=float(ffn_ratio),
            dropout=float(dropout),
            num_patches_per_view=n_visual - 1 - int(getattr(da3, "num_register_tokens", 0)),
            num_register_tokens=int(getattr(da3, "num_register_tokens", 0)),
            # Parallel VLA-GFM carries semantics exclusively in its K action
            # slots. Keeping the old per-block language CA would duplicate
            # conditioning and invalidate the intended ablation.
            use_language=not (
                self.vlm_action_seed_enabled or self.parallel_vla_gfm_enabled
            ),
            language_dim=int(language_dim),
            language_len=int(language_len),
            variable_language_tokens=bool(variable_language_tokens),
            # Keep the historical unused 6D projector shape for old
            # checkpoints; pose-conditioned runs use the new 5D state.
            proprio_dim=5 if self.use_pose_history else 6,
            action_dim=int(action_dim),
            action_chunk_size=1,
            use_proprio_input=self.use_pose_history,
            condition_mode=str(condition_mode),
            input_proj_norm="ln",
            gradient_checkpointing=bool(gradient_checkpointing),
            num_action_slots=(self.action_chunk_size if self.parallel_vla_gfm_enabled else 1),
        )
        if self.vlm_action_seed_enabled:
            # Select one already multimodally contextualized Qwen token, then
            # initialize only GAM's current action slot. Language is therefore
            # consumed once at the entrance and is absent from all Predictor
            # block cross-attention paths.
            self.vlm_action_seed = nn.Sequential(
                nn.LayerNorm(int(language_dim)),
                nn.Linear(int(language_dim), int(d_model)),
            )
        else:
            self.vlm_action_seed = None
        # Dedicated content embedding for a missing/not-yet-executed action.
        # GAM's existing type_embed[4] is still added inside the predictor, so
        # this parameter represents content only and cannot be confused with a
        # real action that happens to normalize to the all-zero vector.
        self.missing_action_embed = nn.Parameter(
            torch.empty(int(d_model)), requires_grad=self.use_action_history
        )
        nn.init.normal_(self.missing_action_embed, std=0.02)
        # A numeric zero is a valid episode-relative pose at F0. Conditioning
        # dropout therefore uses a learned missing-pose content embedding.
        self.missing_pose_embed = nn.Parameter(
            torch.empty(int(d_model)), requires_grad=self.use_pose_history
        )
        nn.init.normal_(self.missing_pose_embed, std=0.02)
        # Explicitly mark the complete prepended F0 timestep. Kept outside the
        # shared GAM predictor state so legacy checkpoints remain loadable.
        self.reference_step_embed = nn.Parameter(
            torch.empty(int(d_model)), requires_grad=self.use_reference_type_embedding
        )
        nn.init.normal_(self.reference_step_embed, std=0.02)
        # Exact GAM direct branch: the predictor's dedicated action-slot hidden
        # is normalized/projected to DA3 width by GAMFuturePredictor, then a
        # standard ActionHeadV2 regresses a continuous 4-DoF action.
        self.direct_action_head = ActionHeadV2(
            input_dim=int(da3.embed_dim),
            n_views=1,
            hidden_dim=int(da3.embed_dim),
            n_dims=int(action_dim),
            chunk_size=self.action_chunk_size,
            num_blocks=2,
            pool_mode="mean",
            chunk_position_encoding=("learned" if self.action_chunk_size > 1 else "none"),
        )
        for parameter in self.direct_action_head.parameters():
            parameter.requires_grad = (
                self.direct_action_enabled
                and not self.causal_action_decoder_enabled
                and not self.parallel_vla_gfm_enabled
            )
        self.dual_action_fusion = (
            DualActionFusion(int(da3.embed_dim))
            if self.geometry_architecture in DUAL_ARCHITECTURES else None
        )
        self.causal_action_decoder = (
            CausalTokenActionHead(
                input_dim=int(da3.embed_dim),
                action_dim=int(action_dim),
                chunk_size=self.action_chunk_size,
                n_bins=int(causal_action_bins),
                model_dim=int(causal_action_model_dim),
                num_heads=int(causal_action_num_heads),
                num_layers=int(causal_action_num_layers),
                dropout=float(dropout),
            )
            if self.causal_action_decoder_enabled else None
        )
        self.semantic_geometry_action = None
        if self.parallel_vla_gfm_enabled:
            initializer_cls = (
                QwenInternalActionProjector
                if self.parallel_vla_gfm_mode == "qwen_tokens"
                else SemanticActionInitializer
            )
            initializer_kwargs = dict(
                language_dim=int(language_dim),
                output_dim=int(d_model),
                chunk_size=self.action_chunk_size,
                width=int(parallel_vla_gfm_width),
                heads=int(parallel_vla_gfm_heads),
            )
            if self.parallel_vla_gfm_mode == "qwen_tokens":
                initializer_kwargs["layers"] = self.parallel_action_post_bidir_layers
            self.semantic_geometry_action = initializer_cls(**initializer_kwargs)
        self.oft_action_tokenizer = (
            OFTInternalActionProjector(
                language_dim=int(language_dim), output_dim=int(d_model),
                chunk_size=self.action_chunk_size, action_dim=self.action_dim,
                width=int(parallel_vla_gfm_width), heads=int(parallel_vla_gfm_heads),
            )
            if self.parallel_vla_gfm_enabled
            and self.parallel_vla_gfm_mode in {"oft_gfm", "oft_direct"}
            else None
        )
        if self.oft_action_tokenizer is not None:
            # Only one semantic initializer is active in a run.
            self.semantic_geometry_action = None
        self.parallel_action_head = (
            ParallelContinuousActionHead(
                input_dim=int(da3.embed_dim), action_dim=int(action_dim)
            )
            if self.parallel_vla_gfm_enabled else None
        )
        self.parallel_action_correction_head = (
            ParallelContinuousActionHead(
                input_dim=int(da3.embed_dim), action_dim=int(action_dim)
            )
            if self.parallel_vla_gfm_enabled
            and self.parallel_action_decode_mode == "geometry_residual"
            else None
        )
        if self.parallel_action_correction_head is not None:
            # R1 starts as the exact VLA base policy. Geometry must learn only
            # a correction, rather than replacing the semantic plan at step 0.
            final = self.parallel_action_correction_head.model[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        # Auxiliary endpoint supervision. Given the observed episode-relative
        # pose token, predict the GT pose reached after the complete action
        # chunk, still relative to episode F0, as [x,y,z,sin(yaw),cos(yaw)].
        pose_hidden = max(int(da3.embed_dim) // 2, 256)
        self.relative_pose_head = nn.Sequential(
            nn.LayerNorm(int(da3.embed_dim)),
            nn.Linear(int(da3.embed_dim), pose_hidden),
            nn.SiLU(),
            nn.Linear(pose_hidden, 5),
        )
        for parameter in self.relative_pose_head.parameters():
            parameter.requires_grad = self.relative_pose_head_enabled
        self.depth_scale_head = nn.Sequential(
            nn.LayerNorm(int(da3.embed_dim)),
            nn.Linear(int(da3.embed_dim), 1),
        )
        nn.init.zeros_(self.depth_scale_head[-1].weight)
        if float(depth_scale_init_meters) <= 0.0:
            raise ValueError("depth_scale_init_meters must be positive.")
        nn.init.constant_(
            self.depth_scale_head[-1].bias,
            float(torch.tensor(float(depth_scale_init_meters)).log().item()),
        )
        for parameter in self.depth_scale_head.parameters():
            parameter.requires_grad = self.depth_scale_head_enabled
        # The same DA3-deep action token feeds both continuous action and Stop.
        # Thus Stop remains explicit while still sharing the frozen geometric
        # reasoning path with action, exactly like GAM's refine branch.
        if self.stop_head_mode == "legacy_action_token":
            self.stop_head = nn.Sequential(
                nn.LayerNorm(int(da3.embed_dim)),
                nn.Linear(int(da3.embed_dim), 1),
            )
            nn.init.zeros_(self.stop_head[-1].weight)
            nn.init.constant_(self.stop_head[-1].bias, -3.0)
        elif self.stop_head_mode == "action_hidden_pose":
            self.stop_head = ActionHiddenPoseStopHead(
                action_dim=int(da3.embed_dim), model_dim=int(stop_model_dim),
            )
        elif self.stop_head_mode == "patch_motion":
            if not self.use_fixed_first_frame or not self.use_pose_history:
                raise ValueError(
                    "patch_motion Stop requires use_fixed_first_frame=true and use_pose_history=true."
                )
            self.stop_head = PatchMotionStopHead(
                visual_dim=int(da3.embed_dim), language_dim=int(language_dim),
                pose_dim=5, model_dim=int(stop_model_dim), num_heads=int(stop_num_heads),
                visual_prefix_tokens=1 + int(getattr(da3, "num_register_tokens", 0)),
                include_plan_tokens=False,
            )
        elif self.stop_head_mode == "hybrid_action_feature":
            if not self.use_fixed_first_frame:
                raise ValueError(
                    "hybrid Stop requires use_fixed_first_frame=true."
                )
            self.stop_head = HybridPatchActionStopHead(
                visual_dim=int(da3.embed_dim), language_dim=int(language_dim),
                pose_dim=5, model_dim=int(stop_model_dim), num_heads=int(stop_num_heads),
                num_queries=int(stop_num_queries), decoder_layers=int(stop_decoder_layers),
                visual_prefix_tokens=1 + int(getattr(da3, "num_register_tokens", 0)),
                future_gate_init=float(stop_future_gate_init),
                # Terminal-window repetition yields roughly a 19% training
                # prior, while validation remains physically unreplicated.
                initial_stop_probability=0.192,
            )
        else:
            raise ValueError(
                f"Unsupported stop_head_mode={self.stop_head_mode!r}; "
                "expected 'legacy_action_token', 'action_hidden_pose', 'patch_motion', or "
                "'hybrid_action_feature'."
            )
        for parameter in self.stop_head.parameters():
            parameter.requires_grad = self.stop_head_enabled
        # These output heads are irrelevant for this architecture. GAM's
        # action-history input projection remains trainable: when enabled it
        # receives only actions already executed before each observed image.
        # Otherwise its all-zero input acts as a learned query slot.
        unused_modules = [
            self.predictor.future_proprio_proj,
            self.predictor.out_proprio_norm,
        ]
        if not self.direct_action_enabled:
            unused_modules.extend([
                self.predictor.action_proj,
                self.predictor.out_action_norm,
            ])
        if not self.use_pose_history:
            unused_modules.append(self.predictor.proprio_token_proj)
        for module in unused_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        gate = min(max(float(residual_gate_init), 1e-4), 1.0 - 1e-4)
        self.residual_gate_logit = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(gate)).item())),
            requires_grad=self.residual_prediction,
        )
        # Construct last: disabled ablations retain their previous RNG sequence.
        parallel_geometry_layers = None
        self.current_geometry_patch_mode = "concat"
        if self.parallel_vla_gfm_enabled and self.parallel_current_geometry_read_enabled:
            transformer = getattr(getattr(da3, "backbone", None), "pretrained", None)
            if transformer is not None and hasattr(transformer, "blocks"):
                start = int(getattr(transformer, "alt_start", 12))
                if self.current_geometry_bank_mode == "every_layer_concat":
                    parallel_geometry_layers = list(range(start, len(transformer.blocks)))
                elif self.current_geometry_bank_mode in {"output_current", "output_concat"}:
                    parallel_geometry_layers = [int(index) for index in da3.out_layers]
                elif self.current_geometry_bank_mode == "global_current":
                    parallel_geometry_layers = [
                        index for index in range(start, len(transformer.blocks)) if index % 2 == 1
                    ]
            self.current_geometry_patch_mode = (
                "concat" if self.current_geometry_bank_mode.endswith("concat") else "current"
            )
        self.current_geometry_layer_indices = (
            None if parallel_geometry_layers is None else tuple(parallel_geometry_layers)
        )
        self.current_geometry_read = (
            CurrentGeometryRead(
                int(da3.embed_dim),
                layer_indices=parallel_geometry_layers,
                gate_init=(0.05 if self.parallel_vla_gfm_enabled else 1e-3),
                memory_dim=(
                    int(2 * da3.embed_dim)
                    if self.current_geometry_patch_mode == "concat"
                    else int(da3.embed_dim)
                ),
            )
            if (
                self.current_geometry_action_enabled
                or (
                    self.parallel_vla_gfm_enabled
                    and self.parallel_current_geometry_read_enabled
                )
            ) else None
        )
        self.direct_current_seed = (
            DirectCurrentActionSeed(int(da3.embed_dim), int(language_dim))
            if self.geometry_architecture == "direct_current" else None
        )
        self.prediction_roles = (
            nn.Parameter(torch.randn(2, int(d_model)) * 0.02)
            if self.geometry_architecture == "dual_predicted" else None
        )
        if self.direct_current_seed is not None:
            if not self.use_fixed_first_frame:
                self.direct_current_seed.reference_role.requires_grad = False
            # Keep the legacy predictor object only for config/checkpoint API
            # compatibility. It is neither executed nor optimized in B.
            for parameter in self.predictor.parameters():
                parameter.requires_grad = False
            for parameter in (self.reference_step_embed, self.residual_gate_logit,
                              self.missing_action_embed, self.missing_pose_embed):
                parameter.requires_grad = False
            if self.use_pose_history or self.use_action_history:
                raise ValueError("direct_current control does not accept numeric pose/action history")

    @property
    def residual_gate(self) -> torch.Tensor:
        return self.residual_gate_logit.sigmoid()

    def train(self, mode: bool = True) -> "UAVFlowPredictorIDM":
        super().train(mode)
        # The shallow encoder remains frozen, while GAM-style runs may fully
        # optimize the post-boundary DA3 blocks. DPT is always a frozen decoder.
        self.da3.eval()
        if mode and self.train_deep_backbone:
            blocks = self.da3.backbone.pretrained.blocks
            for index, block in enumerate(blocks):
                block.train(index >= self.deep_train_start_block)
        self.da3.dpt_head.eval()
        if self.idm is not None:
            self.idm.eval()
        return self

    def rollout_shallow(
        self,
        observed_shallow: torch.Tensor,
        *,
        reference_shallow: torch.Tensor,
        observed_action_history: torch.Tensor | None,
        observed_action_history_valid_mask: torch.Tensor | None,
        observed_pose_history: torch.Tensor | None,
        reference_pose: torch.Tensor | None,
        lang_feats: torch.Tensor,
        lang_padding_mask: torch.Tensor | None,
        action_slot_seed: torch.Tensor | None = None,
        conditioning_generator: torch.Generator | None = None,
        force_action_history_missing: bool = False,
        force_pose_history_missing: bool = False,
        prediction_role: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict future shallow tokens; normal training uses exactly one step.

        ``rollout_steps>1`` is retained only to evaluate checkpoints produced by
        the earlier open-loop protocol. New sliding-IDM configs use one step.
        """
        if observed_shallow.ndim != 5 or observed_shallow.shape[1] < 1:
            raise ValueError(f"Expected observed shallow (B,H,V,N,D), got {observed_shallow.shape}.")
        if self.use_fixed_first_frame and (
            reference_shallow is None
            or reference_shallow.ndim != 5
            or reference_shallow.shape[1] != 1
            or reference_shallow.shape[0] != observed_shallow.shape[0]
            or reference_shallow.shape[2:] != observed_shallow.shape[2:]
        ):
            raise ValueError(
                "reference_shallow must be the episode F0 shaped (B,1,V,N,D), got "
                f"{None if reference_shallow is None else tuple(reference_shallow.shape)} "
                f"for observed {tuple(observed_shallow.shape)}."
            )
        sequence = observed_shallow
        b, observed_h = observed_shallow.shape[:2]
        if self.use_pose_history:
            if observed_pose_history is None or observed_pose_history.shape != (b, observed_h, 5):
                raise ValueError(
                    "use_pose_history=true requires normalized pose shaped "
                    f"(B,H,5)={(b, observed_h, 5)}, got "
                    f"{None if observed_pose_history is None else tuple(observed_pose_history.shape)}."
                )
            # Training noise is applied in raw metres/degrees before
            # normalization by prepare_observed_pose_history().
            pose_history = observed_pose_history
            if self.use_fixed_first_frame and (
                reference_pose is None or reference_pose.shape != (b, 1, 5)
            ):
                raise ValueError(
                    "use_pose_history=true requires normalized F0 pose shaped "
                    f"(B,1,5), got {None if reference_pose is None else tuple(reference_pose.shape)}."
                )
        else:
            pose_history = None
        if self.use_action_history:
            if observed_action_history is None:
                raise ValueError("use_action_history=true requires observed_action_history.")
            if observed_action_history.shape[:2] != (b, observed_h):
                raise ValueError(
                    "Observed action history must align with visual history: "
                    f"visual={(b, observed_h)}, action={tuple(observed_action_history.shape)}."
                )
            action_history = observed_action_history
            if observed_action_history_valid_mask is None:
                raise ValueError(
                    "use_action_history=true requires observed_action_history_valid_mask."
                )
            if observed_action_history_valid_mask.shape[:2] != (b, observed_h):
                raise ValueError(
                    "Observed action-history mask must align with visual history: "
                    f"visual={(b, observed_h)}, mask={tuple(observed_action_history_valid_mask.shape)}."
                )
            action_history_valid = observed_action_history_valid_mask.to(dtype=torch.bool)
        else:
            action_history = torch.zeros(
                b, observed_h, 1, self.action_dim,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            action_history_valid = None
        predicted = []
        direct_action_tokens = None
        for rollout_index in range(self.rollout_steps):
            b = int(sequence.shape[0])
            if self.use_fixed_first_frame:
                predictor_sequence = torch.cat([reference_shallow, sequence], dim=1)
                predictor_step_roles = None
                if self.use_reference_type_embedding:
                    predictor_step_roles = torch.zeros(
                        b, predictor_sequence.shape[1], self.predictor.d_model,
                        device=predictor_sequence.device,
                        dtype=predictor_sequence.dtype,
                    )
                    predictor_step_roles[:, 0] = self.reference_step_embed.to(
                        device=predictor_sequence.device,
                        dtype=predictor_sequence.dtype,
                    )
                predictor_actions = torch.cat([
                    torch.zeros(
                        b, 1, 1, self.action_dim,
                        device=action_history.device, dtype=action_history.dtype,
                    ),
                    action_history,
                ], dim=1)
                predictor_action_valid = None
                if action_history_valid is not None:
                    predictor_action_valid = torch.cat([
                        torch.zeros(
                            b, 1, *action_history_valid.shape[2:],
                            device=action_history_valid.device, dtype=torch.bool,
                        ),
                        action_history_valid,
                    ], dim=1)
                predictor_pose = (
                    torch.cat([reference_pose, pose_history], dim=1)
                    if pose_history is not None else None
                )
            else:
                predictor_sequence = sequence
                predictor_step_roles = None
                predictor_actions = action_history
                predictor_action_valid = action_history_valid
                predictor_pose = pose_history
            predictor_pose_valid = None
            if predictor_pose is not None:
                predictor_pose_valid = torch.ones(
                    b, predictor_pose.shape[1], device=predictor_pose.device, dtype=torch.bool
                )
                if force_pose_history_missing:
                    predictor_pose_valid.zero_()
                if self.training and self.pose_history_keep_prob < 1.0:
                    # Drop the whole trajectory condition per sample instead
                    # of creating impossible per-timestep holes.
                    pose_keep = (
                        torch.rand(
                            b, 1, device=predictor_pose.device,
                            generator=conditioning_generator,
                        )
                        < self.pose_history_keep_prob
                    )
                    predictor_pose_valid = predictor_pose_valid & pose_keep
            if (
                predictor_action_valid is not None
                and force_action_history_missing
            ):
                predictor_action_valid = torch.zeros_like(
                    predictor_action_valid, dtype=torch.bool
                )
            if (
                predictor_action_valid is not None
                and self.training
                and self.action_history_keep_prob < 1.0
            ):
                action_keep = (
                    torch.rand(
                        b, 1, device=predictor_action_valid.device,
                        generator=conditioning_generator,
                    )
                    < self.action_history_keep_prob
                )
                while action_keep.ndim < predictor_action_valid.ndim:
                    action_keep = action_keep.unsqueeze(-1)
                predictor_action_valid = predictor_action_valid & action_keep
            if prediction_role is not None:
                if predictor_step_roles is None:
                    predictor_step_roles = sequence.new_zeros(
                        b, predictor_sequence.shape[1], self.predictor.d_model
                    )
                predictor_step_roles = predictor_step_roles + prediction_role.to(predictor_step_roles)
            out = self.predictor(
                past_visual_tokens=predictor_sequence,
                proprio_history=predictor_pose,
                proprio_history_valid_mask=predictor_pose_valid,
                missing_proprio_embed=(self.missing_pose_embed if self.use_pose_history else None),
                past_action_history=predictor_actions,
                past_action_history_valid_mask=predictor_action_valid,
                missing_action_embed=(self.missing_action_embed if self.use_action_history else None),
                action_slot_seed=action_slot_seed,
                step_role_embeddings=predictor_step_roles,
                lang_feats=(
                    None if (self.vlm_action_seed_enabled or self.parallel_vla_gfm_enabled)
                    else lang_feats
                ),
                lang_padding_mask=(
                    None if (self.vlm_action_seed_enabled or self.parallel_vla_gfm_enabled)
                    else lang_padding_mask
                ),
            )
            if self.dense_context_supervision:
                # The synthetic fixed-F0 reference is conditioning only.  Drop
                # its output and retain one causal next-state/action prediction
                # for every genuine observed anchor, matching released GAM's
                # dense-H training contract.
                output_start = 1 if self.use_fixed_first_frame else 0
                output_end = output_start + observed_h
                dense_update = out["predicted_next_visual_tokens"][:, output_start:output_end]
                dense_action_tokens = out["predicted_action_tokens"][:, output_start:output_end]
                if dense_update.shape[1] != observed_h or dense_action_tokens.shape[1] != observed_h:
                    raise RuntimeError(
                        "Dense predictor output does not align with observed context: "
                        f"observed_h={observed_h}, visual={tuple(dense_update.shape)}, "
                        f"action={tuple(dense_action_tokens.shape)}."
                    )
                dense_next = (
                    sequence + self.residual_gate.to(dense_update.dtype) * dense_update
                    if self.residual_prediction else dense_update
                )
                return dense_next, dense_action_tokens
            update = out["predicted_next_visual_tokens"][:, -1:]
            direct_action_tokens = out["predicted_action_tokens"][:, -1:]
            next_shallow = (
                sequence[:, -1:] + self.residual_gate.to(update.dtype) * update
                if self.residual_prediction
                else update
            )
            predicted.append(next_shallow)
            if rollout_index + 1 >= self.rollout_steps:
                continue
            # Legacy open-loop compatibility: later predicted frames have no
            # executed action, so their action slot is explicitly missing.
            sequence = torch.cat([sequence, next_shallow], dim=1)
            if pose_history is not None:
                # Pose of an unexecuted future observation is unavailable.
                # New Stage-2 training uses rollout_steps=1; reject ambiguous
                # legacy open-loop use instead of silently leaking GT pose.
                raise ValueError("Pose conditioning supports rollout_steps=1 only.")
            action_history = torch.cat([
                action_history,
                torch.zeros(
                    b, 1, 1, self.action_dim,
                    device=action_history.device,
                    dtype=action_history.dtype,
                ),
            ], dim=1)
            if action_history_valid is not None:
                action_history_valid = torch.cat([
                    action_history_valid,
                    torch.zeros(
                        b, 1, *action_history_valid.shape[2:],
                        device=action_history_valid.device,
                        dtype=torch.bool,
                    ),
                ], dim=1)
        return torch.cat(predicted, dim=1), direct_action_tokens

    def build_sliding_idm_window(
        self,
        observed_shallow: torch.Tensor,
        predicted_next: torch.Tensor,
    ) -> torch.Tensor:
        """Left-pad observed history and append one predicted next frame."""
        if self.idm is None:
            raise RuntimeError("Cannot build an IDM window when compute_idm_branch=false.")
        history_slots = int(getattr(self.idm, "window_size")) - 1
        if history_slots < 1:
            raise ValueError(f"Invalid IDM history slots: {history_slots}.")
        history = observed_shallow[:, -history_slots:]
        missing = history_slots - int(history.shape[1])
        if missing > 0:
            history = torch.cat(
                [observed_shallow[:, :1].expand(-1, missing, -1, -1, -1), history],
                dim=1,
            )
        window = torch.cat([history, predicted_next], dim=1)
        if int(window.shape[1]) != int(getattr(self.idm, "window_size")):
            raise RuntimeError(
                f"Sliding IDM window has {window.shape[1]} frames; "
                f"expected {getattr(self.idm, 'window_size')}."
            )
        return window

    def forward(
        self,
        observed_shallow: torch.Tensor,
        *,
        reference_shallow: torch.Tensor,
        observed_action_history: torch.Tensor | None = None,
        observed_action_history_valid_mask: torch.Tensor | None = None,
        observed_pose_history: torch.Tensor | None = None,
        reference_pose: torch.Tensor | None = None,
        stop_pose: torch.Tensor | None = None,
        lang_feats: torch.Tensor,
        lang_padding_mask: torch.Tensor | None,
        action_targets_norm: torch.Tensor | None = None,
        conditioning_generator: torch.Generator | None = None,
        force_action_history_missing: bool = False,
        force_pose_history_missing: bool = False,
    ) -> dict[str, object]:
        if self.current_geometry_action_enabled and observed_shallow.shape[1] != 1:
            raise ValueError("Current geometry action ablation supports H=1 only (avoid temporal leakage)")
        if self.geometry_architecture != "legacy" and observed_shallow.shape[1:3] != (1, 1):
            raise ValueError("Geometry architecture ablations require H=1 and one physical camera")
        action_slot_seed = None
        oft_direct_prediction = None
        if self.parallel_vla_gfm_enabled:
            if self.oft_action_tokenizer is not None:
                oft_output = self.oft_action_tokenizer(lang_feats, lang_padding_mask)
                action_slot_seed = oft_output["plan_tokens"]
                oft_direct_prediction = oft_output["direct_actions_norm"][:, None]
            else:
                if self.semantic_geometry_action is None:
                    raise RuntimeError("parallel semantic action initializer is missing")
                action_slot_seed = self.semantic_geometry_action(
                    lang_feats, lang_padding_mask
                )
        if self.vlm_action_seed_enabled:
            if self.vlm_action_seed is None:
                raise RuntimeError("VLM action seed is enabled but its projector is missing.")
            if lang_padding_mask is None:
                last_index = torch.full(
                    (lang_feats.shape[0],), lang_feats.shape[1] - 1,
                    device=lang_feats.device, dtype=torch.long,
                )
            else:
                keep = lang_padding_mask.to(device=lang_feats.device, dtype=torch.bool)
                last_index = keep.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = lang_feats[
                torch.arange(lang_feats.shape[0], device=lang_feats.device), last_index
            ]
            action_slot_seed = self.vlm_action_seed(pooled)
        rollout_kwargs = dict(
            reference_shallow=reference_shallow,
            observed_action_history=observed_action_history,
            observed_action_history_valid_mask=observed_action_history_valid_mask,
            observed_pose_history=observed_pose_history,
            reference_pose=reference_pose,
            lang_feats=lang_feats,
            lang_padding_mask=lang_padding_mask,
            action_slot_seed=action_slot_seed,
            conditioning_generator=conditioning_generator,
            force_action_history_missing=force_action_history_missing,
            force_pose_history_missing=force_pose_history_missing,
        )
        predicted_current = None
        current_action_tokens = None
        if self.direct_current_seed is not None:
            future = observed_shallow
            direct_action_tokens = self.direct_current_seed(
                observed_shallow,
                reference_shallow if self.use_fixed_first_frame else None,
                lang_feats, lang_padding_mask,
            )
        else:
            role = {} if self.prediction_roles is None else {"prediction_role": self.prediction_roles[1]}
            future, direct_action_tokens = self.rollout_shallow(observed_shallow, **rollout_kwargs, **role)
            if self.prediction_roles is not None:
                # Shared Predictor weights, two target-role-conditioned passes.
                predicted_current, current_action_tokens = self.rollout_shallow(
                    observed_shallow, **rollout_kwargs, prediction_role=self.prediction_roles[0]
                )
        deep_joint_features = None
        current_depth_output = None
        current_geometry_features = None
        refined_action_tokens = None
        dual_refined_action_tokens = None
        dual_action_future_weight = None
        if self.deep_action_enabled or self.stop_head_enabled:
            if direct_action_tokens is None:
                raise RuntimeError("Joint DA3-deep propagation requires predictor action tokens.")
            # Keep the current predictor/data contract unchanged: its predicted
            # visual feature and dedicated action seed enter frozen DA3 deeper
            # blocks together. Gradients flow through those frozen operations
            # back into both predictor outputs.
            is_dual = self.geometry_architecture in DUAL_ARCHITECTURES
            deep_visuals = future
            deep_actions = direct_action_tokens
            deep_kwargs = {}
            if self.parallel_vla_gfm_enabled and (
                self.parallel_current_depth_enabled
                or self.parallel_current_geometry_read_enabled
            ):
                # Bank 2: current observation takes a pure action-free DA3
                # deep pass.  Each corresponding deep layer becomes geometry
                # memory for the K action tokens that already passed through
                # the Future Predictor with the predicted future features.
                current_geometry_features = self.da3.propagate_shallow_visual_slots_grad(
                    observed_shallow,
                    gradient_checkpointing=self.deep_gradient_checkpointing,
                    return_layer_patches=self.parallel_current_geometry_read_enabled,
                    layer_patch_indices=(
                        self.current_geometry_layer_indices
                        if self.parallel_current_geometry_read_enabled else None
                    ),
                    layer_patch_mode=self.current_geometry_patch_mode,
                )
                if self.parallel_current_geometry_read_enabled:
                    if self.current_geometry_read is None:
                        raise RuntimeError("Parallel Current Geometry read has no CA module")
                    deep_kwargs.update(
                        current_geometry_by_layer=current_geometry_features["layer_patches"],
                        current_geometry_read=self.current_geometry_read,
                    )
            if (not self.parallel_vla_gfm_enabled and self.current_geometry_read is not None
                    and self.current_geometry_read_mode == "per_layer"):
                current_geometry_features = self.da3.propagate_shallow_visual_slots_grad(
                    observed_shallow, gradient_checkpointing=self.deep_gradient_checkpointing,
                    return_layer_patches=True,
                )
                deep_kwargs.update(
                    current_geometry_by_layer=current_geometry_features["layer_patches"],
                    current_geometry_read=self.current_geometry_read,
                )
            if is_dual:
                current = observed_shallow if predicted_current is None else predicted_current
                seed = direct_action_tokens if current_action_tokens is None else current_action_tokens
                deep_visuals = torch.cat([current, future], dim=2)
                deep_actions = torch.cat([seed, direct_action_tokens], dim=2)
                deep_kwargs["dual_state_attention"] = (
                    "action_bridge" if self.geometry_architecture == "dual_action_bridge" else "full"
                )
            deep_joint_features = self.da3.propagate_shallow_with_actions_grad(
                deep_visuals,
                deep_actions,
                decode_visuals=self.depth_decode_enabled,
                gradient_checkpointing=self.deep_gradient_checkpointing,
                deep_temporal_causal_mask=True,
                **deep_kwargs,
            )
            deep_tokens = deep_joint_features.get("action_tokens")
            if not isinstance(deep_tokens, torch.Tensor):
                raise RuntimeError("DA3 joint propagation did not return action_tokens.")
            b, steps, views = deep_visuals.shape[:3]
            if is_dual:
                # Preserve and supervise both branches, then learn a scalar
                # Current/Future gate per temporal action step.  This replaces
                # the old destructive fixed 0.5/0.5 hidden-state average.
                dual_refined_action_tokens = deep_tokens.reshape(b, steps, 2, -1)
                if self.dual_action_fusion is None:
                    raise RuntimeError("Dual geometry mode is missing dual_action_fusion")
                fused, dual_action_future_weight = self.dual_action_fusion(
                    dual_refined_action_tokens[:, :, 0],
                    dual_refined_action_tokens[:, :, 1],
                )
                refined_action_tokens = fused.unsqueeze(2)
                direct_action_tokens = (
                    (1.0 - dual_action_future_weight) * deep_actions[:, :, 0]
                    + dual_action_future_weight * deep_actions[:, :, 1]
                ).unsqueeze(2)
                current_depth_output = select_dual_view(deep_joint_features, b, 0)
                deep_joint_features = select_dual_view(deep_joint_features, b, 1)
            else:
                if self.parallel_vla_gfm_enabled:
                    refined_action_tokens = deep_tokens.reshape(
                        b, steps, views, self.action_chunk_size, -1
                    )
                    direct_action_tokens = deep_actions
                else:
                    refined_action_tokens = deep_tokens.reshape(b, steps, views, -1)
                if self.geometry_architecture in {"current_prediction", "direct_current"}:
                    current_depth_output = deep_joint_features
            if (not self.parallel_vla_gfm_enabled and self.current_geometry_read is not None
                    and self.current_geometry_read_mode == "terminal"):
                current_geometry_features = self.da3.propagate_shallow_visual_slots_grad(
                    observed_shallow,
                    gradient_checkpointing=self.deep_gradient_checkpointing,
                )
                prefix = 1 + int(getattr(self.da3, "num_register_tokens", 0))
                patches = current_geometry_features["deep_levels"][-1][..., prefix:, :]
                refined_action_tokens = self.current_geometry_read(refined_action_tokens, patches)
        direct_actions_norm = None
        action_token_output = None
        geometry_action_residual = None
        if self.parallel_vla_gfm_enabled:
            if self.parallel_action_head is None:
                raise RuntimeError("dual_vla_gfm parallel action head is missing")
            direct_actions_norm = (
                oft_direct_prediction
                if self.parallel_vla_gfm_mode == "oft_direct"
                else self.parallel_action_head(direct_action_tokens)
            )
        elif self.direct_action_enabled and not self.causal_action_decoder_enabled:
            if direct_action_tokens is None:
                raise RuntimeError("Direct action branch requires predicted_action_tokens.")
            # Released GAM applies the same action head before and after the
            # DA3 deep stack and supervises both predictions.
            direct_actions_norm = self.direct_action_head(direct_action_tokens)
        refine_actions_norm = None
        if self.parallel_vla_gfm_enabled:
            if self.parallel_action_decode_mode == "geometry_residual":
                if self.parallel_action_correction_head is None:
                    raise RuntimeError("Geometry-residual decode has no correction head")
                geometry_action_residual = self.parallel_action_correction_head(
                    refined_action_tokens
                )
                refine_actions_norm = direct_actions_norm + geometry_action_residual
            else:
                geometry_action_residual = None
                refine_actions_norm = self.parallel_action_head(refined_action_tokens)
        elif self.causal_action_decoder_enabled:
            decoder_tokens = (
                refined_action_tokens
                if refined_action_tokens is not None else direct_action_tokens
            )
            if decoder_tokens is None or self.causal_action_decoder is None:
                raise RuntimeError("Causal action decoding requires a predicted action token.")
            action_token_output = self.causal_action_decoder(
                decoder_tokens, target_actions_norm=action_targets_norm
            )
            # Teacher forcing supplies the stable token CE used for training,
            # but its per-position argmax is not a faithful policy rollout:
            # every position has seen the preceding GT action tokens.  During
            # validation, keep the teacher-forced logits/targets for CE while
            # reporting actions from a genuinely autoregressive 20-token
            # decode.  Closed-loop inference already enters the decoder with
            # ``action_targets_norm=None`` and therefore takes this path once.
            if not self.training and action_targets_norm is not None:
                autoregressive = self.causal_action_decoder(
                    decoder_tokens, target_actions_norm=None
                )
                action_token_output["actions_norm"] = autoregressive["actions_norm"]
                action_token_output["token_ids"] = autoregressive["token_ids"]
            refine_actions_norm = action_token_output["actions_norm"]
        elif refined_action_tokens is not None:
            refine_actions_norm = self.direct_action_head(refined_action_tokens)
        dual_refine_actions_norm = None
        if dual_refined_action_tokens is not None:
            dual_refine_actions_norm = torch.stack([
                self.direct_action_head(dual_refined_action_tokens[:, :, index:index + 1])
                for index in range(2)
            ], dim=2)

        def auxiliary_action_hidden(tokens: torch.Tensor) -> torch.Tensor:
            """Pool only for scalar/endpoint auxiliary heads, never Action output."""
            if tokens.ndim == 5:  # [B,T,V,K,D]
                return tokens.mean(dim=(2, 3))
            if tokens.ndim == 4:  # [B,T,V,D]
                return tokens.mean(dim=2)
            if tokens.ndim == 3:
                return tokens
            raise ValueError(f"Unexpected auxiliary action-token shape {tuple(tokens.shape)}")

        stop_logits = None
        if self.stop_head_enabled:
            if self.stop_head_mode in {"patch_motion", "hybrid_action_feature"}:
                if observed_pose_history is None and self.stop_head_mode == "patch_motion":
                    raise RuntimeError("patch_motion Stop requires observed_pose_history.")
                stop_logits = self.stop_head(
                    reference_visual=reference_shallow,
                    current_visual=observed_shallow,
                    current_pose=observed_pose_history,
                    language=lang_feats,
                    language_padding_mask=lang_padding_mask,
                    predicted_future=(
                        future if self.stop_head_mode == "hybrid_action_feature" else None
                    ),
                    deep_action_tokens=(
                        (
                            refined_action_tokens.mean(dim=3)
                            if refined_action_tokens.ndim == 5
                            else refined_action_tokens
                        )
                        if self.stop_head_mode == "hybrid_action_feature" else None
                    ),
                )
            elif self.stop_head_mode == "action_hidden_pose":
                if stop_pose is None:
                    raise ValueError("Action-hidden Stop requires stop-only current pose.")
                stop_logits = self.stop_head(
                    auxiliary_action_hidden(refined_action_tokens), stop_pose,
                )
            else:
                stop_logits = self.stop_head(
                    auxiliary_action_hidden(refined_action_tokens)
                ).squeeze(-1)

        relative_pose = None
        if self.relative_pose_head_enabled:
            pose_tokens = (
                refined_action_tokens
                if refined_action_tokens is not None
                else direct_action_tokens
            )
            if pose_tokens is None:
                raise RuntimeError("Relative-pose head requires predicted action tokens.")
            relative_pose = self.relative_pose_head(auxiliary_action_hidden(pose_tokens))

        depth_log_scale = None
        if self.depth_scale_head_enabled:
            scale_tokens = (
                refined_action_tokens
                if refined_action_tokens is not None
                else direct_action_tokens
            )
            if scale_tokens is None:
                raise RuntimeError("Depth scale head requires predictor action tokens.")
            depth_log_scale = self.depth_scale_head(
                auxiliary_action_hidden(scale_tokens)
            ).squeeze(-1)

        idm_features = None
        all_window_actions_norm = None
        idm_actions_norm = None
        if self.compute_idm_branch:
            if self.rollout_steps == 1:
                idm_shallow = self.build_sliding_idm_window(observed_shallow, future)
            else:
                # Exact compatibility with pre-sliding three-step checkpoints.
                idm_shallow = torch.cat([observed_shallow[:, -1:], future], dim=1)
            idm_features = self.da3.propagate_shallow_visual_slots_grad(
                idm_shallow,
                gradient_checkpointing=self.deep_gradient_checkpointing,
            )
            all_window_actions_norm = self.idm(idm_features)
            if all_window_actions_norm.ndim != 3 or all_window_actions_norm.shape[1] < 1:
                raise ValueError(
                    "IDM must return adjacent window actions shaped (B,W-1,D), got "
                    f"{tuple(all_window_actions_norm.shape)}."
                )
            idm_actions_norm = (
                all_window_actions_norm[:, -1:]
                if self.rollout_steps == 1
                else all_window_actions_norm
            )
        actions_norm = (
            direct_actions_norm
            if self.parallel_vla_gfm_enabled and self.parallel_vla_gfm_mode == "oft_direct"
            else refine_actions_norm
            if refine_actions_norm is not None
            else direct_actions_norm if direct_actions_norm is not None else idm_actions_norm
        )
        if actions_norm is None:
            raise RuntimeError("Neither direct nor IDM action branch is enabled.")
        return {
            "future_shallow": future,
            "predicted_current_shallow": predicted_current,
            "current_depth_output": current_depth_output,
            "idm_features": idm_features,
            "actions_norm": actions_norm,
            "direct_actions_norm": direct_actions_norm,
            "refine_actions_norm": refine_actions_norm,
            "geometry_action_residual_norm": geometry_action_residual,
            "dual_refine_actions_norm": dual_refine_actions_norm,
            "dual_action_future_weight": dual_action_future_weight,
            "action_token_logits": (
                None if action_token_output is None else action_token_output["logits"]
            ),
            "action_token_target_ids": (
                None if action_token_output is None else action_token_output.get("target_ids")
            ),
            "action_token_ids": (
                None if action_token_output is None else action_token_output["token_ids"]
            ),
            "idm_actions_norm": idm_actions_norm,
            "all_window_actions_norm": all_window_actions_norm,
            "deep_joint_features": deep_joint_features,
            "current_geometry_features": current_geometry_features,
            "stop_logits": stop_logits,
            "relative_pose": relative_pose,
            "depth_log_scale": depth_log_scale,
            "stop_future_gate": (
                self.stop_head.future_gate
                if self.stop_head_enabled
                and self.stop_head_mode == "hybrid_action_feature"
                else future.new_zeros(())
            ),
            "residual_gate": self.residual_gate,
        }
