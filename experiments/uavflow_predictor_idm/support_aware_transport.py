"""Candidate B, with optional completion-only scalar error estimation.

The two future readouts live in the SAME final DA3 feature space. Confidence
scales completion VALUES after softmax; it is neither an attention-logit bias
nor a shallow covariance propagated unchanged through DA3.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .patch_transport import (
    actions_to_camera_transform, bilinear_patch_transport, heading_motion,
    nominal_intrinsics,
)


class GeometryRead(nn.Module):
    """Soft support weights, finite empty memories, and post-softmax confidence."""
    def __init__(self, query_dim, memory_dim, width=256, heads=8):
        super().__init__()
        if width % heads:
            raise ValueError("Geometry reader width must divide heads")
        self.heads, self.width = heads, width
        self.q = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, width, bias=False))
        self.kv = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, 2*width, bias=False))

    def forward(self, queries, memory, support, confidence=None):
        b, slots, _ = queries.shape
        if memory.ndim != 3 or support.shape != memory.shape[:2]:
            raise ValueError("Geometry memory/support must be [B,N,D] / [B,N]")
        q = self.q(queries).reshape(b, slots, self.heads, -1).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, self.width//self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, self.width//self.heads).transpose(1, 2)
        keep = support > 0
        logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits = logits + support.float().clamp_min(1e-12).log()[:, None, None]
        # No dummy learnable key or projection bias can resurrect an empty
        # branch. An entirely unsupported memory gives exactly zero output.
        logits = logits.masked_fill(~keep[:, None, None], -1e9)
        attention = logits.softmax(-1) * keep[:, None, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        if confidence is not None:
            if confidence.shape != support.shape:
                raise ValueError("Completion confidence must be [B,N]")
            attention = attention * confidence.detach().float()[:, None, None]
        result = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(b, slots, self.width)
        return result


class DeepReadout(nn.Module):
    def __init__(self, dim, width, *, residual):
        super().__init__()
        self.residual = residual
        self.delta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, value):
        delta = self.delta(value)
        return delta if self.residual else value + delta


class _DeepTeacherForward(nn.Module):
    def __init__(self, da3, layer_indices=None):
        super().__init__()
        self.da3 = da3
        self.layer_indices = layer_indices

    def forward(self, shallow):
        if self.layer_indices is None:
            return self.da3.propagate_shallow_visual_slots_grad(shallow, gradient_checkpointing=False)
        return self.da3.propagate_shallow_visual_slots_grad(
            shallow, gradient_checkpointing=False, return_layer_patches=True,
            layer_patch_indices=self.layer_indices, layer_patch_mode='current')


class FrozenDeepTeacher:
    """Snapshot trainable base weights BEFORE init/resume mutates the student.

    Only updated parameters need a second copy. Frozen shallow weights remain
    shared. functional_call swaps references rather than overwriting tensors,
    so outstanding student autograd graphs remain valid.
    """
    def __init__(self, da3, layer_indices=None):
        self.wrapper = _DeepTeacherForward(da3, layer_indices)
        self.weights = {name: value.detach().clone() for name, value in self.wrapper.named_parameters()
                        if value.requires_grad}

    @torch.no_grad()
    def __call__(self, shallow):
        from robot.modeling.lora import LoRALinear
        modules = list(self.wrapper.modules())
        modes = [module.training for module in modules]
        adapters = [module for module in modules if isinstance(module, LoRALinear)]
        scales = [module.scaling for module in adapters]
        try:
            self.wrapper.eval()
            for adapter in adapters:
                adapter.scaling = 0.
            return torch.func.functional_call(self.wrapper, self.weights, (shallow.detach(),), strict=False)
        finally:
            for module, mode in zip(modules, modes):
                module.training = mode
            for adapter, scale in zip(adapters, scales):
                adapter.scaling = scale


def decode_levels(da3, levels, prefix):
    """Native [B,1,1,N,2D] -> official DPT decode, retaining CLS separately."""
    pairs = [(level[:, 0, 0, prefix:], level[:, 0, 0, 0]) for level in levels]
    return da3.decode_depth_full(pairs, batch_size=levels[0].shape[0], views_per_sequence=1,
                                 frames_chunk_size=1)


class SupportAwareTransport(nn.Module):
    def __init__(self, *, visual_dim, predictor_dim, proposal_input_dim, chunk_size, levels=4, config=None):
        super().__init__()
        from omegaconf import OmegaConf
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        cfg = dict(config or {})
        allowed = {"enabled", "uncertainty", "uncertainty_tau", "uncertainty_weight",
                   "geometry_action_source", "geometry_translation_gate_meters",
                   "geometry_yaw_gate_degrees", "warp_depth_scale_meters", "fov_degrees",
                   "heading_from_camera", "cleanup_kernel", "min_hole_area", "read_width",
                   "read_heads", "readout_width", "z_absolute_tolerance", "z_relative_tolerance"}
        if set(cfg) - allowed:
            raise ValueError(f"Unknown transport settings: {sorted(set(cfg)-allowed)}")
        self.config = cfg
        self.uncertainty_mode = str(cfg.get("uncertainty", "none"))
        if self.uncertainty_mode not in {"none", "scalar"}:
            raise ValueError("Completion uncertainty supports none or scalar")
        self.action_source = str(cfg.get("geometry_action_source", "proposal"))
        if self.action_source not in {"gt", "proposal"}:
            raise ValueError("geometry_action_source must be gt or proposal")
        self.tau = float(cfg.get("uncertainty_tau", 1.))
        self.depth_scale = float(cfg.get("warp_depth_scale_meters", 100.))
        if not math.isfinite(self.tau) or not math.isfinite(self.depth_scale) or self.tau <= 0 or self.depth_scale <= 0:
            raise ValueError("Completion tau and metric depth scale must be positive")
        for name, default in (("geometry_translation_gate_meters", .25),
                              ("geometry_yaw_gate_degrees", 5.), ("uncertainty_weight", 1.)):
            if not math.isfinite(float(cfg.get(name, default))) or float(cfg.get(name, default)) < 0:
                raise ValueError(f"{name} must be non-negative")
        self.dim, self.chunk_size = visual_dim, chunk_size
        read_width = int(cfg.get("read_width", 256))
        read_heads = int(cfg.get("read_heads", 8))
        head_width = int(cfg.get("readout_width", 256))
        self.proposal_projection = nn.Sequential(nn.LayerNorm(proposal_input_dim), nn.Linear(proposal_input_dim, visual_dim))
        self.geometry_projection = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, predictor_dim))
        self.support_xy_projection = nn.Linear(3, predictor_dim)
        self.motion_projection = nn.Sequential(nn.Linear(chunk_size*4, predictor_dim), nn.SiLU(), nn.Linear(predictor_dim, predictor_dim))
        self.residual_heads = nn.ModuleList([DeepReadout(2*visual_dim, head_width, residual=True) for _ in range(levels)])
        self.completion_heads = nn.ModuleList([DeepReadout(2*visual_dim, head_width, residual=False) for _ in range(levels)])
        self.current_read = GeometryRead(visual_dim, visual_dim, read_width, read_heads)
        self.supported_read = GeometryRead(visual_dim, visual_dim, read_width, read_heads)
        self.completion_read = GeometryRead(visual_dim, visual_dim, read_width, read_heads)
        self.fusion = nn.Sequential(nn.Linear(visual_dim + 3*read_width + 4, visual_dim), nn.GELU())
        self.error_head = (
            nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, head_width), nn.SiLU(), nn.Linear(head_width, 1))
            if self.uncertainty_mode == "scalar" else None
        )

    def forward_policy(self, model, observed, reference, seed, rollout_kwargs, *,
                       action_denormalizer, action_targets_norm=None,
                       current_depth_meters=None, current_depth_valid=None, intrinsics=None):
        if observed.shape[1:3] != (1, 1):
            raise ValueError("Candidate B requires H=1 and one physical camera")
        if action_denormalizer is None:
            raise ValueError("Candidate B requires the checkpoint ActionNormalizer for metric warp")
        prefix = 1 + int(getattr(model.da3, "num_register_tokens", 0))
        b, _, _, nt, _ = observed.shape
        n = nt - prefix
        grid = math.isqrt(n)
        if grid*grid != n:
            raise ValueError("DA3 patch grid must be square")
        # VLA proposal is decoded BEFORE GAM, never from its future output.
        proposal_input = (rollout_kwargs["lang_feats"] if model.parallel_vla_gfm_mode == "qwen_tokens" else seed)
        proposal_tokens = self.proposal_projection(proposal_input)[:, None, None]
        proposal = model.parallel_action_head(proposal_tokens)
        proposal_raw = action_denormalizer(proposal.float()).detach()[:, 0]
        target_raw = (None if action_targets_norm is None else
                      action_denormalizer(action_targets_norm.float()).detach()[:, 0])
        gt_geometry = self.training and self.action_source == "gt"
        if gt_geometry and target_raw is None:
            raise ValueError("GT-action geometry pretraining requires action_targets_norm")
        geometry_actions = target_raw if gt_geometry else proposal_raw
        gate = observed.new_ones(b, dtype=torch.bool)
        if not gt_geometry and target_raw is not None:
            pm, tm = heading_motion(proposal_raw), heading_motion(target_raw)
            translation_error = (pm[:, :3, 3]-tm[:, :3, 3]).norm(dim=-1)
            yaw_error = torch.atan2(pm[:, 1, 0], pm[:, 0, 0])-torch.atan2(tm[:, 1, 0], tm[:, 0, 0])
            yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos()).abs()
            gate = ((translation_error <= float(self.config.get("geometry_translation_gate_meters", .25)))
                    & (yaw_error <= math.radians(float(self.config.get("geometry_yaw_gate_degrees", 5.)))))
        current = model.da3.propagate_shallow_visual_slots_grad(
            observed, gradient_checkpointing=model.deep_gradient_checkpointing)
        current_decoded = decode_levels(model.da3, current["deep_levels"], prefix)
        if current_depth_meters is None:
            metric = current_decoded["depth"].reshape(b, grid*14, grid*14).detach().float() * self.depth_scale
        else:
            metric = current_depth_meters.detach().float()
        if metric.shape != (b, grid*14, grid*14):
            raise ValueError("Current metric z-depth must match the DA3 input pixel grid")
        if intrinsics is None:
            intrinsics = nominal_intrinsics(metric.shape[-2], metric.shape[-1],
                                           fov_degrees=self.config.get("fov_degrees", 90.), device=metric.device)
        transform = actions_to_camera_transform(geometry_actions, self.config.get("heading_from_camera"))
        transport = bilinear_patch_transport(
            metric, intrinsics, transform, valid=current_depth_valid,
            cleanup_kernel=int(self.config.get("cleanup_kernel", 5)),
            min_hole_area=int(self.config.get("min_hole_area", 128)),
            z_absolute_tolerance=float(self.config.get("z_absolute_tolerance", .01)),
            z_relative_tolerance=float(self.config.get("z_relative_tolerance", .01)))
        w, rho = transport["matrix"], transport["rho"]
        current_last = current["deep_levels"][-1][:, 0, 0, prefix:, -self.dim:]
        warp_last = w.to(current_last.dtype) @ current_last
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, grid, device=metric.device),
                                torch.linspace(-1, 1, grid, device=metric.device), indexing="ij")
        support_xy = torch.cat((rho[..., None], torch.stack((xx, yy), -1).reshape(1, n, 2).expand(b, -1, -1)), -1)
        memory = self.geometry_projection(warp_last) + self.support_xy_projection(support_xy.to(warp_last.dtype))
        # Explicit numerical chunk conditioning, including its path order.
        motion_seed = self.motion_projection(geometry_actions.flatten(1).to(seed.dtype))[:, None]
        kwargs = dict(rollout_kwargs)
        kwargs.update(action_slot_seed=seed + motion_seed, geometry_context=memory,
                      geometry_context_keep_mask=torch.ones(b, n, dtype=torch.bool, device=observed.device))
        future, predicted_actions = model.rollout_shallow(observed, **kwargs)
        deep = model.da3.propagate_shallow_with_actions_grad(
            future, predicted_actions, decode_visuals=False, return_multi_level=True,
            gradient_checkpointing=model.deep_gradient_checkpointing, deep_temporal_causal_mask=True)
        pairs = deep.get("level_feats")
        if pairs is None or len(pairs) != len(self.residual_heads):
            raise RuntimeError("Candidate B needs every native DA3 deep output level")
        supported, completed, mixed = [], [], []
        for (patches, cls), current_level, residual_head, completion_head in zip(
                pairs, current["deep_levels"], self.residual_heads, self.completion_heads):
            patches = patches.reshape(b, n, 2*self.dim)
            source = current_level[:, 0, 0, prefix:]
            sup = w.to(source.dtype) @ source + residual_head(patches)
            new = completion_head(patches)
            mixture = rho[..., None].to(sup.dtype)*sup + (1-rho[..., None]).to(new.dtype)*new
            supported.append(sup)
            completed.append(new)
            # Prefix/registers are not transported. Official DPT consumes CLS
            # and patches separately; retain the native predicted CLS.
            padding = source.new_zeros(b, prefix-1, 2*self.dim)
            mixed.append(torch.cat((cls.reshape(b, 1, 2*self.dim), padding, mixture), 1)[:, None, None])
        uncertainty = None
        if self.error_head is not None:
            # Error head sees inference-available completion only. Neither its
            # input trunk nor its value confidence receives Action gradients.
            uncertainty = F.softplus(self.error_head(completed[-1][..., -self.dim:].detach()).float()[..., 0])
        confidence = None if uncertainty is None else torch.exp(-uncertainty.detach()/self.tau)
        # Queries come from VLA BEFORE Future DA3. Using its refined action
        # hidden here would let completion bypass its confidence-controlled
        # read through joint self-attention and the fusion's query input.
        query = proposal_tokens[:, 0, 0]
        r_cur = self.current_read(query, current_last, torch.ones_like(rho))
        r_sup = self.supported_read(query, supported[-1][..., -self.dim:], rho)
        r_new = self.completion_read(query, completed[-1][..., -self.dim:], 1-rho, confidence)
        fused = self.fusion(torch.cat((query, proposal[:, 0], r_cur, r_sup, r_new), -1))[:, None]
        # GT future conditions must not leak into the policy during geometry
        # pretraining. Keep a zero graph connection for DDP's unused detection.
        correction = model.parallel_action_correction_head(fused)
        if gt_geometry:
            correction = correction * 0.
        actions = proposal + correction
        decoded = decode_levels(model.da3, mixed, prefix)
        return {"future_shallow": future, "predicted_current_shallow": None,
                "actions_norm": actions, "direct_actions_norm": proposal,
                "refine_actions_norm": actions, "geometry_action_residual_norm": correction,
                "dual_refine_actions_norm": None, "dual_action_future_weight": None,
                "action_token_logits": None, "action_token_target_ids": None, "action_token_ids": None,
                "idm_features": None, "idm_actions_norm": None, "all_window_actions_norm": None,
                "current_geometry_features": current, "current_depth_output": current_decoded,
                "deep_joint_features": decoded, "stop_logits": None, "relative_pose": None,
                "depth_log_scale": None, "stop_future_gate": future.new_zeros(()),
                "residual_gate": model.residual_gate,
                "transport": {**transport, "supported_levels": supported, "completion_levels": completed,
                              "mixed_levels": mixed, "completion_uncertainty": uncertainty,
                              "completion_confidence": confidence, "supervision_gate": gate,
                              "geometry_actions_raw": geometry_actions, "proposal_actions_raw": proposal_raw}}


def transport_losses(output, teacher_levels, *, prefix, visual_dim):
    """Supervise final deep mixture and both branches in their support regions."""
    info = output["transport"]
    rho = info["rho"].detach().float()
    gate = info["supervision_gate"].detach().float()[:, None]
    values = []
    for sup, new, mixture, target in zip(info["supported_levels"], info["completion_levels"],
                                          info["mixed_levels"], teacher_levels):
        target = target[:, 0, 0, prefix:].detach().float()
        # Fixed per-example normalization does not increase the effective
        # weight as matched proposals become scarce.
        sup_error = (sup.float()-target).square().mean(-1)
        new_error = (new.float()-target).square().mean(-1)
        mix_error = (mixture[:, 0, 0, prefix:].float()-target).square().mean(-1)
        values.append(((mix_error + rho*sup_error + (1-rho)*new_error)*gate).mean()/2)
    total = torch.stack(values).mean()
    uncertainty = info["completion_uncertainty"]
    calibration = total.new_zeros(())
    error_target = None
    if uncertainty is not None:
        prediction = info["completion_levels"][-1][..., -visual_dim:].detach().float()
        target = teacher_levels[-1][:, 0, 0, prefix:, -visual_dim:].detach().float()
        # Feature scale is fixed by non-affine channel normalization. u is an
        # expected absolute feature error, not a claimed Bayesian variance.
        error_target = (F.layer_norm(prediction, (visual_dim,))-
                        F.layer_norm(target, (visual_dim,))).abs().mean(-1)
        weight = (1-rho)*gate
        calibration = ((uncertainty-error_target).square()*weight).mean()
    return {"geometry": total, "uncertainty": calibration, "error_target": error_target}


def initialize_pure_qwen_proposal(model, conditioner, normalizer, checkpoint):
    """Exact prompt-Pure-Qwen proposal transfer, leaving GAM/readouts new."""
    from robot.data.dataset import ActionNormalizer
    if checkpoint.get("format") != "uavflow_qwen_pure_v1":
        raise ValueError("--proposal-checkpoint requires a pure-Qwen checkpoint")
    expected_pose='numeric_token' if getattr(conditioner,'numeric_pose_token',False) else 'prompt'
    if checkpoint.get("config", {}).get("model", {}).get("pose_mode", "prompt") != expected_pose:
        raise ValueError(f'Proposal pose state-injection architecture must match {expected_pose}')
    if expected_pose=='numeric_token':
        import warnings
        warnings.warn('Proposal weight initialization only: historical Pure-Qwen runs used '
                      'preprocessed full-first-rotation pose. This does not repair their '
                      'train/test state mismatch or constitute exact yaw4d resume.',UserWarning)
    policy = model.transport_policy or getattr(model, 'provenance_correction', None) or getattr(model,'rgbwarp_correction',None)
    if (policy is None or model.parallel_vla_gfm_mode != "qwen_tokens"
            or model.parallel_action_position_mode != "decoder_only"):
        raise ValueError("Pure proposal transfer requires qwen_tokens and decoder_only")
    state = checkpoint["trainable_state"]
    encoded = {name: value for name, value in state.items() if name.startswith("encoder.") or name.startswith('state_encoder.')}
    trainable = {name: p for name, p in conditioner.named_parameters() if p.requires_grad}
    if set(encoded) != set(trainable):
        raise ValueError("Qwen trainable scope/slot embeddings differ from the proposal checkpoint")
    old_stats = ActionNormalizer({"old": checkpoint["action_stats"]}, default_key="old")
    probes = torch.stack((torch.full((4,), -1.), torch.zeros(4), torch.ones(4)))
    for key in normalizer.stats_by_key:
        if not torch.allclose(normalizer.denormalize(probes, stats_key=key),
                              old_stats.denormalize(probes), atol=1e-6, rtol=1e-5):
            raise ValueError("Proposal and GAM action normalization differ; do not reinterpret the decoder")
    with torch.no_grad():
        for name, value in encoded.items():
            trainable[name].copy_(value.to(trainable[name]))
    projection = {name.removeprefix("projection."): value for name, value in state.items() if name.startswith("projection.")}
    head = {name.removeprefix("head."): value for name, value in state.items() if name.startswith("head.")}
    policy.proposal_projection.load_state_dict(projection, strict=True)
    model.parallel_action_head.decoder.load_state_dict(head, strict=True)
