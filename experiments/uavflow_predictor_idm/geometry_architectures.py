"""Small support modules for the depth-architecture ablation, no GT inputs."""
import torch
from torch import nn

GEOMETRY_ARCHITECTURES = {
    "legacy", "current_prediction", "direct_current", "dual_observed",
    "dual_predicted", "dual_action_bridge", "dual_vla_gfm",
}
DUAL_ARCHITECTURES = {"dual_observed", "dual_predicted", "dual_action_bridge"}


class DualActionFusion(nn.Module):
    """Learn a per-plan-step Current/Future mixture instead of a fixed mean.

    The scalar gate is shared across hidden channels but differs per sample and
    temporal prediction step.  A zero-initialized final layer starts from the
    legacy 50/50 mixture while allowing training to select the reliable branch.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        width = max(int(hidden_dim) // 4, 128)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * int(hidden_dim)),
            nn.Linear(2 * int(hidden_dim), width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, current: torch.Tensor, future: torch.Tensor):
        if current.shape != future.shape or current.ndim != 3:
            raise ValueError(
                "Dual action branches must be matching [B,T,D] tensors; got "
                f"{tuple(current.shape)} and {tuple(future.shape)}"
            )
        future_weight = self.gate(torch.cat([current, future], dim=-1)).sigmoid()
        fused = (1.0 - future_weight) * current + future_weight * future
        return fused, future_weight


class DirectCurrentActionSeed(nn.Module):
    """No causal Predictor: an action query reads F0/Ft and instruction.

    Visual shallow tokens are passed unchanged to DA3 deep. Only this small
    query adapter injects language/reference information; no future feature
    is reconstructed. True in language_mask means valid, matching conditioning.
    """
    def __init__(self, visual_dim, language_dim, width=256, heads=8):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, width))
        self.language = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, width))
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.reference_role = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.visual_read = nn.MultiheadAttention(width, heads, batch_first=True)
        self.language_read = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, visual_dim)

    def forward(self, observed, reference, language, language_mask):
        b, h, v, n, d = observed.shape
        if h != 1 or v != 1:
            raise ValueError("direct_current requires H=1 and one physical camera")
        memory = self.visual(observed[:, 0, 0])
        if reference is not None:
            memory = torch.cat([self.visual(reference[:, 0, 0]) + self.reference_role, memory], 1)
        q = self.query.expand(b, -1, -1)
        q = q + self.visual_read(q, memory, memory, need_weights=False)[0]
        text = self.language(language)
        keep = (torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
                if language_mask is None else language_mask.bool())
        # A zero dummy key keeps all-masked inputs finite without exposing padding.
        text = torch.cat([text, torch.zeros_like(text[:, :1])], 1)
        keep = torch.cat([keep, ~keep.any(1, keepdim=True)], 1)
        q = q + self.language_read(q, text, text, key_padding_mask=~keep, need_weights=False)[0]
        return self.output(self.norm(q)).reshape(b, 1, 1, d)


def select_dual_view(outputs, batch_size, view):
    """Select one view from DA3's batched DPT tensors and flattened actions."""
    selected = {}
    for name, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Unsupported dual output {name}; handle its view axis explicitly")
        if name == "action_tokens":
            selected[name] = value.reshape(batch_size, 2, -1)[:, view]
        elif value.ndim >= 2 and value.shape[:2] == (batch_size, 2):
            selected[name] = value[:, view:view + 1]
        elif value.shape[0] == batch_size * 2:
            selected[name] = value.reshape(batch_size, 2, *value.shape[1:])[:, view]
        else:
            raise ValueError(f"Unknown DA3 dual layout for {name}: {tuple(value.shape)}")
    return selected


def architecture_state(model):
    state = {
        "mode": model.geometry_architecture,
        "direct_current_seed": (None if model.direct_current_seed is None
                                else model.direct_current_seed.state_dict()),
        "prediction_roles": (None if model.prediction_roles is None
                             else model.prediction_roles.detach().cpu()),
        "dual_action_fusion": (
            None if getattr(model, "dual_action_fusion", None) is None
            else model.dual_action_fusion.state_dict()
        ),
        "deep_action_step_embed": (
            None if getattr(model, "deep_action_step_embed", None) is None
            else model.deep_action_step_embed.detach().cpu()
        ),
        "parallel_action_decode_mode": getattr(
            model, "parallel_action_decode_mode", "full"
        ),
        "parallel_action_position_mode": getattr(model, "parallel_action_position_mode", "legacy"),
        "action_residual_gate_logit": (None if model.action_residual_gate_logit is None
                                       else model.action_residual_gate_logit.detach().cpu()),
    }
    transport = getattr(model, "transport_policy", None)
    rgbwarp=getattr(model,'rgbwarp_correction',None)
    if rgbwarp is not None:
        state['rgbwarp_correction']={'config':rgbwarp.config,'weights':rgbwarp.state_dict()}
    from .provenance_correction import checkpoint_state
    correction = checkpoint_state(model)
    if correction is not None:
        state['provenance_correction'] = correction
    if transport is not None:
        state["support_aware_transport"] = {"config": transport.config, "weights": transport.state_dict()}
    return state


def load_architecture_state(model, checkpoint, *, allow_new_transport=False):
    state = checkpoint.get("geometry_architecture_state")
    if state is None:
        if model.geometry_architecture != "legacy":
            raise KeyError("Missing geometry_architecture_state; do not resume an old depth architecture")
        return
    if state["mode"] != model.geometry_architecture:
        raise ValueError("Geometry architecture changed; checkpoint is not compatible")
    saved_action_gate=state.get('action_residual_gate_logit')
    if (model.action_residual_gate_logit is None)!=(saved_action_gate is None):
        raise ValueError('Action residual gating changed; checkpoint is not an exact resume')
    if saved_action_gate is not None:
        with torch.no_grad():
            model.action_residual_gate_logit.copy_(saved_action_gate.to(model.action_residual_gate_logit))
    rgbwarp=getattr(model,'rgbwarp_correction',None)
    saved_rgb=state.get('rgbwarp_correction')
    if (rgbwarp is None)!=(saved_rgb is None):
        raise ValueError('RGB-warp topology changed; no silent checkpoint reinterpretation')
    if rgbwarp is not None:
        if rgbwarp.config!=saved_rgb['config']:
            raise ValueError('RGB-warp settings changed during checkpoint load')
        loaded=rgbwarp.load_state_dict(saved_rgb['weights'],strict=True)
        print(f'RGB-warp load: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}',flush=True)
    from .provenance_correction import load_checkpoint_state
    load_checkpoint_state(model, state.get('provenance_correction'), initialize=allow_new_transport)
    transport = getattr(model, "transport_policy", None)
    saved_transport = state.get("support_aware_transport")
    if transport is None and saved_transport is not None:
        raise ValueError("Candidate B checkpoint requires support_aware_transport")
    if transport is not None:
        if saved_transport is None:
            if not allow_new_transport:
                raise KeyError("Candidate B resume checkpoint has no support_aware_transport state")
        else:
            # GT->proposal is an explicitly requested NEW training stage, not
            # an exact resume. Other storage/tau/render semantics stay fixed.
            saved_cfg, current_cfg = dict(saved_transport["config"]), dict(transport.config)
            adding_scalar = False
            if allow_new_transport:
                saved_cfg.pop("geometry_action_source", None)
                current_cfg.pop("geometry_action_source", None)
                adding_scalar = (saved_cfg.get("uncertainty", "none") == "none"
                                 and current_cfg.get("uncertainty", "none") == "scalar")
                if adding_scalar:
                    saved_cfg.pop("uncertainty", None)
                    current_cfg.pop("uncertainty", None)
            if saved_cfg != current_cfg:
                raise ValueError("Candidate B transport configuration changed")
            missing = transport.load_state_dict(saved_transport["weights"], strict=not adding_scalar)
            if adding_scalar and (missing.unexpected_keys or any(
                    not key.startswith("error_head.") for key in missing.missing_keys)):
                raise ValueError("B->C transfer may initialize only the scalar error head")
    if state.get("parallel_action_position_mode", "legacy") != getattr(model, "parallel_action_position_mode", "legacy"):
        raise ValueError("Parallel action position mode changed; checkpoint is not compatible")
    saved_decode = state.get("parallel_action_decode_mode", "full")
    current_decode = getattr(model, "parallel_action_decode_mode", "full")
    if saved_decode != current_decode:
        raise ValueError(
            "Parallel action decode mode changed: "
            f"checkpoint={saved_decode}, current={current_decode}"
        )
    if model.direct_current_seed is not None:
        model.direct_current_seed.load_state_dict(state["direct_current_seed"], strict=True)
    if model.prediction_roles is not None:
        with torch.no_grad():
            model.prediction_roles.copy_(state["prediction_roles"])
    if getattr(model, "dual_action_fusion", None) is not None:
        fusion_state = state.get("dual_action_fusion")
        # Old dual-view checkpoints used an exact 50/50 mean.  The fusion
        # module is zero-initialized to that same behavior, so read-only
        # evaluation/init transfer remains backward compatible.
        if fusion_state is not None:
            model.dual_action_fusion.load_state_dict(fusion_state, strict=True)
    if getattr(model, "deep_action_step_embed", None) is not None:
        step_embed = state.get("deep_action_step_embed")
        if step_embed is None:
            raise KeyError("Parallel VLA-GFM checkpoint is missing deep_action_step_embed")
        with torch.no_grad():
            model.deep_action_step_embed.copy_(step_embed)
