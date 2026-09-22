"""Small support modules for the depth-architecture ablation, no GT inputs."""
import torch
from torch import nn

GEOMETRY_ARCHITECTURES = {
    "legacy", "current_prediction", "direct_current", "dual_observed",
    "dual_predicted", "dual_action_bridge",
}
DUAL_ARCHITECTURES = {"dual_observed", "dual_predicted", "dual_action_bridge"}


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
    return {
        "mode": model.geometry_architecture,
        "direct_current_seed": (None if model.direct_current_seed is None
                                else model.direct_current_seed.state_dict()),
        "prediction_roles": (None if model.prediction_roles is None
                             else model.prediction_roles.detach().cpu()),
    }


def load_architecture_state(model, checkpoint):
    state = checkpoint.get("geometry_architecture_state")
    if state is None:
        if model.geometry_architecture != "legacy":
            raise KeyError("Missing geometry_architecture_state; do not resume an old depth architecture")
        return
    if state["mode"] != model.geometry_architecture:
        raise ValueError("Geometry architecture changed; checkpoint is not compatible")
    if model.direct_current_seed is not None:
        model.direct_current_seed.load_state_dict(state["direct_current_seed"], strict=True)
    if model.prediction_roles is not None:
        with torch.no_grad():
            model.prediction_roles.copy_(state["prediction_roles"])
