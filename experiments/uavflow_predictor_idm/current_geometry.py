"""One observation-to-action read; no map, extra geometry loss or future GT."""
import torch
from torch import nn


class CurrentGeometryRead(nn.Module):
    """[B,H,V,D] queries read matching [B,H,V,N,2D] current patches.

    Time and view are folded into batch, never attended across. The caller
    currently restricts this ablation to H=1 because DA3's observed deep pass
    itself is not temporally causal. A small residual gate preserves the
    existing action route while allowing gradients into the new branch.
    """
    def __init__(self, action_dim, width=256, heads=8, layer_indices=None,
                 gate_init=1e-3, memory_dim=None):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("current_geometry width must be positive and divisible by heads")
        self.memory_dim = int(2 * action_dim if memory_dim is None else memory_dim)
        if self.memory_dim <= 0:
            raise ValueError("current_geometry memory_dim must be positive")
        self.q = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, width))
        self.memory = nn.Sequential(
            nn.LayerNorm(self.memory_dim), nn.Linear(self.memory_dim, width)
        )
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, action_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.layer_gates = nn.ParameterDict({
            str(int(index)): nn.Parameter(torch.tensor(0.0))
            for index in (layer_indices or [])
        })

    def forward(self, actions, patches, layer_index=None):
        legacy = actions.ndim == 4
        if legacy:
            actions = actions.unsqueeze(-2)
        if actions.ndim != 5 or patches.ndim != 5 or actions.shape[:3] != patches.shape[:3]:
            raise ValueError(
                "Expected [B,H,V,D] or [B,H,V,K,D] actions and "
                "matching [B,H,V,N,2D] patches"
            )
        if patches.shape[-2] == 0:
            raise ValueError("Current geometry memory must contain visual patches")
        batch_views = actions.shape[0] * actions.shape[1] * actions.shape[2]
        action_count = actions.shape[-2]
        q = self.q(actions).reshape(batch_views, action_count, self.attention.embed_dim)
        memory = self.memory(patches).reshape(batch_views, patches.shape[-2], -1)
        read, _ = self.attention(q, memory, memory, need_weights=False)
        gate = self.gate + (
            self.layer_gates[str(int(layer_index))]
            if layer_index is not None and str(int(layer_index)) in self.layer_gates
            else 0.0
        )
        updated = actions + gate * self.output(read).reshape_as(actions)
        return updated.squeeze(-2) if legacy else updated


def load_current_geometry_read(model, checkpoint, *, required):
    """Used for exact resume and Stage-1 -> Stage-2 transfer."""
    if model.current_geometry_read is None:
        return
    saved_mode = checkpoint.get("current_geometry_read_mode", "terminal")
    if saved_mode != model.current_geometry_read_mode:
        raise ValueError("Current-geometry read mode changed; do not resume terminal CA as per-layer CA")
    saved_bank_mode = checkpoint.get(
        "current_geometry_bank_mode", "output_current"
    )
    if saved_bank_mode != model.current_geometry_bank_mode:
        raise ValueError(
            "Current-geometry bank mode changed; do not resume incompatible "
            f"{saved_bank_mode!r} weights as {model.current_geometry_bank_mode!r}"
        )
    state = checkpoint.get("current_geometry_read")
    if state is None:
        if required:
            raise KeyError("Current-geometry checkpoint is missing current_geometry_read")
        return
    model.current_geometry_read.load_state_dict(state, strict=True)
