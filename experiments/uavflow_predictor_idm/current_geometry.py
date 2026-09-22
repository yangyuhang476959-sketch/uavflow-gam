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
    def __init__(self, action_dim, width=256, heads=8):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("current_geometry width must be positive and divisible by heads")
        self.q = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, width))
        self.memory = nn.Sequential(nn.LayerNorm(2 * action_dim), nn.Linear(2 * action_dim, width))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Linear(width, action_dim)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, actions, patches):
        if actions.ndim != 4 or patches.ndim != 5 or actions.shape[:3] != patches.shape[:3]:
            raise ValueError("Expected matching [B,H,V,D] actions and [B,H,V,N,2D] patches")
        if patches.shape[-2] == 0:
            raise ValueError("Current geometry memory must contain visual patches")
        q = self.q(actions).reshape(-1, 1, self.attention.embed_dim)
        memory = self.memory(patches).reshape(q.shape[0], patches.shape[-2], -1)
        read, _ = self.attention(q, memory, memory, need_weights=False)
        return actions + self.gate * self.output(read).reshape_as(actions)


def load_current_geometry_read(model, checkpoint, *, required):
    """Used for exact resume and Stage-1 -> Stage-2 transfer."""
    if model.current_geometry_read is None:
        return
    state = checkpoint.get("current_geometry_read")
    if state is None:
        if required:
            raise KeyError("Current-geometry checkpoint is missing current_geometry_read")
        return
    model.current_geometry_read.load_state_dict(state, strict=True)
