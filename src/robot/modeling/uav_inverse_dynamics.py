"""Lightweight two-frame inverse-dynamics baselines for UAV-Flow.

These models are intentionally independent from GAM/DA3.  They are diagnostic
baselines: given two consecutive RGB observations, predict the action that
connects them.  This is *not* a deployable closed-loop policy because it uses
the next frame, but it is the cleanest way to test whether UAV-Flow's visual
transition and action labels are mutually learnable.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallImageEncoder(nn.Module):
    """Small ConvNet for 224x224 RGB frames.

    The design is deliberately boring: strided conv blocks followed by global
    pooling.  For a sanity baseline, boring is good; if this cannot overfit a
    tiny set, the problem is likely data/target alignment rather than GAM.
    """

    def __init__(self, in_channels: int = 3, width: int = 64, out_dim: int = 512):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 4]
        layers: list[nn.Module] = []
        c = in_channels
        for i, ch in enumerate(channels):
            layers.extend(
                [
                    nn.Conv2d(c, ch, kernel_size=5 if i == 0 else 3, stride=2, padding=2 if i == 0 else 1),
                    nn.GroupNorm(num_groups=min(16, ch), num_channels=ch),
                    nn.SiLU(inplace=True),
                ]
            )
            c = ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        return self.proj(self.pool(self.net(x)))


class TwoFrameInverseMLP(nn.Module):
    """Predict action_t from (image_t, image_{t+1}).

    Args:
        action_dim: output action dimensions, e.g. 4 for UAV yaw4d.
        chunk_size: output chunk size.  The common sanity setting is 1.
        encoder_dim: per-frame CNN embedding dimension.
        hidden_dim: MLP hidden dimension.
        num_blocks: number of residual MLP blocks.
        use_delta_feature: concatenate f1-f0 in addition to f0/f1.
    """

    def __init__(
        self,
        action_dim: int = 4,
        chunk_size: int = 1,
        encoder_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 3,
        cnn_width: int = 64,
        use_delta_feature: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.use_delta_feature = bool(use_delta_feature)
        self.encoder = SmallImageEncoder(in_channels=3, width=int(cnn_width), out_dim=int(encoder_dim))
        in_dim = int(encoder_dim) * (3 if self.use_delta_feature else 2)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([_MLPResidualBlock(int(hidden_dim)) for _ in range(int(num_blocks))])
        self.out = nn.Linear(int(hidden_dim), self.chunk_size * self.action_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            frames: (B, 2, 3, H, W) or (B, 2, V, 3, H, W).  If V>1, views are
                averaged after per-frame encoding.

        Returns:
            (B, action_dim) when chunk_size=1, else (B, chunk_size, action_dim).
        """
        if frames.ndim == 6:
            b, t, v, c, h, w = frames.shape
            if t != 2:
                raise ValueError(f"TwoFrameInverseMLP expects exactly 2 frames, got T={t}.")
            x = frames.reshape(b * t * v, c, h, w)
            feat = self.encoder(x).reshape(b, t, v, -1).mean(dim=2)
        elif frames.ndim == 5:
            b, t, c, h, w = frames.shape
            if t != 2:
                raise ValueError(f"TwoFrameInverseMLP expects exactly 2 frames, got T={t}.")
            feat = self.encoder(frames.reshape(b * t, c, h, w)).reshape(b, t, -1)
        else:
            raise ValueError(f"Expected frames with 5 or 6 dims, got {tuple(frames.shape)}.")

        f0, f1 = feat[:, 0], feat[:, 1]
        if self.use_delta_feature:
            z = torch.cat([f0, f1, f1 - f0], dim=-1)
        else:
            z = torch.cat([f0, f1], dim=-1)
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h)
        out = self.out(h).reshape(frames.shape[0], self.chunk_size, self.action_dim)
        return out[:, 0] if self.chunk_size == 1 else out


class MultiFrameInverseMLP(nn.Module):
    """Predict adjacent actions from a short RGB frame window.

    This is the image-space counterpart of the WorldVLN-style diagnostic:

        W frames -> W-1 actions

    For ``window_size=4``:

        [f0, f1, f2, f3] -> [a0->1, a1->2, a2->3]

    The model encodes each RGB frame with the same small CNN, adds a learned
    temporal embedding, runs a lightweight temporal Transformer over frame
    tokens, and predicts one action from each adjacent pair of temporal states.
    """

    def __init__(
        self,
        action_dim: int = 4,
        window_size: int = 4,
        encoder_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 3,
        cnn_width: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >=2, got {self.window_size}")
        self.num_actions = self.window_size - 1
        self.encoder_dim = int(encoder_dim)
        self.encoder = SmallImageEncoder(in_channels=3, width=int(cnn_width), out_dim=self.encoder_dim)
        self.time_embed = nn.Parameter(torch.zeros(1, self.window_size, self.encoder_dim))
        nn.init.normal_(self.time_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=int(num_blocks))
        pair_dim = self.encoder_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            frames: (B,W,3,H,W) or (B,W,V,3,H,W).  If V>1, views are averaged
                after per-frame encoding.

        Returns:
            (B, W-1, action_dim).
        """
        if frames.ndim == 6:
            b, t, v, c, h, w = frames.shape
            if int(t) != self.window_size:
                raise ValueError(f"MultiFrameInverseMLP expects T={self.window_size}, got T={t}.")
            feat = self.encoder(frames.reshape(b * t * v, c, h, w)).reshape(b, t, v, -1).mean(dim=2)
        elif frames.ndim == 5:
            b, t, c, h, w = frames.shape
            if int(t) != self.window_size:
                raise ValueError(f"MultiFrameInverseMLP expects T={self.window_size}, got T={t}.")
            feat = self.encoder(frames.reshape(b * t, c, h, w)).reshape(b, t, -1)
        else:
            raise ValueError(f"Expected frames with 5 or 6 dims, got {tuple(frames.shape)}.")

        z = self.temporal(feat + self.time_embed[:, : self.window_size])
        z0, z1 = z[:, :-1], z[:, 1:]
        pair = torch.cat([z0, z1, z1 - z0, z0 * z1], dim=-1)
        return self.head(pair)


class FrameStackInverseCNN(nn.Module):
    """Spatial RGB-frame-stack inverse dynamics baseline.

    ``MultiFrameInverseMLP`` pools each frame before comparing frames, which can
    erase tiny optical-flow-like changes.  This baseline keeps spatial
    differences visible to the first convolution by stacking raw frames and
    adjacent frame differences along the channel dimension:

        input channels = frames + diffs + abs(diffs)

    For W=4 this is ``3 * (4 + 3 + 3) = 30`` channels.
    """

    def __init__(
        self,
        action_dim: int = 4,
        window_size: int = 4,
        encoder_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 3,
        cnn_width: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >=2, got {self.window_size}")
        self.num_actions = self.window_size - 1
        in_channels = 3 * (self.window_size + 2 * self.num_actions)
        self.encoder = SmallImageEncoder(in_channels=in_channels, width=int(cnn_width), out_dim=int(encoder_dim))
        self.in_proj = nn.Sequential(
            nn.Linear(int(encoder_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([_MLPResidualBlock(int(hidden_dim)) for _ in range(int(num_blocks))])
        self.out = nn.Linear(int(hidden_dim), self.num_actions * self.action_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 6:
            # (B,W,V,3,H,W) -> average views in RGB space; UAV-Flow is V=1.
            frames = frames.mean(dim=2)
        if frames.ndim != 5:
            raise ValueError(f"Expected frames (B,W,3,H,W) or (B,W,V,3,H,W), got {tuple(frames.shape)}.")
        b, t, c, h, w = frames.shape
        if int(t) != self.window_size or int(c) != 3:
            raise ValueError(f"Expected frames (B,{self.window_size},3,H,W), got {tuple(frames.shape)}.")
        diffs = frames[:, 1:] - frames[:, :-1]
        stack = torch.cat(
            [
                frames.reshape(b, self.window_size * 3, h, w),
                diffs.reshape(b, self.num_actions * 3, h, w),
                diffs.abs().reshape(b, self.num_actions * 3, h, w),
            ],
            dim=1,
        )
        z = self.in_proj(self.encoder(stack))
        for block in self.blocks:
            z = block(z)
        return self.out(z).reshape(b, self.num_actions, self.action_dim)


class _MLPResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
