"""Frozen-DA3-shallow inverse-dynamics heads for UAV-Flow.

These heads are diagnostic probes.  They take two consecutive DA3 shallow
visual-token sets and regress the action connecting the two frames.  The DA3
backbone stays frozen; only this head is trained.
"""

from __future__ import annotations

import torch
from torch import nn


class DA3ShallowInverseMLP(nn.Module):
    """Predict action_t from two DA3 shallow visual-token frames.

    Args:
        token_dim: DA3 shallow token dimension.
        num_register_tokens: number of DA3 register tokens between CLS and
            patches in the layout [CLS, registers..., patches...].
        action_dim: output action dimension, e.g. 4 for yaw4d.
        hidden_dim: residual MLP hidden size.
        proj_dim: per-frame pooled feature projection size.
        num_blocks: number of residual MLP blocks after pair fusion.
        dropout: dropout inside residual blocks.

    Input:
        visual_tokens: (B, 2, V, N, D), from
            DA3GiantEncoder.encode_shallow_visual_slots()["visual_tokens"].

    Output:
        (B, action_dim).
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_register_tokens: int,
        action_dim: int = 4,
        hidden_dim: int = 1024,
        proj_dim: int = 1024,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.action_dim = int(action_dim)

        # Per frame we keep three coarse summaries.  This is intentionally
        # simple: if even this cannot overfit, the issue is unlikely to be a
        # fancy attention-head design problem.
        frame_dim = self.token_dim * 3  # CLS, register mean, patch mean
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, int(proj_dim)),
            nn.SiLU(inplace=True),
            nn.LayerNorm(int(proj_dim)),
        )

        pair_dim = int(proj_dim) * 4  # f0, f1, f1-f0, f0*f1
        self.in_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, int(hidden_dim)),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [_ResidualMLPBlock(int(hidden_dim), dropout=float(dropout)) for _ in range(int(num_blocks))]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def _pool_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D), layout [CLS, registers..., patches...]
        if tokens.ndim != 3:
            raise ValueError(f"Expected frame tokens (B,N,D), got {tuple(tokens.shape)}")
        cls = tokens[:, 0]
        reg_start = 1
        reg_end = 1 + max(0, self.num_register_tokens)
        if reg_end > reg_start and tokens.shape[1] >= reg_end:
            reg = tokens[:, reg_start:reg_end].mean(dim=1)
            patch_tokens = tokens[:, reg_end:]
        else:
            reg = torch.zeros_like(cls)
            patch_tokens = tokens[:, 1:]
        if patch_tokens.numel() == 0:
            patch = torch.zeros_like(cls)
        else:
            patch = patch_tokens.mean(dim=1)
        return torch.cat([cls, reg, patch], dim=-1)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 5 or visual_tokens.shape[1] < 2:
            raise ValueError(f"Expected visual_tokens (B,2,V,N,D), got {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Token dim mismatch: model token_dim={self.token_dim}, "
                f"input={visual_tokens.shape[-1]}"
            )
        # UAV-Flow is single-view today.  For safety, average views after DA3.
        vt = visual_tokens[:, :2].mean(dim=2)
        f0 = self.frame_proj(self._pool_frame(vt[:, 0]).float())
        f1 = self.frame_proj(self._pool_frame(vt[:, 1]).float())
        h = self.in_proj(torch.cat([f0, f1, f1 - f0, f0 * f1], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class DA3ShallowPatchInverseTransformer(nn.Module):
    """Patch-level two-frame inverse-dynamics head.

    Unlike ``DA3ShallowInverseMLP``, this head keeps the spatial patch grid.
    For every corresponding DA3 shallow patch position it builds a motion token
    from ``[p_t, p_{t+1}, p_{t+1}-p_t, p_t*p_{t+1}]``.  A small Transformer then
    aggregates those motion tokens into an action prediction.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_register_tokens: int,
        action_dim: int = 4,
        model_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cls_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.use_cls_tokens = bool(use_cls_tokens)

        pair_dim = self.token_dim * 4
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.model_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(self.model_dim),
        )
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.model_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(self.model_dim),
        )
        self.action_query = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        nn.init.normal_(self.action_query, std=0.02)
        self.type_embed = nn.Parameter(torch.zeros(1, 3, self.model_dim))
        nn.init.normal_(self.type_embed, std=0.02)

        ff_dim = int(round(self.model_dim * float(mlp_ratio)))
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, self.action_dim)

    def _split_frame(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tokens: (B, N, D), layout [CLS, registers..., patches...]
        if tokens.ndim != 3:
            raise ValueError(f"Expected frame tokens (B,N,D), got {tuple(tokens.shape)}")
        cls = tokens[:, :1]
        reg_start = 1
        reg_end = 1 + max(0, self.num_register_tokens)
        if reg_end > reg_start and tokens.shape[1] >= reg_end:
            regs = tokens[:, reg_start:reg_end]
            patches = tokens[:, reg_end:]
        else:
            regs = tokens[:, 1:1]
            patches = tokens[:, 1:]
        if patches.numel() == 0:
            raise ValueError(f"No patch tokens found in DA3 shallow tokens {tuple(tokens.shape)}")
        return cls, regs, patches

    @staticmethod
    def _pair_features(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return torch.cat([x0, x1, x1 - x0, x0 * x1], dim=-1)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 5 or visual_tokens.shape[1] < 2:
            raise ValueError(f"Expected visual_tokens (B,2,V,N,D), got {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"Token dim mismatch: model token_dim={self.token_dim}, "
                f"input={visual_tokens.shape[-1]}"
            )
        vt = visual_tokens[:, :2].mean(dim=2)
        cls0, regs0, patches0 = self._split_frame(vt[:, 0].float())
        cls1, regs1, patches1 = self._split_frame(vt[:, 1].float())
        if patches0.shape != patches1.shape:
            raise ValueError(f"Patch shape mismatch: {tuple(patches0.shape)} vs {tuple(patches1.shape)}")

        patch_tokens = self.patch_proj(self._pair_features(patches0, patches1))
        tokens = [
            self.action_query.expand(visual_tokens.shape[0], -1, -1) + self.type_embed[:, 0:1],
        ]
        if self.use_cls_tokens:
            cls_token = self.summary_proj(self._pair_features(cls0, cls1)) + self.type_embed[:, 1:2]
            tokens.append(cls_token)
            if regs0.numel() > 0:
                reg0 = regs0.mean(dim=1, keepdim=True)
                reg1 = regs1.mean(dim=1, keepdim=True)
                reg_token = self.summary_proj(self._pair_features(reg0, reg1)) + self.type_embed[:, 1:2]
                tokens.append(reg_token)
        tokens.append(patch_tokens + self.type_embed[:, 2:3])
        x = torch.cat(tokens, dim=1)
        x = self.encoder(x)
        return self.out(self.norm(x[:, 0]))


class DA3MultiLevelPatchInversePerceiver(nn.Module):
    """Multi-level DA3 inverse-dynamics head.

    Input is a dict with:
      - shallow: (B, 2, V, N, D_shallow)
      - deep_levels: list[(B, 2, V, N, D_deep)]

    Each level contributes patch-motion memory tokens.  A small set of learned
    latent tokens cross-attends to this memory, avoiding quadratic attention
    over all DA3 levels.
    """

    def __init__(
        self,
        *,
        shallow_dim: int,
        deep_dim: int,
        num_register_tokens: int,
        num_deep_levels: int = 4,
        action_dim: int = 4,
        model_dim: int = 384,
        num_latents: int = 64,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        include_cls: bool = True,
    ) -> None:
        super().__init__()
        self.shallow_dim = int(shallow_dim)
        self.deep_dim = int(deep_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.num_deep_levels = int(num_deep_levels)
        self.action_dim = int(action_dim)
        self.model_dim = int(model_dim)
        self.include_cls = bool(include_cls)

        self.shallow_proj = _MotionProjector(self.shallow_dim, self.model_dim)
        self.deep_proj = nn.ModuleList(
            [_MotionProjector(self.deep_dim, self.model_dim) for _ in range(self.num_deep_levels)]
        )
        self.level_embed = nn.Parameter(torch.zeros(1, 1 + self.num_deep_levels, self.model_dim))
        nn.init.normal_(self.level_embed, std=0.02)
        self.latents = nn.Parameter(torch.zeros(1, int(num_latents) + 1, self.model_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _PerceiverBlock(
                    dim=self.model_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, self.action_dim)

    def _split_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 2, V, N, D) -> averaged views, return cls and patches.
        if x.ndim != 5 or x.shape[1] < 2:
            raise ValueError(f"Expected level tokens (B,2,V,N,D), got {tuple(x.shape)}")
        y = x[:, :2].mean(dim=2).float()
        cls = y[:, :, :1]
        patch_start = 1 + max(0, self.num_register_tokens)
        patches = y[:, :, patch_start:]
        if patches.numel() == 0:
            raise ValueError(f"No patch tokens found in {tuple(x.shape)}")
        return cls, patches

    @staticmethod
    def _pair_features(x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, 0], x[:, 1]
        return torch.cat([x0, x1, x1 - x0, x0 * x1], dim=-1)

    def _level_memory(self, x: torch.Tensor, projector: nn.Module, level_idx: int) -> torch.Tensor:
        cls, patches = self._split_tokens(x)
        mem = projector(self._pair_features(patches))
        if self.include_cls:
            cls_mem = projector(self._pair_features(cls))
            mem = torch.cat([cls_mem, mem], dim=1)
        return mem + self.level_embed[:, level_idx : level_idx + 1]

    def forward(self, features: dict[str, object]) -> torch.Tensor:
        shallow = features.get("shallow")
        deep_levels = features.get("deep_levels")
        if not isinstance(shallow, torch.Tensor):
            raise TypeError("DA3MultiLevelPatchInversePerceiver expects features['shallow'] tensor.")
        if not isinstance(deep_levels, (list, tuple)):
            raise TypeError("DA3MultiLevelPatchInversePerceiver expects features['deep_levels'] list.")
        if len(deep_levels) != self.num_deep_levels:
            raise ValueError(f"Expected {self.num_deep_levels} deep levels, got {len(deep_levels)}.")
        memories = [self._level_memory(shallow, self.shallow_proj, 0)]
        for i, level in enumerate(deep_levels):
            if not isinstance(level, torch.Tensor):
                raise TypeError(f"deep_levels[{i}] is not a tensor.")
            memories.append(self._level_memory(level, self.deep_proj[i], i + 1))
        memory = torch.cat(memories, dim=1)
        latents = self.latents.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, memory)
        return self.out(self.norm(latents[:, 0]))


class DA3MultiLevelWindowInversePerceiver(nn.Module):
    """WorldVLN-style multi-frame inverse-dynamics head.

    This head consumes a short sequence of frozen DA3 features and predicts the
    action deltas between adjacent frames:

        input  : W frames of DA3 tokens
        output : W-1 actions

    For example, ``window_size=4`` mirrors WorldVLN's action decoder protocol:

        [f0, f1, f2, f3] -> [a0->1, a1->2, a2->3]

    Unlike ``DA3MultiLevelPatchInversePerceiver`` this does not collapse the
    input into a single pair-motion token set.  It keeps explicit temporal
    identity with learned time embeddings, then uses a small Perceiver stack so
    memory cost stays roughly linear in the number of DA3 levels/tokens.
    """

    def __init__(
        self,
        *,
        shallow_dim: int,
        deep_dim: int,
        num_register_tokens: int,
        num_deep_levels: int = 4,
        action_dim: int = 4,
        window_size: int = 4,
        model_dim: int = 384,
        num_latents: int = 64,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        include_cls: bool = True,
    ) -> None:
        super().__init__()
        self.shallow_dim = int(shallow_dim)
        self.deep_dim = int(deep_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.num_deep_levels = int(num_deep_levels)
        self.action_dim = int(action_dim)
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >=2, got {self.window_size}")
        self.num_actions = self.window_size - 1
        self.model_dim = int(model_dim)
        self.include_cls = bool(include_cls)

        self.shallow_proj = _TokenProjector(self.shallow_dim, self.model_dim)
        self.deep_proj = nn.ModuleList(
            [_TokenProjector(self.deep_dim, self.model_dim) for _ in range(self.num_deep_levels)]
        )
        self.level_embed = nn.Parameter(torch.zeros(1, 1 + self.num_deep_levels, self.model_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.window_size, self.model_dim))
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

        # First W-1 latents are action queries; the rest are scratch latents.
        self.latents = nn.Parameter(torch.zeros(1, self.num_actions + int(num_latents), self.model_dim))
        nn.init.normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _PerceiverBlock(
                    dim=self.model_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, self.action_dim)

    def _split_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, V, N, D) -> average views, keep [CLS? + patches].
        if x.ndim != 5 or int(x.shape[1]) < self.window_size:
            raise ValueError(
                f"Expected level tokens (B,{self.window_size},V,N,D), got {tuple(x.shape)}"
            )
        y = x[:, : self.window_size].mean(dim=2).float()
        patch_start = 1 + max(0, self.num_register_tokens)
        if self.include_cls:
            tokens = torch.cat([y[:, :, :1], y[:, :, patch_start:]], dim=2)
        else:
            tokens = y[:, :, patch_start:]
        if tokens.numel() == 0:
            raise ValueError(f"No tokens found in {tuple(x.shape)}")
        return tokens

    def _level_memory(self, x: torch.Tensor, projector: nn.Module, level_idx: int) -> torch.Tensor:
        tokens = projector(self._split_tokens(x))
        tokens = tokens + self.time_embed[:, : self.window_size].unsqueeze(2)
        tokens = tokens + self.level_embed[:, level_idx : level_idx + 1].unsqueeze(1)
        bsz, win, n_tok, dim = tokens.shape
        return tokens.reshape(bsz, win * n_tok, dim)

    def forward(self, features: dict[str, object]) -> torch.Tensor:
        shallow = features.get("shallow")
        deep_levels = features.get("deep_levels")
        if not isinstance(shallow, torch.Tensor):
            raise TypeError("DA3MultiLevelWindowInversePerceiver expects features['shallow'] tensor.")
        if not isinstance(deep_levels, (list, tuple)):
            raise TypeError("DA3MultiLevelWindowInversePerceiver expects features['deep_levels'] list.")
        if len(deep_levels) != self.num_deep_levels:
            raise ValueError(f"Expected {self.num_deep_levels} deep levels, got {len(deep_levels)}.")

        memories = [self._level_memory(shallow, self.shallow_proj, 0)]
        for i, level in enumerate(deep_levels):
            if not isinstance(level, torch.Tensor):
                raise TypeError(f"deep_levels[{i}] is not a tensor.")
            memories.append(self._level_memory(level, self.deep_proj[i], i + 1))
        memory = torch.cat(memories, dim=1)

        latents = self.latents.expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, memory)
        action_latents = self.norm(latents[:, : self.num_actions])
        return self.out(action_latents)


class DA3HierarchicalWindowInversePerceiver(nn.Module):
    """Hierarchical multi-frame IDM with per-level query tokens.

    Unlike ``DA3MultiLevelWindowInversePerceiver`` which flattens all level
    tokens into one shared memory, this head assigns ``num_queries_per_level``
    dedicated latent tokens to each DA3 level.  Each level independently
    cross-attends to its own tokens, producing a level summary.  All level
    summaries are then concatenated and refined by shared self-attention before
    predicting actions.

        shallow  ──cross-attn──→ N queries ──┐
        deep_0  ──cross-attn──→ N queries ──┤
        deep_1  ──cross-attn──→ N queries ──┼── concat ── self-attn ──→ actions
        deep_2  ──cross-attn──→ N queries ──┤
        deep_3  ──cross-attn──→ N queries ──┘
    """

    def __init__(
        self,
        *,
        shallow_dim: int,
        deep_dim: int,
        num_register_tokens: int,
        num_deep_levels: int = 4,
        action_dim: int = 4,
        window_size: int = 4,
        model_dim: int = 512,
        num_queries_per_level: int = 8,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        include_cls: bool = True,
        use_pos_embed: bool = False,
    ) -> None:
        super().__init__()
        self.shallow_dim = int(shallow_dim)
        self.deep_dim = int(deep_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.num_deep_levels = int(num_deep_levels)
        self.action_dim = int(action_dim)
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >=2, got {self.window_size}")
        self.num_actions = self.window_size - 1
        self.model_dim = int(model_dim)
        self.num_queries_per_level = int(num_queries_per_level)
        self.include_cls = bool(include_cls)
        self.use_pos_embed = bool(use_pos_embed)
        self.num_levels = 1 + self.num_deep_levels  # shallow + deep levels

        # Per-level token projectors (shared structure, separate weights)
        self.shallow_proj = _TokenProjector(self.shallow_dim, self.model_dim)
        self.deep_proj = nn.ModuleList(
            [_TokenProjector(self.deep_dim, self.model_dim) for _ in range(self.num_deep_levels)]
        )

        # Per-level learnable query tokens
        self.level_queries = nn.Parameter(
            torch.zeros(1, self.num_levels, self.num_queries_per_level, self.model_dim)
        )
        nn.init.normal_(self.level_queries, std=0.02)

        # Per-level cross-attention (independent Q/KV/O weights)
        self.level_cross_attn = nn.ModuleList([
            _CrossAttnBlock(dim=self.model_dim, num_heads=int(num_heads), dropout=float(dropout))
            for _ in range(self.num_levels)
        ])

        # Level embedding and time embedding
        self.level_embed = nn.Parameter(torch.zeros(1, self.num_levels, self.model_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.window_size, self.model_dim))
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

        # Optional 2D position embedding for patch tokens
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, 256, self.model_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)

        # Shared self-attention blocks over concatenated level summaries
        self.total_shared_tokens = self.num_levels * self.num_queries_per_level
        self.shared_blocks = nn.ModuleList([
            _PerceiverBlock(
                dim=self.model_dim,
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                dropout=float(dropout),
            )
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(self.model_dim)
        self.out = nn.Linear(self.model_dim, self.action_dim)

    def _split_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or int(x.shape[1]) < self.window_size:
            raise ValueError(
                f"Expected level tokens (B,{self.window_size},V,N,D), got {tuple(x.shape)}"
            )
        y = x[:, : self.window_size].mean(dim=2).float()
        patch_start = 1 + max(0, self.num_register_tokens)
        if self.include_cls:
            tokens = torch.cat([y[:, :, :1], y[:, :, patch_start:]], dim=2)
        else:
            tokens = y[:, :, patch_start:]
        return tokens

    def _level_memory(self, x: torch.Tensor, projector: nn.Module, level_idx: int) -> torch.Tensor:
        tokens = projector(self._split_tokens(x))
        tokens = tokens + self.time_embed[:, : self.window_size].unsqueeze(2)
        tokens = tokens + self.level_embed[:, level_idx : level_idx + 1].unsqueeze(1)
        bsz, win, n_tok, dim = tokens.shape
        if self.pos_embed is not None:
            prefix = 0 if not self.include_cls else 1
            n_patches = n_tok - prefix
            pos_up_to = min(n_patches, self.pos_embed.shape[2])
            if n_patches > 0:
                tokens[:, :, prefix : prefix + pos_up_to] = (
                    tokens[:, :, prefix : prefix + pos_up_to] + self.pos_embed[:, :, :pos_up_to].unsqueeze(1)
                )
        return tokens.reshape(bsz, win * n_tok, dim)

    def forward(self, features: dict[str, object]) -> torch.Tensor:
        shallow = features.get("shallow")
        deep_levels = features.get("deep_levels")
        if not isinstance(shallow, torch.Tensor):
            raise TypeError("DA3HierarchicalWindowInversePerceiver expects features['shallow'] tensor.")
        if not isinstance(deep_levels, (list, tuple)):
            raise TypeError("DA3HierarchicalWindowInversePerceiver expects features['deep_levels'] list.")
        if len(deep_levels) != self.num_deep_levels:
            raise ValueError(f"Expected {self.num_deep_levels} deep levels, got {len(deep_levels)}.")

        # Build per-level memory
        memories = [self._level_memory(shallow, self.shallow_proj, 0)]
        for i, level in enumerate(deep_levels):
            if not isinstance(level, torch.Tensor):
                raise TypeError(f"deep_levels[{i}] is not a tensor.")
            memories.append(self._level_memory(level, self.deep_proj[i], i + 1))

        # Per-level cross-attention: N independent queries per level
        bsz = shallow.shape[0]
        level_latents = []
        for lvl_idx, memory in enumerate(memories):
            queries = self.level_queries[:, lvl_idx].expand(bsz, -1, -1)
            latents = self.level_cross_attn[lvl_idx](queries, memory)
            level_latents.append(latents)

        # Concatenate all level summaries
        shared = torch.cat(level_latents, dim=1)

        # Shared self-attention refinement
        for block in self.shared_blocks:
            shared = block(shared, shared)  # self-attention: Q=K=V=shared

        # Predict actions from the first num_actions tokens of each level's summary
        # (take the first query from each level, which receives the strongest
        # cross-attention signal)
        action_tokens = shared[:, : self.num_levels * self.num_queries_per_level]
        action_tokens = action_tokens.reshape(bsz, self.num_levels, self.num_queries_per_level, self.model_dim)
        # Pool: mean over levels, take first num_actions queries
        pooled = action_tokens.mean(dim=1)[:, : self.num_actions]
        return self.out(self.norm(pooled))


class _CrossAttnBlock(nn.Module):
    """Single cross-attention block: cross-attn + FFN with residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(int(dim))
        self.kv_norm = nn.LayerNorm(int(dim))
        self.cross = nn.MultiheadAttention(
            int(dim), int(num_heads), dropout=float(dropout), batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(int(dim))
        hidden = int(round(int(dim) * 4.0))
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(dim)),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query)
        kv = self.kv_norm(memory)
        x = query + self.cross(q, kv, kv, need_weights=False)[0]
        return x + self.ffn(self.ffn_norm(x))


class _MotionProjector(nn.Module):
    def __init__(self, token_dim: int, model_dim: int) -> None:
        super().__init__()
        pair_dim = int(token_dim) * 4
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, int(model_dim)),
            nn.SiLU(inplace=True),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TokenProjector(nn.Module):
    def __init__(self, token_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(model_dim)),
            nn.SiLU(inplace=True),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PerceiverBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(int(dim))
        self.kv_norm = nn.LayerNorm(int(dim))
        self.cross = nn.MultiheadAttention(
            int(dim),
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(int(dim))
        self.self_attn = nn.MultiheadAttention(
            int(dim),
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(int(dim))
        hidden = int(round(int(dim) * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(dim)),
        )

    def forward(self, latents: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross(
            self.q_norm(latents),
            self.kv_norm(memory),
            self.kv_norm(memory),
            need_weights=False,
        )[0]
        latents = latents + self.self_attn(
            self.self_norm(latents),
            self.self_norm(latents),
            self.self_norm(latents),
            need_weights=False,
        )[0]
        return latents + self.ffn(self.ffn_norm(latents))


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(dim) * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim) * 4, int(dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
