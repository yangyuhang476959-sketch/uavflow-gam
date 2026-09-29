"""Semantic action-plan initializers and parallel continuous action heads.

Semantic memory is consumed before GAM's Future Predictor. Current geometric
memory is deliberately not consumed here: action tokens read it layer-by-layer
inside DA3's refine/deep stack.
"""
from __future__ import annotations

import torch
from torch import nn


def _append_safe_key(language, mask):
    batch, length = language.shape[:2]
    keep = (torch.ones(batch, length, device=language.device, dtype=torch.bool)
            if mask is None else mask.to(device=language.device, dtype=torch.bool))
    if keep.shape != (batch, length):
        raise ValueError(f"language mask {tuple(keep.shape)} != {(batch, length)}")
    language = torch.cat([language, torch.zeros_like(language[:, :1])], dim=1)
    keep = torch.cat([keep, ~keep.any(dim=1, keepdim=True)], dim=1)
    return language, keep


class _SemanticQueryBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.kv_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, query, memory, keep):
        kv = self.kv_norm(memory)
        read, _ = self.cross(self.q_norm(query), kv, kv,
                             key_padding_mask=~keep, need_weights=False)
        query = query + read
        q = self.self_norm(query)
        mixed, _ = self.self_attn(q, q, q, need_weights=False)
        query = query + mixed
        return query + self.ffn(self.ffn_norm(query))


class SemanticActionInitializer(nn.Module):
    """External K-query Semantic Bank reader (bank 1 of the two-bank design)."""
    def __init__(self, *, language_dim, output_dim, chunk_size,
                 width=512, heads=8, layers=1):
        super().__init__()
        if chunk_size <= 0 or width <= 0 or heads <= 0 or width % heads:
            raise ValueError("invalid semantic action initializer dimensions")
        self.chunk_size = int(chunk_size)
        self.memory = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, width))
        self.queries = nn.Parameter(torch.empty(1, self.chunk_size, width))
        self.action_type = nn.Parameter(torch.empty(1, 1, width))
        self.blocks = nn.ModuleList([_SemanticQueryBlock(width, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.action_type, std=0.02)

    def forward(self, language, language_mask=None):
        language, keep = _append_safe_key(language, language_mask)
        memory = self.memory(language)
        query = (self.queries + self.action_type).expand(language.shape[0], -1, -1)
        query = query.to(dtype=memory.dtype)
        for block in self.blocks:
            query = block(query, memory, keep)
        return self.output(query)


class QwenParallelActionTokenizer(SemanticActionInitializer):
    """Five bidirectionally mixed timestep tokens reading all Qwen states."""
    def __init__(self, **kwargs):
        kwargs.setdefault("layers", 2)
        super().__init__(**kwargs)


class QwenInternalActionProjector(nn.Module):
    """Project K action placeholders already contextualized inside Qwen."""
    def __init__(self, *, language_dim, output_dim, chunk_size,
                 width=512, heads=8, layers=0, **_):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.layers = int(layers)
        if self.layers < 0:
            raise ValueError("post-Qwen bidirectional layer count must be non-negative")
        if self.layers:
            self.input = nn.Sequential(
                nn.LayerNorm(language_dim), nn.Linear(language_dim, width)
            )
            self.bidirectional = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=width, nhead=heads, dim_feedforward=4 * width,
                    dropout=0.0, activation="gelu", batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.layers)
            ])
            self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))
        else:
            self.input = nn.Identity()
            self.bidirectional = nn.ModuleList()
            self.output = nn.Sequential(
                nn.LayerNorm(language_dim), nn.Linear(language_dim, output_dim)
            )

    def forward(self, language, language_mask=None):
        if language.shape[1] != self.chunk_size:
            raise ValueError(
                f"internal Qwen mode expects {self.chunk_size} action states, "
                f"got {language.shape[1]}"
            )
        if language_mask is not None and not bool(language_mask.bool().all().item()):
            raise ValueError("internal Qwen action placeholders may not be padded")
        tokens = self.input(language)
        for block in self.bidirectional:
            tokens = block(tokens)
        return self.output(tokens)


class OFTDimensionActionTokenizer(nn.Module):
    """OFT-style K*A placeholders, exposed both directly and as K GFM tokens."""
    def __init__(self, *, language_dim, output_dim, chunk_size, action_dim,
                 width=512, heads=8, layers=2):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.memory = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, width))
        self.placeholders = nn.Parameter(
            torch.empty(1, self.chunk_size * self.action_dim, width)
        )
        self.blocks = nn.ModuleList([_SemanticQueryBlock(width, heads) for _ in range(layers)])
        self.plan_projection = nn.Sequential(
            nn.LayerNorm(self.action_dim * width),
            nn.Linear(self.action_dim * width, output_dim),
        )
        self.direct_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.normal_(self.placeholders, std=0.02)

    def forward(self, language, language_mask=None):
        language, keep = _append_safe_key(language, language_mask)
        memory = self.memory(language)
        tokens = self.placeholders.expand(language.shape[0], -1, -1).to(memory.dtype)
        for block in self.blocks:
            tokens = block(tokens, memory, keep)
        grouped = tokens.reshape(
            language.shape[0], self.chunk_size, self.action_dim * tokens.shape[-1]
        )
        return {
            "dimension_tokens": tokens,
            "plan_tokens": self.plan_projection(grouped),
            "direct_actions_norm": self.direct_projection(tokens).reshape(
                language.shape[0], self.chunk_size, self.action_dim
            ),
        }


class OFTInternalActionProjector(nn.Module):
    """Decode Qwen placeholders after OFT-style bidirectional action mixing."""
    def __init__(self, *, language_dim, output_dim, chunk_size, action_dim,
                 width=512, heads=8, layers=2, **_):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.input = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, width))
        self.bidirectional = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=4 * width,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(layers)
        ])
        self.plan_projection = nn.Sequential(
            nn.LayerNorm(self.action_dim * width),
            nn.Linear(self.action_dim * width, output_dim),
        )
        self.direct_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 1)
        )

    def forward(self, language, language_mask=None):
        expected = self.chunk_size * self.action_dim
        if language.shape[1] != expected:
            raise ValueError(f"internal OFT mode expects {expected} states, got {language.shape[1]}")
        tokens = self.input(language)
        for block in self.bidirectional:
            tokens = block(tokens)
        grouped = tokens.reshape(
            tokens.shape[0], self.chunk_size, self.action_dim * tokens.shape[-1]
        )
        return {
            "dimension_tokens": tokens,
            "plan_tokens": self.plan_projection(grouped),
            "direct_actions_norm": self.direct_projection(tokens).reshape(
                tokens.shape[0], self.chunk_size, self.action_dim
            ),
        }


SemanticGeometryActionBridge = SemanticActionInitializer


class ParallelContinuousActionHead(nn.Module):
    """Shared token-wise head: [B,T,V,K,D] -> [B,T,K,A]."""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim), nn.GELU(),
            nn.Linear(input_dim, input_dim), nn.GELU(), nn.Linear(input_dim, action_dim),
        )

    def forward(self, tokens):
        if tokens.ndim == 5:
            tokens = tokens.mean(dim=2)
        elif tokens.ndim != 4:
            raise ValueError(
                "parallel action tokens must be [B,T,V,K,D] or [B,T,K,D], got "
                f"{tuple(tokens.shape)}"
            )
        return self.model(tokens)
