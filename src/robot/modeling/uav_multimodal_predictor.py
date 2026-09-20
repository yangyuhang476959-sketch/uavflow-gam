"""GAM-aligned geometry/DINO/action-separated predictor for UAV-Flow.

The original GAM puts DA3 visual, proprio, and previous-action tokens in one
Transformer sequence.  This variant keeps GAM's token/position/mask semantics
but separates the parameters into three streams::

    geometry (DA3 shallow tokens) <-> action (proprio + previous action)
                                      <-> DINO (semantic patch tokens)

There is deliberately no direct geometry-DINO attention edge.  Symbols used
throughout this file:
    B: batch size, H: observed timesteps, V: camera views,
    Pg: DA3 tokens/view (CLS + registers + patches),
    Pd: DINO patches/view, D: predictor hidden width.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .future_predictor import (
    BNMLPProjector,
    LNMLPProjector,
    LayerScale,
    PlainTextCrossBlock,
    QKNormCrossAttention,
    ShallowRoPE,
    _HAS_FLEX,
    _create_block_mask,
    _make_block_causal_mask_mod,
    _make_rmsnorm,
)


def _dense_causal_mask(steps: int, width: int, device: torch.device) -> torch.Tensor:
    """Square block-causal mask for one stream, shape ``[1,1,H*W,H*W]``.

    Tokens inside the same timestep are fully connected; a timestep may read
    itself and earlier timesteps, but never a future timestep.
    """
    times = torch.arange(steps, device=device).repeat_interleave(width)
    return (times[:, None] >= times[None, :]).unsqueeze(0).unsqueeze(0)


def _cross_causal_mask(
    steps: int,
    query_width: int,
    memory_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Rectangular causal mask for cross-stream attention.

    ``query_width`` and ``memory_width`` differ, e.g. 257 geometry tokens ask
    questions of two action tokens at every timestep.
    """
    query_times = torch.arange(steps, device=device).repeat_interleave(query_width)
    memory_times = torch.arange(steps, device=device).repeat_interleave(memory_width)
    return (query_times[:, None] >= memory_times[None, :]).unsqueeze(0).unsqueeze(0)


def _merge_key_keep(mask: torch.Tensor, key_keep: Optional[torch.Tensor]) -> torch.Tensor:
    """Prevent attention from reading padded/invalid memory tokens."""
    if key_keep is None:
        return mask
    return mask & key_keep[:, None, None, :].to(device=mask.device, dtype=torch.bool)


class CausalCrossAttention(nn.Module):
    """One directed cross-stream edge with its own Q/KV/O parameters.

    This reuses GAM's QK-normalized attention projections and zero-initialized
    LayerScale.  Consequently a bridge starts as an identity mapping and learns
    how strongly one modality should affect another.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = _make_rmsnorm(dim)
        self.norm_memory = _make_rmsnorm(dim)
        self.attn = QKNormCrossAttention(dim, heads, dropout)
        self.gate = LayerScale(dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        query_rope: ShallowRoPE,
        memory_rope: ShallowRoPE,
        query_positions: torch.Tensor,
        memory_positions: torch.Tensor,
    ) -> torch.Tensor:
        # QKNormCrossAttention's public keep-mask is for padding-only masks.
        # These bridges need a rectangular timestep-causal mask, so use its
        # projections while preserving the exact GAM QK-norm convention.
        q_in = self.norm_q(query)
        kv_in = self.norm_memory(memory)
        b, lq, dim = q_in.shape
        lk = kv_in.shape[1]
        attn = self.attn
        q = attn.q_proj(q_in).reshape(b, lq, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        kv = attn.kv_proj(kv_in).reshape(b, lk, 2, attn.num_heads, attn.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)
        q = attn.q_norm(q).to(dtype=v.dtype)
        k = attn.k_norm(k).to(dtype=v.dtype)
        # Apply the same axial (t,v,y,x) RoPE used by GAM self-attention.
        # Query and memory lengths differ, so rotate them independently.
        q, _ = query_rope.apply_rope(q, q, query_positions)
        k, _ = memory_rope.apply_rope(k, k, memory_positions)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, lq, dim)
        # Residual update: only the query stream is written.  ``memory`` is
        # read-only for this directed edge.
        return query + self.gate(attn.dropout(attn.out_proj(out)))


class ActionBridge(nn.Module):
    """Four directed edges implementing ``Geometry <-> Action <-> DINO``.

    Separate modules matter here: geometry-reading-action and
    action-reading-geometry do not share Q/KV/O weights.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.geometry_reads_action = CausalCrossAttention(dim, heads, dropout)
        self.dino_reads_action = CausalCrossAttention(dim, heads, dropout)
        self.action_reads_geometry = CausalCrossAttention(dim, heads, dropout)
        self.action_reads_dino = CausalCrossAttention(dim, heads, dropout)

    def forward(
        self, geometry, dino, action, g_from_a, d_from_a, a_from_g, a_from_d,
        geometry_rope, dino_rope, geometry_positions, dino_positions, action_positions,
    ):
        # First let both representation streams read the current action state.
        geometry = self.geometry_reads_action(
            geometry, action, g_from_a, geometry_rope, geometry_rope,
            geometry_positions, action_positions
        )
        dino = self.dino_reads_action(
            dino, action, d_from_a, dino_rope, geometry_rope,
            dino_positions, action_positions
        )
        # Then update action from the already-updated geometry and DINO states.
        # Action is the only route by which geometry and DINO can communicate.
        action = self.action_reads_geometry(
            action, geometry, a_from_g, geometry_rope, geometry_rope,
            action_positions, geometry_positions
        )
        action = self.action_reads_dino(
            action, dino, a_from_d, geometry_rope, dino_rope,
            action_positions, dino_positions
        )
        return geometry, dino, action


class UAVMultimodalGAMFuturePredictor(nn.Module):
    """Three parameter-separated streams with GAM-equivalent token semantics.

    Geometry, DINO, and action/proprio own their self-attention and FFN weights.
    Every stream uses GAM's 4D axial RoPE, block-causal attention, language
    cross-attention, QK normalization, RMSNorm, SwiGLU, and LayerScale.

    Input/output contract::

        DA3:  [B,H,V,Pg,1536] -> geometry stream [B,H*V*Pg,D]
        DINO: [B,H,V,Pd, 768] -> DINO stream     [B,H*V*Pd,D]
        state/action history  -> action stream   [B,H*2,D]

    In the GAM-aligned setting, the predictor emits one next visual/state
    latent and one action latent; the downstream action head may expand that
    latent into a multi-action chunk.
    """

    def __init__(
        self,
        d_da3=1536,
        d_dino=768,
        d_model=768,
        depth=8,
        num_heads=12,
        geometry_ffn_ratio=4.0,
        dino_ffn_ratio=2.0,
        action_ffn_ratio=2.0,
        dino_layer_stride=2,
        dropout=0.0,
        num_patches_per_view=256,
        num_dino_patches_per_view=196,
        num_register_tokens=4,
        proprio_dim=6,
        action_dim=6,
        action_chunk_size=1,
        use_proprio_input=True,
        future_horizon=1,
        use_language=True,
        language_dim=768,
        language_len=77,
        variable_language_tokens=False,
        input_proj_norm="ln",
        gradient_checkpointing=False,
        use_flex_attention=False,
        sigreg=None,
        sigreg_proj_dim=256,
        sigreg_pool_mode="cls",
        condition_mode="cross_attn",
        **kwargs,
    ):
        super().__init__()
        if str(condition_mode).lower() != "cross_attn":
            raise ValueError("UAV multimodal predictor currently supports condition_mode='cross_attn' only.")
        if num_patches_per_view <= 0:
            raise ValueError("num_patches_per_view must be positive.")
        geometry_side = int(round(math.sqrt(num_patches_per_view)))
        if geometry_side * geometry_side != num_patches_per_view:
            raise ValueError("num_patches_per_view must form a square patch grid.")
        dino_side = int(round(math.sqrt(num_dino_patches_per_view)))
        if dino_side * dino_side != num_dino_patches_per_view:
            raise ValueError("num_dino_patches_per_view must form a square patch grid.")

        # ----- Shape metadata -------------------------------------------------
        self.d_da3 = int(d_da3)
        self.d_dino = int(d_dino)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.patches = int(num_patches_per_view)
        self.dino_patches = int(num_dino_patches_per_view)
        self.num_register_tokens = int(num_register_tokens)
        self.num_prefix_visual = 1 + self.num_register_tokens
        self.visual_tokens_per_view = self.num_prefix_visual + self.patches
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        self.action_history_dim = self.action_dim * self.action_chunk_size
        self.use_proprio_input = bool(use_proprio_input)
        self.action_tokens_per_step = 2 if self.use_proprio_input else 1
        # Explicitly declare the temporal contract. Original GAM predicts one
        # next state/action from every observed anchor, so output length is H.
        # The constructor argument is accepted only for config compatibility.
        self.is_multi_horizon = False
        self.config_future_horizon_ignored = int(future_horizon)
        self.dino_layer_stride = max(1, int(dino_layer_stride))
        self.use_language = bool(use_language)
        self.language_len = int(language_len)
        self.variable_language_tokens = bool(variable_language_tokens)
        self.use_gradient_checkpointing = bool(gradient_checkpointing)
        self.use_flex_attention = bool(use_flex_attention)

        # ----- Input projections: native modality width -> shared D ----------
        # Reuse the same configurable projector classes as original GAM.
        projector = {"ln": LNMLPProjector, "layernorm": LNMLPProjector,
                     "bn": BNMLPProjector, "batchnorm": BNMLPProjector}.get(str(input_proj_norm).lower())
        if projector is None:
            raise ValueError("input_proj_norm must be 'ln' or 'bn'.")
        self.geo_in = projector(self.d_da3, self.d_model)
        self.dino_in = projector(self.d_dino, self.d_model)
        self.proprio_token_proj = nn.Sequential(
            nn.Linear(self.proprio_dim, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        self.action_history_proj = nn.Sequential(
            nn.Linear(self.action_history_dim, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )

        # ----- Token identity and position -----------------------------------
        # GAM indices 0..4 are preserved exactly:
        #   0 CLS, 1 register, 2 DA3 patch, 3 proprio, 4 previous action.
        # DINO has no type offset: its independent input/blocks already identify
        # the modality, and every token in that stream is the same patch type.
        self.type_embed = nn.Parameter(torch.zeros(5, self.d_model))
        nn.init.normal_(self.type_embed, std=0.02)
        self.rope = ShallowRoPE(
            self.d_model // self.num_heads, geometry_side, geometry_side
        )
        self.dino_rope = ShallowRoPE(
            self.d_model // self.num_heads, dino_side, dino_side
        )
        # ----- Three independent temporal streams ----------------------------
        # PlainTextCrossBlock is the original GAM block:
        # self-attention -> language cross-attention -> SwiGLU FFN.
        self.geometry_blocks = nn.ModuleList([
            PlainTextCrossBlock(self.d_model, self.num_heads, geometry_ffn_ratio, dropout)
            for _ in range(self.depth)
        ])
        # Only instantiate blocks that execute; avoids dormant optimizer state.
        self.dino_block_indices = list(range(0, self.depth, self.dino_layer_stride))
        self.dino_blocks = nn.ModuleDict({
            str(i): PlainTextCrossBlock(self.d_model, self.num_heads, dino_ffn_ratio, dropout)
            for i in self.dino_block_indices
        })
        self.action_blocks = nn.ModuleList([
            PlainTextCrossBlock(self.d_model, self.num_heads, action_ffn_ratio, dropout)
            for _ in range(self.depth)
        ])
        # One four-edge cross-modal bridge follows every stream layer.
        self.bridges = nn.ModuleList([
            ActionBridge(self.d_model, self.num_heads, dropout) for _ in range(self.depth)
        ])

        # ----- Shared language memory ----------------------------------------
        # The projected CLIP tokens are shared as memory, but every stream block
        # owns an independent language cross-attention module.
        if self.use_language:
            self.lang_proj = nn.Linear(language_dim, self.d_model, bias=True)
            # Fixed CLIP/T5 tokens retain GAM's learned language positions.
            # Joint VLM tokens already contain image/text positions, so their
            # variable-length path bypasses this table entirely.
            if self.variable_language_tokens:
                self.register_parameter("lang_pos", None)
            else:
                self.lang_pos = nn.Parameter(torch.zeros(self.language_len, self.d_model))
                nn.init.normal_(self.lang_pos, std=0.02)
        else:
            self.lang_proj = None
            self.lang_pos = None

        # ----- Pre-normalized output heads: shared D -> native widths ---------
        self.out_visual_norm = _make_rmsnorm(self.d_model)
        self.out_dino_norm = _make_rmsnorm(self.d_model)
        self.out_action_norm = _make_rmsnorm(self.d_model)
        self.out_proprio_norm = _make_rmsnorm(self.d_model)
        self.future_visual_proj = nn.Linear(self.d_model, self.d_da3)
        self.future_dino_proj = nn.Linear(self.d_model, self.d_dino)
        self.action_proj = nn.Linear(self.d_model, self.d_da3)
        self.future_proprio_proj = nn.Linear(self.d_model, self.proprio_dim)

        self.sigreg = sigreg
        self.sigreg_proj_dim = int(sigreg_proj_dim)
        self.sigreg_pool_mode = str(sigreg_pool_mode)
        self.sigreg_proj = nn.Linear(self.d_model, self.sigreg_proj_dim) if sigreg is not None else None
        self._flex_mask_cache = {}

    def _visual_type_embed(self, device, dtype):
        """Build one-view ``[CLS, registers, patches]`` GAM type offsets."""
        types = torch.full((self.visual_tokens_per_view,), 2, device=device, dtype=torch.long)
        types[0] = 0
        types[1:self.num_prefix_visual] = 1
        return self.type_embed[types].to(dtype=dtype)

    def _positions(self, H: int, V: int, device: torch.device):
        """Return 4D RoPE coordinates for all three flattened streams.

        Geometry/DINO patches get ``(t, view, y, x)``. When enabled, proprio
        and previous action use virtual views ``V`` and ``V+1`` exactly as in
        original GAM; without proprio, previous action uses virtual view ``V``.
        """
        all_positions = self.rope.build_positions(
            H, V, self.patches, self.num_prefix_visual, device,
            include_proprio_slot=self.use_proprio_input,
        ).reshape(H, V * self.visual_tokens_per_view + self.action_tokens_per_step, 4)
        geometry = all_positions[:, :V * self.visual_tokens_per_view].reshape(-1, 4)
        action = all_positions[:, -self.action_tokens_per_step:].reshape(-1, 4)
        dino_all = self.dino_rope.build_positions(
            H, V, self.dino_patches, 0, device,
            include_proprio_slot=self.use_proprio_input,
        ).reshape(
            H, V * self.dino_patches + self.action_tokens_per_step, 4
        )
        dino = dino_all[:, :V * self.dino_patches].reshape(-1, 4)
        return geometry, dino, action

    def _flex_or_dense(self, H: int, width: int, device: torch.device, key_keep=None):
        """Use GAM FlexAttention when possible, otherwise dense SDPA mask.

        A sample-specific validity mask cannot be represented by the cached,
        shape-only BlockMask, so that case intentionally uses a dense mask.
        """
        if self.use_flex_attention and key_keep is None and _HAS_FLEX:
            key = (H, width, device)
            block_mask = self._flex_mask_cache.get(key)
            if block_mask is None:
                try:
                    block_mask = _create_block_mask(
                        _make_block_causal_mask_mod(width), B=None, H=None,
                        Q_LEN=H * width, KV_LEN=H * width, device=device,
                    )
                    self._flex_mask_cache[key] = block_mask
                except Exception:
                    block_mask = None
            if block_mask is not None:
                return None, block_mask
        dense = _merge_key_keep(_dense_causal_mask(H, width, device), key_keep)
        return dense, None

    @staticmethod
    def _apply_keep(x, keep):
        return x if keep is None else x * keep.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

    def _language(self, lang_feats, lang_padding_mask, b, dtype, device):
        """Project CLIP language tokens once for all three stream blocks."""
        if not self.use_language or lang_feats is None:
            return None, None
        lang = self.lang_proj(lang_feats.to(self.lang_proj.weight.dtype)).to(dtype)
        if not self.variable_language_tokens:
            if lang.shape[1] > self.language_len:
                lang = lang[:, :self.language_len]
            lang = lang + self.lang_pos[:lang.shape[1]][None].to(dtype)
        if lang_padding_mask is None:
            keep = torch.ones(b, lang.shape[1], device=device, dtype=torch.bool)
        else:
            keep = lang_padding_mask[:, :lang.shape[1]].to(device=device, dtype=torch.bool)
        empty = ~keep.any(dim=1)
        if empty.any():
            keep = keep.clone()
            keep[empty, 0] = True
        return lang, keep

    def _run_block(self, block, x, positions, dense, flex, lang, lang_keep, use_checkpoint, rope=None):
        """Run one original-GAM block, optionally with activation checkpointing."""
        kwargs = dict(
            rope=self.rope if rope is None else rope,
            rope_positions=positions,
            self_attn_mask=dense,
            flex_block_mask=flex,
            text_context=lang,
            text_keep_mask=lang_keep,
        )
        if use_checkpoint:
            return torch_checkpoint(lambda value: block(value, **kwargs), x, use_reentrant=False)
        return block(x, **kwargs)

    def forward(
        self,
        past_visual_tokens,
        past_dino_tokens,
        proprio=None,
        proprio_history=None,
        past_action_history=None,
        lang_feats=None,
        lang_padding_mask=None,
        context_valid_mask=None,
        view_valid_mask=None,
    ):
        """Predict one next transition from every observed timestep.

        This mirrors original GAM: the sequence contains only observed anchors
        `[o_t, proprio_t, prev_action_{t-1}]`. There are no learned future query
        slots. The returned tensors all have length H and are supervised against
        anchors/actions `[t+1 ... t+H]` by `gam_ar_unified_loss`.
        """
        # =====================================================================
        # 1) Validate the native encoder outputs.
        # =====================================================================
        b, H, V, Pg, d_geo = past_visual_tokens.shape
        expected_geo = (b, H, V, self.visual_tokens_per_view, self.d_da3)
        if tuple(past_visual_tokens.shape) != expected_geo:
            raise ValueError(f"Geometry shape {tuple(past_visual_tokens.shape)} != {expected_geo}")
        expected_dino = (b, H, V, self.dino_patches, self.d_dino)
        if tuple(past_dino_tokens.shape) != expected_dino:
            raise ValueError(f"DINO shape {tuple(past_dino_tokens.shape)} != {expected_dino}")
        device, dtype = past_visual_tokens.device, past_visual_tokens.dtype

        # =====================================================================
        # 2) Expand timestep/view validity into one keep bit per stream token.
        # =====================================================================
        context_keep = torch.ones(b, H, device=device, dtype=torch.bool)
        if context_valid_mask is not None:
            context_keep = context_valid_mask.to(device=device, dtype=torch.bool)
            if tuple(context_keep.shape) != (b, H):
                raise ValueError(f"context_valid_mask must be {(b, H)}, got {tuple(context_keep.shape)}")
        view_keep = torch.ones(b, H, V, device=device, dtype=torch.bool)
        if view_valid_mask is not None:
            view_keep = view_valid_mask.to(device=device, dtype=torch.bool)
            if tuple(view_keep.shape) != (b, H, V):
                raise ValueError(f"view_valid_mask must be {(b, H, V)}, got {tuple(view_keep.shape)}")
        view_keep = view_keep & context_keep.unsqueeze(-1)
        if not bool(view_keep.any(dim=2).any(dim=1).all().item()):
            raise ValueError("Every sample must contain at least one valid context view.")

        S = H
        geo_keep = view_keep[:, :, :, None].expand(-1, -1, -1, Pg).reshape(b, -1)
        dino_keep = view_keep[:, :, :, None].expand(
            -1, -1, -1, self.dino_patches
        ).reshape(b, -1)
        action_keep = context_keep[:, :, None].expand(-1, -1, self.action_tokens_per_step).reshape(b, -1)

        # =====================================================================
        # 3) Build the three sequences in common width D.
        # =====================================================================
        # Geometry: [B,H,V,Pg,1536] -> [B,H*V*Pg,D], retaining GAM token types.
        geometry = self.geo_in(past_visual_tokens.reshape(b, -1, d_geo))
        geometry = geometry + self._visual_type_embed(device, geometry.dtype).repeat(H * V, 1)[None]
        # DINO: [B,H,V,Pd,768] -> [B,H*V*Pd,D].
        dino = self.dino_in(past_dino_tokens.reshape(b, -1, self.d_dino))

        if past_action_history is None:
            raise ValueError("past_action_history is required.")
        past_action_history = past_action_history[:, -H:]
        # Action stream has either [proprio_t, previous-action_(t-1)] or just
        # [previous-action_(t-1)].  The previous action may itself be a chunk,
        # flattened from [chunk, action_dim] before projection.
        prev = self.action_history_proj(
            past_action_history.reshape(b, H, -1).to(self.action_history_proj[0].weight.dtype)
        ).to(dtype)
        prev = prev + self.type_embed[4].to(dtype=dtype)
        if self.use_proprio_input:
            if proprio_history is None:
                if proprio is None:
                    raise ValueError("proprio_history is required when proprio is absent.")
                proprio_history = proprio[:, None].expand(-1, H, -1)
            proprio_history = proprio_history[:, -H:]
            prop = self.proprio_token_proj(proprio_history.to(self.proprio_token_proj[0].weight.dtype)).to(dtype)
            prop = prop + self.type_embed[3].to(dtype=dtype)
            action = torch.stack((prop, prev), dim=2).reshape(b, H * 2, self.d_model)
        else:
            action = prev.reshape(b, H, 1, self.d_model).reshape(b, H, self.d_model)

        # =====================================================================
        # 4) Build RoPE coordinates and causal masks once, reused by all layers.
        # =====================================================================
        geometry = self._apply_keep(geometry, geo_keep)
        dino = self._apply_keep(dino, dino_keep)
        action = self._apply_keep(action, action_keep)
        geo_pos, dino_pos, action_pos = self._positions(S, V, device)
        geo_dense, geo_flex = self._flex_or_dense(S, V * Pg, device, None if geo_keep.all() else geo_keep)
        dino_dense, dino_flex = self._flex_or_dense(
            S, V * self.dino_patches, device, None if dino_keep.all() else dino_keep
        )
        action_dense, action_flex = self._flex_or_dense(
            S, self.action_tokens_per_step, device, None if action_keep.all() else action_keep
        )
        lang, lang_keep = self._language(lang_feats, lang_padding_mask, b, dtype, device)

        # Mask names are "query_from_memory": g_from_a means G queries A.
        g_from_a = _merge_key_keep(_cross_causal_mask(S, V * Pg, self.action_tokens_per_step, device), action_keep)
        d_from_a = _merge_key_keep(
            _cross_causal_mask(S, V * self.dino_patches, self.action_tokens_per_step, device), action_keep
        )
        a_from_g = _merge_key_keep(_cross_causal_mask(S, self.action_tokens_per_step, V * Pg, device), geo_keep)
        a_from_d = _merge_key_keep(
            _cross_causal_mask(S, self.action_tokens_per_step, V * self.dino_patches, device), dino_keep
        )

        # =====================================================================
        # 5) Repeated update: per-stream GAM block, then four-edge bridge.
        # =====================================================================
        use_checkpoint = self.use_gradient_checkpointing and self.training and torch.is_grad_enabled()
        for i in range(self.depth):
            geometry = self._run_block(
                self.geometry_blocks[i], geometry, geo_pos, geo_dense, geo_flex,
                lang, lang_keep, use_checkpoint,
            )
            if str(i) in self.dino_blocks:
                dino = self._run_block(
                    self.dino_blocks[str(i)], dino, dino_pos, dino_dense, dino_flex,
                    lang, lang_keep, use_checkpoint, rope=self.dino_rope,
                )
            action = self._run_block(
                self.action_blocks[i], action, action_pos, action_dense, action_flex,
                lang, lang_keep, use_checkpoint,
            )
            if use_checkpoint:
                geometry, dino, action = torch_checkpoint(
                    lambda g, d, a, bridge=self.bridges[i]: bridge(
                        g, d, a, g_from_a, d_from_a, a_from_g, a_from_d,
                        self.rope, self.dino_rope, geo_pos, dino_pos, action_pos,
                    ),
                    geometry, dino, action, use_reentrant=False,
                )
            else:
                geometry, dino, action = self.bridges[i](
                    geometry, dino, action, g_from_a, d_from_a, a_from_g, a_from_d,
                    self.rope, self.dino_rope, geo_pos, dino_pos, action_pos,
                )
            # Residual/bias terms can recreate nonzero padded queries, therefore
            # clear invalid tokens again after every complete layer.
            geometry = self._apply_keep(geometry, geo_keep)
            dino = self._apply_keep(dino, dino_keep)
            action = self._apply_keep(action, action_keep)

        # =====================================================================
        # 6) Restore structured shapes and project to loss/deep-DA3 interfaces.
        # =====================================================================
        # Original-GAM alignment: decode every observed timestep hidden state.
        # Step i predicts the next visual/DINO/proprio anchor and the action
        # chunk starting at that anchor.
        geometry_h = geometry.reshape(b, S, V, Pg, self.d_model)
        dino_h = dino.reshape(b, S, V, self.dino_patches, self.d_model)
        action_h = action.reshape(b, S, self.action_tokens_per_step, self.d_model)
        # One action latent is predicted per timestep. DA3 deep propagation
        # expects one seed per view, so the same latent is repeated over V.
        action_slot_index = 1 if self.use_proprio_input else 0
        action_per_step = self.action_proj(self.out_action_norm(action_h[:, :, action_slot_index]))
        action_all = action_per_step[:, :, None].expand(-1, -1, V, -1).contiguous()

        # Optional anti-collapse regularizer, identical pooling choices to GAM.
        sigreg_loss = None
        if self.sigreg is not None:
            if self.sigreg_pool_mode == "cls":
                pooled = geometry_h[:, :, :, 0]
            elif self.sigreg_pool_mode == "patch_mean":
                pooled = geometry_h[:, :, :, self.num_prefix_visual:].mean(dim=3)
            elif self.sigreg_pool_mode == "all_mean":
                pooled = geometry_h.mean(dim=3)
            else:
                raise ValueError(f"Unknown sigreg_pool_mode={self.sigreg_pool_mode!r}")
            sigreg_loss = self.sigreg(self.sigreg_proj(pooled).reshape(-1, self.sigreg_proj_dim))

        return {
            "predicted_next_visual_tokens": self.future_visual_proj(self.out_visual_norm(geometry_h)).to(dtype),
            "predicted_next_dino_tokens": self.future_dino_proj(self.out_dino_norm(dino_h)).to(dtype),
            "predicted_next_proprio": (
                self.future_proprio_proj(self.out_proprio_norm(action_h[:, :, 0])).to(dtype)
                if self.use_proprio_input else None
            ),
            "predicted_action_tokens": action_all.to(dtype),
            "encoded_prev_action_tokens": action_h[:, :, action_slot_index],
            "encoded_proprio_tokens": action_h[:, :, 0] if self.use_proprio_input else None,
            "sigreg_loss": sigreg_loss,
        }
