"""Directed dual-state mask for [current view, predicted-future view].

Each view has [camera, action, registers, patches]. Action tokens may read all
tokens. Current visual queries see only current visual keys. Future visual
queries see future visuals and both actions, never current visuals directly.
This restriction MUST be applied to local as well as global DA3 blocks.
"""
import torch
from torch.nn import functional as F


def dual_allow_mask(tokens_per_view, *, local=False, device=None):
    if tokens_per_view < 3:
        raise ValueError("Expected camera, action and visual tokens")
    i = torch.arange(2 * tokens_per_view, device=device)
    action = i.remainder(tokens_per_view) == 1
    current = (i < tokens_per_view) & ~action
    future = (i >= tokens_per_view) & ~action
    allowed = (action[:, None]
               | (current[:, None] & current[None, :])
               | (future[:, None] & (future | action)[None, :]))
    if local:
        n = tokens_per_view
        return torch.stack([allowed[:n, :n], allowed[n:, n:]])
    return allowed


def run_dual_masked_block(x, block, pos, *, global_attention):
    """DA3 pre-norm attention/FFN with explicit allowed=True SDPA masks.

    H=1, two views only. This follows the existing deep flex runner's block
    residual semantics. No new projections or parameters are introduced.
    """
    b, views, n, c = x.shape
    if views != 2:
        raise ValueError("Directed dual-state attention requires exactly two views")
    original_shape = x.shape
    if global_attention:
        x = x.reshape(b, 2 * n, c)
        positions = None if pos is None else pos.reshape(b, 2 * n, -1)
        mask = dual_allow_mask(n, device=x.device)[None, None]
    else:
        x = x.reshape(b * 2, n, c)
        positions = None if pos is None else pos.reshape(b * 2, n, -1)
        mask = dual_allow_mask(n, local=True, device=x.device).repeat(b, 1, 1)[:, None]
    h = block.norm1(x)
    attn = block.attn
    batch, length, _ = h.shape
    qkv = attn.qkv(h).reshape(batch, length, 3, attn.num_heads, c // attn.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and positions is not None:
        q, k = attn.rope(q, positions), attn.rope(k, positions)
    dropout = float(getattr(getattr(attn, "attn_drop", None), "p", 0.0)) if attn.training else 0.0
    y = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                                      attn_mask=mask, dropout_p=dropout)
    y = y.transpose(1, 2).reshape(batch, length, c)
    x = x + block.ls1(attn.proj_drop(attn.proj(y)))
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x.reshape(original_shape)
