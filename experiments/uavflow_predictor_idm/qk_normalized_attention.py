"""Geometry attention with per-head normalization after the Q/K projections."""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class QKNormalizedAttention(nn.MultiheadAttention):
    """Keep legacy MHA state keys; opt-in adds only two head LayerNorms.

    CUDA uses math SDPA: the efficient masked backward produced incorrect
    gradients on captured saturated correspondence-attention inputs.
    """

    def __init__(self, embed_dim, num_heads, *, qk_norm=False):
        super().__init__(embed_dim, num_heads, batch_first=True)
        self.qk_norm_enabled = bool(qk_norm)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, query, key, value, *, attn_mask=None, need_weights=False):
        if not self.qk_norm_enabled:
            return super().forward(query, key, value, attn_mask=attn_mask,
                                   need_weights=need_weights)
        if need_weights:
            raise ValueError('Geometry QK-normalized attention does not return attention weights')
        batch, query_count, _ = query.shape
        key_count = key.shape[1]
        weights = self.in_proj_weight.chunk(3)
        biases = self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None,) * 3
        projected = [F.linear(x, w, b).reshape(batch, -1, self.num_heads, self.head_dim)
                     .transpose(1, 2)
                     for x, w, b in zip((query, key, value), weights, biases)]
        q, k, v = projected
        # Normalize the actual per-head projected Q/K, not their input features.
        q = self.q_norm(q.float()).to(v.dtype)
        k = self.k_norm(k.float()).to(v.dtype)
        mask = attn_mask
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.reshape(batch, self.num_heads, query_count, key_count)
            elif mask.ndim != 2:
                raise ValueError('Expected 2D or batch*heads 3D geometry attention mask')
            # MHA True=blocked; SDPA True=allowed.
            if mask.dtype == torch.bool:
                mask = ~mask
        backend = nullcontext()
        if query.device.type == 'cuda':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = sdpa_kernel(SDPBackend.MATH)
        with backend:
            result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0)
        result = result.transpose(1, 2).reshape(batch, query_count, self.embed_dim)
        return F.linear(result, self.out_proj.weight, self.out_proj.bias), None
