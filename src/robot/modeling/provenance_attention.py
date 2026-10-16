"""Two action ACTIVATIONS, one DA3 view trunk and the original block weights.

Geometry queries use paired logical action sources, not an averaged hidden.
Each action branch reads only its own action keys and the shared visual keys.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LocalReliability(nn.Module):
    """Expected normalized feature error; not a covariance or depth variance."""
    def __init__(self, dim, layers=(27, 39), width=64, tau=1.):
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or tau <= 0:
            raise ValueError('Reliability needs distinct layers and positive tau')
        self.tau = float(tau)
        self.norms = nn.ModuleDict({str(i): nn.LayerNorm(dim) for i in self.layers})
        self.head = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, patches, layer):
        u = F.softplus(self.head(self.norms[str(layer)](patches.detach())).float()[..., 0])
        return u, torch.exp(-u.detach() / self.tau)

    def loss(self, records, teacher, rho, gate=None):
        losses, metrics = [], {}
        weight = 1-rho.detach().float()
        if gate is not None:
            weight = weight * gate.detach().float()[:, None]
        for layer, record in records.items():
            # Student/teacher both refer to INPUT to block `layer`.
            target = teacher['layer_patches'][layer-1][:, 0, 0].detach().float()
            pred = record['patches'].detach().float()
            error = (F.layer_norm(pred, (pred.shape[-1],))
                     - F.layer_norm(target, (target.shape[-1],))).abs().mean(-1)
            u = record['uncertainty']
            calibrated = F.smooth_l1_loss(torch.log(u+1e-6), torch.log(error+1e-6), reduction='none')
            losses.append((calibrated*weight).sum()/weight.sum().clamp_min(1.))
            mass = weight.sum().clamp_min(1.)
            metrics[f'uncertainty_L{layer}'] = (u.detach()*weight).sum()/mass
            metrics[f'confidence_L{layer}'] = (record['confidence']*weight).sum()/mass
            metrics[f'uncertainty_mae_L{layer}'] = ((u.detach()-error).abs()*weight).sum()/mass
        if not losses:
            raise ValueError('No selected deep-layer reliability records')
        return torch.stack(losses).mean(), metrics


def routed_block(x, unsupported, block, pos, *, action_count, rho, confidence=None):
    """H=1/V=1 block, original [camera, K actions, registers, patches] layout.

    No independent masked softmax: P is normalized over the SAME original keys,
    then rho/(1-rho)/confidence multiply only visual contributions. Prefix
    visual tokens use mean spatial membership (no ungated camera bypass).
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError('Provenance routing currently requires H=1, V=1')
    rho = rho.detach()
    b, _, n, d = x.shape
    if unsupported.shape != (b, 1, action_count, d):
        raise ValueError('Unsupported action state shape mismatch')
    patches = rho.shape[-1]
    prefix = n-patches
    if rho.shape != (b, patches) or prefix < 1+action_count:
        raise ValueError('Support shape / DA3 token layout mismatch')
    sup = x[:, 0, 1:1+action_count]
    uns = unsupported[:, 0]
    attn = block.attn
    h = block.norm1(x[:, 0])
    qkv = attn.qkv(h).reshape(b, n, 3, attn.num_heads, d//attn.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    positions = None if pos is None else pos.reshape(b, n, -1)
    if attn.rope is not None and positions is not None:
        q, k = attn.rope(q, positions), attn.rope(k, positions)
    dropout = float(getattr(getattr(attn, 'attn_drop', None), 'p', 0.)) if attn.training else 0.
    uqkv = attn.qkv(block.norm1(uns)).reshape(b, action_count, 3, attn.num_heads, d//attn.num_heads)
    uq, uk, uv = uqkv.permute(2, 0, 3, 1, 4).unbind(0)
    uq, uk = attn.q_norm(uq), attn.k_norm(uk)
    if attn.rope is not None and positions is not None:
        ap = positions[:, 1:1+action_count]
        uq, uk = attn.rope(uq, ap), attn.rope(uk, ap)
    visual = torch.ones(n, device=x.device, dtype=torch.bool)
    visual[1:1+action_count] = False
    membership = rho.new_ones(b, n)
    membership[:, :prefix] = rho.mean(-1, keepdim=True)
    membership[:, prefix:] = rho
    cfull = torch.ones_like(membership)
    if confidence is not None:
        cfull[:, :prefix] = confidence.detach().mean(-1, keepdim=True)
        cfull[:, prefix:] = confidence.detach()
    scale = math.sqrt(d//attn.num_heads)
    scores = q.float() @ k.float().transpose(-1, -2) / scale
    # A pair occupies ONE logical slot: weighted exponents share the full
    # geometry denominator. logsumexp/softmax provides stable aggregation.
    ls = scores[..., 1:1+action_count]
    lu = q.float() @ uk.float().transpose(-1, -2) / scale
    rs = membership[:, None, :, None]
    ru = ((1-membership)*cfull)[:, None, :, None]
    log_s = ls + torch.where(rs > 0, rs.log(), -torch.inf)
    log_u = lu + torch.where(ru > 0, ru.log(), -torch.inf)
    active = (rs+ru) > 0
    # Avoid logaddexp(-inf,-inf)'s undefined backward at rho=0,c=0.
    pair = torch.logaddexp(torch.where(active, log_s, torch.zeros_like(log_s)),
                          torch.where(active, log_u, torch.zeros_like(log_u)))
    pair = torch.where(active, pair, -torch.inf)
    logical = torch.cat((scores[..., :1], pair, scores[..., 1+action_count:]), -1)
    prob = logical.softmax(-1)
    prob = F.dropout(prob, p=dropout, training=attn.training)
    pair_p = prob[..., 1:1+action_count]
    finite = torch.isfinite(pair)
    safe_pair = torch.where(finite, pair, torch.zeros_like(pair))
    ps = pair_p * torch.exp(log_s-safe_pair)
    pu = pair_p * torch.exp(log_u-safe_pair)
    trunk = prob.to(v.dtype) @ v
    trunk = trunk - pair_p.to(v.dtype) @ v[..., 1:1+action_count, :]
    trunk = trunk + ps.to(v.dtype) @ v[..., 1:1+action_count, :] + pu.to(v.dtype) @ uv
    # Action queries have a separate ORIGINAL-size denominator containing
    # their own K action keys only. There is no cross-branch edge.
    own_uk = torch.cat((k[..., :1, :], uk, k[..., 1+action_count:, :]), -2)
    own_uv = torch.cat((v[..., :1, :], uv, v[..., 1+action_count:, :]), -2)
    sp = scores[..., 1:1+action_count, :].softmax(-1)
    up = (uq.float() @ own_uk.float().transpose(-1, -2) / scale).softmax(-1)
    sp = F.dropout(sp, p=dropout, training=attn.training)
    up = F.dropout(up, p=dropout, training=attn.training)
    sw = torch.where(visual[None], membership, torch.ones_like(membership))
    uw = torch.where(visual[None], 1-membership, torch.ones_like(membership))
    before = (up*uw[:, None, None]*visual[None,None,None]).to(v.dtype) @ own_uv
    uw = uw * torch.where(visual[None], cfull, torch.ones_like(cfull))
    sup_y = (sp*sw[:, None,None]).to(v.dtype) @ v
    uns_y = (up*uw[:, None,None]).to(v.dtype) @ own_uv
    after = (up*uw[:, None,None]*visual[None,None,None]).to(v.dtype) @ own_uv
    # Trunk has precisely the old number of tokens. Replace only action rows.
    trunk = torch.cat((trunk[:, :, :1], sup_y, trunk[:, :, 1+action_count:]), 2)
    y = trunk.transpose(1, 2).reshape(b, n, d)
    out = x[:, 0] + block.ls1(attn.proj_drop(attn.proj(y)))
    out = out + block.ls2(block.mlp(block.norm2(out)))
    uy = uns_y.transpose(1, 2).reshape(b, action_count, d)
    uout = uns + block.ls1(attn.proj_drop(attn.proj(uy)))
    uout = uout + block.ls2(block.mlp(block.norm2(uout)))
    return out[:, None], uout[:, None], {
        'unsupported_read_before': before.detach().float().norm(dim=-1).mean(),
        'unsupported_read_after': after.detach().float().norm(dim=-1).mean(),
        'supported_read': ((sp*sw[:,None,None]*visual[None,None,None]).to(v.dtype) @ v).detach().float().norm(dim=-1).mean(),
    }


def shared_residual(head, supported, unsupported, rho):
    """Support-mass convex merge; the shared decoder bias occurs once."""
    ds, du = head(supported), head(unsupported)
    alpha = rho.detach().mean(-1).reshape(rho.shape[0], *([1]*(ds.ndim-1)))
    return alpha*ds, (1-alpha)*du, alpha
