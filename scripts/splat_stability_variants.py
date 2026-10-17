"""Offline paper-formula probes, NOT production warp or official renderer.

Log-kernel normalization follows Briedis et al., SIGGRAPH 2023 suppl. 1.1.
Point disks + depth-sorted alpha-over follow the mechanism of SynSin (2020),
with explicitly selected pixel-space radius, gamma=1 and top K=8.
Discrete footprint membership, ordering and max stabilization are detached.
"""
import torch


def _geometry(rgb,flow,z):
    b,c,h,w=rgb.shape
    if flow.shape!=(b,2,h,w) or z.shape!=(b,1,h,w):
        raise ValueError('Expected flow [B,2,H,W], target z [B,1,H,W]')
    yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
    pixels=torch.stack((xx,yy),-1).float().reshape(1,h*w,2)
    uv=pixels+flow.permute(0,2,3,1).reshape(b,h*w,2)
    colors=rgb.permute(0,2,3,1).reshape(b*h*w,c)
    return b,c,h,w,uv,colors


def log_kernel_splat(rgb,flow,z,*,importance=None,epsilon=1e-7):
    """Stable per-target exp(Z+log(kernel)-max), no hard z-buffer.

    epsilon=1e-7 follows paper-style normalization. epsilon=0 tests exact
    normalized interpolation; empty pixels still use a safe denominator.
    Stable arithmetic does NOT guarantee bounded derivatives w.r.t. flow.
    """
    b,c,h,w,uv,colors=_geometry(rgb,flow,z)
    base=uv.detach().floor().long();fraction=uv-base
    dx,dy=fraction.unbind(-1)
    weights=torch.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),-1)
    coords=base[:,:,None]+base.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    good=(coords[...,0]>=0)&(coords[...,0]<w)&(coords[...,1]>=0)&(coords[...,1]<h)&(weights.detach()>0)
    batch=torch.arange(b,device=rgb.device)[:,None,None].expand_as(weights)
    source=torch.arange(b*h*w,device=rgb.device).reshape(b,h*w,1).expand_as(weights)[good]
    destination=(batch*h*w+coords[...,1]*w+coords[...,0])[good]
    logits=weights[good].log()
    if importance is not None:
        if importance.shape!=(b,1,h,w):raise ValueError('Expected importance [B,1,H,W]')
        logits=logits+importance.flatten()[source]
    maximum=rgb.new_full((b*h*w,),-float('inf'))
    maximum.scatter_reduce_(0,destination,logits.detach(),reduce='amax',include_self=True)
    shifted=(logits-maximum[destination]).exp()
    mass=rgb.new_zeros(b*h*w).scatter_add(0,destination,shifted)
    numerator=rgb.new_zeros(b*h*w,c).scatter_add(0,destination[:,None].expand(-1,c),colors[source]*shifted[:,None])
    result=numerator/(mass+epsilon).clamp_min(torch.finfo(rgb.dtype).tiny)[:,None]
    return result.reshape(b,h,w,c).permute(0,3,1,2)


def alpha_point_splat(rgb,flow,z,*,radius=1.5,points_per_pixel=8):
    """Finite disk opacity, front-to-back alpha-over, black background.

    Does NOT divide by total coverage. No triangle mesh and no learned
    confidence. Not an exact PyTorch3D/SynSin implementation or performance
    reproduction: camera projection is ours, disk distances are in pixels.
    """
    from experiments.uavflow_predictor_idm.alpha_warp import composite_alpha_points
    return composite_alpha_points(rgb,flow,z,radius=radius,points_per_pixel=points_per_pixel)['rgb']
