"""Shared finite-disk alpha-over compositor, used by training and probes.

SynSin-inspired mechanism, not an exact reproduction of the author's renderer:
pixel-space radius=1.5, gamma=1, depth-sorted nearest K=8, black background.
RGB is premultiplied coverage: NEVER divide it by accumulated splat mass.
Footprint membership/depth ordering are discrete; opacity keeps flow gradients.
"""
import math

import torch


def composite_alpha_points(rgb,flow,z,*,radius=1.5,points_per_pixel=8,patch_size=None):
    if not math.isfinite(radius) or radius<=0 or points_per_pixel<1:
        raise ValueError('Positive finite alpha radius and positive K required')
    if rgb.ndim!=4:raise ValueError('Expected RGB [B,C,H,W]')
    b,c,h,w=rgb.shape
    if flow.shape!=(b,2,h,w) or z.shape!=(b,1,h,w):
        raise ValueError('Expected flow [B,2,H,W], target z [B,1,H,W]')
    yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
    pixels=torch.stack((xx,yy),-1).float().reshape(1,h*w,2)
    uv=pixels+flow.permute(0,2,3,1).reshape(b,h*w,2)
    colors=rgb.permute(0,2,3,1).reshape(b*h*w,c)
    reach=math.ceil(radius)
    oy,ox=torch.meshgrid(torch.arange(-reach,reach+1,device=rgb.device),
                         torch.arange(-reach,reach+1,device=rgb.device),indexing='ij')
    offsets=torch.stack((ox,oy),-1).reshape(-1,2)
    coords=uv.detach().floor().long()[:,:,None]+offsets
    d2=(uv[:,:,None]-coords).square().sum(-1)
    good=(coords[...,0]>=0)&(coords[...,0]<w)&(coords[...,1]>=0)&(coords[...,1]<h)&(d2.detach()<radius**2)
    batch=torch.arange(b,device=rgb.device)[:,None,None].expand_as(d2)
    destination=(batch*h*w+coords[...,1]*w+coords[...,0])[good]
    source=torch.arange(b*h*w,device=rgb.device).reshape(b,h*w,1).expand_as(d2)[good]
    alpha=(1-d2[good]/radius**2).clamp(0,1)
    # Exact same deterministic tie-break/order as the tested offline probe.
    by_depth=torch.argsort(z.flatten()[source].detach(),stable=True)
    order=by_depth[torch.argsort(destination[by_depth],stable=True)]
    destination,source,alpha=[v[order] for v in (destination,source,alpha)]
    position=torch.arange(len(destination),device=rgb.device)
    first=torch.ones_like(destination,dtype=torch.bool)
    first[1:]=destination[1:]!=destination[:-1]
    start=torch.where(first,position,0).cummax(0).values
    rank=position-start
    keep=rank<points_per_pixel
    destination,source,alpha,rank=[v[keep] for v in (destination,source,alpha,rank)]
    slots=destination*points_per_pixel+rank
    opacity=rgb.new_zeros(b*h*w*points_per_pixel).scatter(0,slots,alpha).reshape(b*h*w,points_per_pixel)
    values=rgb.new_zeros(b*h*w*points_per_pixel,c).scatter(0,slots[:,None].expand(-1,c),colors[source]).reshape(b*h*w,points_per_pixel,c)
    transmission=torch.cat((torch.ones_like(opacity[:,:1]),(1-opacity[:,:-1]).cumprod(-1)),dim=-1)
    effective=opacity*transmission
    result=(values*effective[:,:,None]).sum(1)
    coverage=effective.sum(-1).reshape(b,1,h,w).clamp(0,1)
    output=dict(rgb=result.reshape(b,h,w,c).permute(0,3,1,2),
                opacity=coverage,valid=(coverage.detach()>1e-6),mass=coverage[:,0])
    if patch_size is not None:
        if h%patch_size or w%patch_size:raise ValueError('Image must match integral patch grid')
        # Correspondence follows ACTUAL alpha-over contributors: a fully
        # occluded farther point has zero weight and cannot enter the CA mask.
        with torch.no_grad():
            contributing=effective.flatten()[slots]>0
            dst,src=destination[contributing],source[contributing]
            n=(h//patch_size)*(w//patch_size)
            dp=dst%(h*w);sp=src%(h*w)
            dst_patch=(dp//w//patch_size)*(w//patch_size)+(dp%w//patch_size)
            src_patch=(sp//w//patch_size)*(w//patch_size)+(sp%w//patch_size)
            matrix=rgb.new_zeros(b*n*n)
            matrix.scatter_add_(0,(dst//(h*w))*n*n+dst_patch*n+src_patch,torch.ones_like(dst,dtype=rgb.dtype))
            output['correspondence']=(matrix.reshape(b,n,n)>0)
    return output
