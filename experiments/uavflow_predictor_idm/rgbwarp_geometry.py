"""Small differentiable bilinear forward splat, not inverse grid sampling.

Only discrete raster/visibility/correspondence decisions are detached. RGB
values and bilinear weights keep their depth/action gradients. Visibility is
piecewise differentiable, not a differentiable topology change.
"""
import torch


def forward_alpha_splat(rgb,depth,transform,intrinsics,*,valid=None,patch_size=14,
                        radius=1.5,points_per_pixel=8):
    """Metric projection + finite-disk alpha-over; RGB has no mass division."""
    from .alpha_warp import composite_alpha_points
    if rgb.ndim!=4 or depth.shape!=(rgb.shape[0],*rgb.shape[-2:]):
        raise ValueError('Expected RGB [B,C,H,W], z-depth [B,H,W]')
    b,_,h,w=rgb.shape;device=rgb.device
    with torch.autocast(device_type=device.type,enabled=False):
        rgb,depth,t=rgb.float(),depth.float(),transform.float()
        k=torch.as_tensor(intrinsics,device=device,dtype=torch.float32)
        if k.ndim==2:k=k.expand(b,-1,-1)
        if k.shape!=(b,3,3) or t.shape!=(b,4,4):raise ValueError('Camera matrices must match batch')
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3).float()
        good=torch.isfinite(depth)&(depth>0)
        if valid is not None:good=good&valid.bool()
        points=(pixels[None]@torch.linalg.inv(k).transpose(-1,-2))*torch.where(good,depth,0).flatten(1)[...,None]
        target=points@t[:,:3,:3].transpose(-1,-2)+t[:,None,:3,3]
        projected=target@k.transpose(-1,-2)
        uv=projected[...,:2]/projected[...,2:].clamp_min(1e-6)
        front=good.flatten(1)&torch.isfinite(target).all(-1)&(target[...,2]>1e-6)
        flow=torch.where(front[...,None],uv-pixels[None,:,:2],-float(h+w+10))
        return composite_alpha_points(rgb,flow.reshape(b,h,w,2).permute(0,3,1,2),
            target[...,2].reshape(b,1,h,w),radius=radius,points_per_pixel=points_per_pixel,patch_size=patch_size)


def action_camera_transform(actions, heading_from_camera=None):
    if actions.ndim!=3 or actions.shape[-1]!=4:
        raise ValueError('Expected [B,K,4] local metre/radian increments')
    # Outer BF16 AMP would downcast pose matmuls even after actions.float().
    # Compose and invert camera geometry in FP32, keeping action gradients.
    with torch.autocast(device_type=actions.device.type, enabled=False):
        actions=actions.float();b=actions.shape[0]
        pose=torch.eye(4,device=actions.device).expand(b,-1,-1)
        for step in actions.unbind(1):
            c,s=step[:,3].cos(),step[:,3].sin();z=torch.zeros_like(c);o=torch.ones_like(c)
            rotation=torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(b,3,3)
            upper=torch.cat((rotation,step[:,:3,None]),-1)
            bottom=step.new_tensor([0.,0.,0.,1.]).expand(b,1,4)
            pose=pose @ torch.cat((upper,bottom),-2)
        e=(actions.new_tensor([[0.,0.,1.,0.],[1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,0.,1.]])
           if heading_from_camera is None else torch.as_tensor(heading_from_camera,device=actions.device,dtype=torch.float32))
        if e.shape[-2:]!=(4,4) or not torch.isfinite(e).all():
            raise ValueError('Expected finite camera-to-body extrinsic')
        return torch.linalg.inv(e) @ torch.linalg.inv(pose) @ e



def forward_splat(rgb, depth, transform, intrinsics, *, valid=None, patch_size=14,
                  z_absolute_tolerance=.01,z_relative_tolerance=.01,
                  return_diagnostics=False):
    if rgb.ndim!=4 or depth.shape!=(rgb.shape[0],*rgb.shape[-2:]):
        raise ValueError('Expected RGB [B,C,H,W], z-depth [B,H,W]')
    b,c,h,w=rgb.shape;device=rgb.device
    if h%patch_size or w%patch_size:
        raise ValueError('Image must match integral patch grid')
    # Geometry and accumulation stay FP32 under outer AMP, preserving weights.
    with torch.autocast(device_type=device.type,enabled=False):
        depth=depth.float();rgb=rgb.float();t=transform.float()
        k=torch.as_tensor(intrinsics,device=device,dtype=torch.float32)
        if k.ndim==2:k=k.expand(b,-1,-1)
        if k.shape!=(b,3,3) or t.shape!=(b,4,4):
            raise ValueError('Camera matrices must match batch')
        yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1).reshape(-1,3).float()
        good=torch.isfinite(depth)&(depth>0)
        if valid is not None:good=good & valid.bool()
        points=(pixels[None] @ torch.linalg.inv(k).transpose(-1,-2))*torch.where(good,depth,0).flatten(1)[...,None]
        target=points @ t[:,:3,:3].transpose(-1,-2)+t[:,None,:3,3]
        projected=target @ k.transpose(-1,-2)
        uv=projected[...,:2]/projected[...,2:].clamp_min(1e-6)
        front=good.flatten(1)&torch.isfinite(target).all(-1)&(target[...,2]>1e-6)
        inside=front&torch.isfinite(uv).all(-1)&(uv[...,0]>-1)&(uv[...,0]<w)&(uv[...,1]>-1)&(uv[...,1]<h)
        uv=torch.where(inside[...,None],uv,0.)
        base=uv.detach().floor().long();fraction=uv-base.float()
        offsets=base.new_tensor([[0,0],[1,0],[0,1],[1,1]])
        coords=base[:,:,None]+offsets
        dx,dy=fraction.unbind(-1)
        weights=torch.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),-1)
        keep=inside[...,None]&(coords[...,0]>=0)&(coords[...,0]<w)&(coords[...,1]>=0)&(coords[...,1]<h)&(weights.detach()>1e-6)
        batches=torch.arange(b,device=device)[:,None,None].expand_as(weights)
        destination=(batches*h*w+coords[...,1]*w+coords[...,0])[keep]
        source=torch.arange(h*w,device=device)[None,:,None].expand_as(weights)[keep]
        bi=batches[keep];weight=weights[keep];z=target[...,2,None].expand_as(weights)[keep]
        zmin=depth.new_full((b*h*w,),float('inf'))
        zmin.scatter_reduce_(0,destination,z.detach(),reduce='amin',include_self=True)
        visible=(z.detach()<=zmin[destination]+z_absolute_tolerance+z_relative_tolerance*zmin[destination])
        destination,source,bi,weight,z=[v[visible] for v in (destination,source,bi,weight,z)]
        mass=depth.new_zeros(b*h*w).scatter_add(0,destination,weight)
        colors=rgb.permute(0,2,3,1).reshape(b,h*w,c)[bi,source]
        numerator=rgb.new_zeros(b*h*w,c).scatter_add(0,destination[:,None].expand(-1,c),colors*weight[:,None])
        coverage=(mass.detach()>1e-6).reshape(b,1,h,w)
        result=(numerator/mass[:,None].clamp_min(1e-6)).reshape(b,h,w,c).permute(0,3,1,2)
        zd=depth.new_zeros(b*h*w).scatter_add(0,destination,z*weight)/mass.clamp_min(1e-6)
        n=(h//patch_size)*(w//patch_size)
        dst_pixel=destination%(h*w)
        dst_patch=(dst_pixel//w//patch_size)*(w//patch_size)+(dst_pixel%w//patch_size)
        src_patch=(source//w//patch_size)*(w//patch_size)+(source%w//patch_size)
        matrix=depth.new_zeros(b*n*n)
        matrix.scatter_add_(0,bi*n*n+dst_patch*n+src_patch,torch.ones_like(weight))
        correspondence=(matrix.reshape(b,n,n)>0).detach()
        output=dict(rgb=result,valid=coverage,correspondence=correspondence,
                    depth=zd.reshape(b,h,w),mass=mass.reshape(b,h,w))
        if return_diagnostics:
            # Optional offline audit only; default forward and autograd are
            # unchanged. Frobenius norm of each target RGB's Jacobian w.r.t.
            # its contributing source projected coordinates (u_i,v_i).
            with torch.no_grad():
                count=depth.new_zeros(b*h*w).scatter_add_(0,destination,torch.ones_like(weight))
                du=torch.stack((-(1-dy),1-dy,-dy,dy),-1)[keep][visible]
                dv=torch.stack((-(1-dx),-dx,1-dx,dx),-1)[keep][visible]
                target_rgb=result.permute(0,2,3,1).reshape(-1,c)
                contrast=(colors-target_rgb[destination]).square().sum(-1)
                squared=(du.square()+dv.square())*contrast/mass[destination].clamp_min(1e-6).square()
                local_jacobian=depth.new_zeros(b*h*w).scatter_add_(0,destination,squared).sqrt()
                # Exact normalized single-contributor derivative is zero;
                # exclude FP32 numerator/denominator cancellation residue.
                local_jacobian[count<=1]=0
                output['diagnostics']=dict(contributors=count.reshape(b,h,w),
                    coordinate_jacobian_norm=local_jacobian.reshape(b,h,w))
        return output
