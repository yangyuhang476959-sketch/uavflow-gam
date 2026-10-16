"""Batched bilinear camera-z splatting and pixel-to-patch transport.

Projection/z-buffer are deliberately detached. Feature multiplication by W
remains differentiable. No future RGB, depth, validity, or pose is used here.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def nominal_intrinsics(height, width, *, fov_degrees=90., device=None):
    if not 0. < float(fov_degrees) < 180.:
        raise ValueError("Horizontal camera FOV must be between 0 and 180 degrees")
    focal = width / (2. * math.tan(math.radians(float(fov_degrees)) / 2.))
    return torch.tensor([[focal, 0., (width - 1) / 2.],
                         [0., focal, (height - 1) / 2.], [0., 0., 1.]],
                        device=device, dtype=torch.float32)


@torch.no_grad()
def heading_motion(actions):
    """[B,K,4] metre/radian yaw-local deltas -> current-from-future body pose."""
    actions = actions.detach().float()
    if actions.ndim != 3 or actions.shape[-1] != 4 or not torch.isfinite(actions).all():
        raise ValueError("Expected finite [B,K,4] actions in metres and radians")
    b = actions.shape[0]
    motion = torch.eye(4, device=actions.device).expand(b, -1, -1).clone()
    for step in actions.unbind(1):
        c, s = step[:, 3].cos(), step[:, 3].sin()
        transform = torch.eye(4, device=actions.device).expand(b, -1, -1).clone()
        transform[:, 0, 0] = transform[:, 1, 1] = c
        transform[:, 0, 1], transform[:, 1, 0] = -s, s
        transform[:, :3, 3] = step[:, :3]
        motion = motion @ transform
    return motion


@torch.no_grad()
def actions_to_camera_transform(actions, heading_from_camera=None):
    motion = heading_motion(actions)
    if heading_from_camera is None:
        extrinsic = motion.new_tensor([[0., 0., 1., 0.], [1., 0., 0., 0.],
                                      [0., -1., 0., 0.], [0., 0., 0., 1.]])
    else:
        extrinsic = torch.as_tensor(heading_from_camera, device=motion.device, dtype=motion.dtype)
        if extrinsic.shape[-2:] != (4, 4) or not torch.isfinite(extrinsic).all():
            raise ValueError("heading_from_camera must be a finite 4x4 extrinsic")
    return torch.linalg.inv(extrinsic) @ torch.linalg.inv(motion) @ extrinsic


def _coarse_support(support, kernel, min_hole_area):
    """Remove thin raster holes; this is a coarse prior, not exact visibility."""
    if kernel < 1 or kernel % 2 != 1 or min_hole_area < 0:
        raise ValueError("Cleanup kernel must be positive/odd and hole area non-negative")
    missing = (~support).float()[:, None]
    if kernel > 1:
        pad = kernel // 2
        eroded = -F.max_pool2d(-F.pad(missing, (pad,) * 4, value=0.), kernel, stride=1)
        missing = F.max_pool2d(eroded, kernel, stride=1, padding=pad)
    if min_hole_area > 1:
        # Only a small binary mask crosses the device boundary. W/features do
        # not move to CPU. SciPy is already a locked project dependency.
        from scipy.ndimage import label
        masks = []
        for mask in missing[:, 0].bool().cpu().numpy():
            components, _ = label(mask)
            import numpy as np
            areas = np.bincount(components.ravel())
            keep = areas >= min_hole_area
            keep[0] = False
            masks.append(torch.from_numpy(keep[components]))
        missing = torch.stack(masks).to(support.device)[:, None]
    return ~missing[:, 0].bool()


@torch.no_grad()
def bilinear_patch_transport(depth, intrinsics, transform, *, valid=None,
                             patch_size=14, z_absolute_tolerance=.01,
                             z_relative_tolerance=.01, cleanup_kernel=1,
                             min_hole_area=0, weight_epsilon=1e-6):
    """Return W[B,N,N], rho_raw/coarse[B,N], and pixel coverage.

    Each floating projection contributes to FOUR neighbours. Each neighbour
    has its own nearest-z competition. Coverage tests nonzero retained mass,
    not a 0.5 bilinear-weight threshold. Rows of W average source-patch values
    over retained mass; rho counts covered target pixels, not source points.
    """
    if depth.ndim != 3:
        raise ValueError("Expected camera z-depth [B,H,W] in metres")
    b, h, w = depth.shape
    if h % patch_size or w % patch_size:
        raise ValueError("Depth size must be divisible by DA3 patch size")
    if min(z_absolute_tolerance, z_relative_tolerance, weight_epsilon) < 0:
        raise ValueError("Splat tolerances must be non-negative")
    device = depth.device
    depth = depth.detach().float()
    k = torch.as_tensor(intrinsics, device=device, dtype=torch.float32)
    t = torch.as_tensor(transform, device=device, dtype=torch.float32)
    if k.ndim == 2:
        k = k.expand(b, -1, -1)
    if t.ndim == 2:
        t = t.expand(b, -1, -1)
    if k.shape != (b, 3, 3) or t.shape != (b, 4, 4):
        raise ValueError("Intrinsics/transform must align with the depth batch")
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    pixel = torch.stack((xx, yy, torch.ones_like(xx)), -1).float().reshape(-1, 3)
    keep = torch.isfinite(depth) & (depth > 0)
    if valid is not None:
        if valid.shape != depth.shape:
            raise ValueError("Current validity must match current depth")
        keep &= valid.bool()
    rays = pixel[None] @ torch.linalg.inv(k).transpose(-1, -2)
    points = rays * torch.where(keep, depth, 0.).flatten(1)[..., None]
    target = points @ t[:, :3, :3].transpose(-1, -2) + t[:, None, :3, 3]
    projected = target @ k.transpose(-1, -2)
    front = keep.flatten(1) & torch.isfinite(target).all(-1) & (target[..., 2] > 1e-6)
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    near_image = front & torch.isfinite(uv).all(-1) & (uv[..., 0] > -1) & (uv[..., 0] < w) & (uv[..., 1] > -1) & (uv[..., 1] < h)
    uv = torch.where(near_image[..., None], uv, 0.)
    base = uv.floor().long()
    fraction = uv - base.float()
    neighbours = depth.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).long()
    coords = base[:, :, None] + neighbours[None, None]
    dx, dy = fraction.unbind(-1)
    weights = torch.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), -1)
    inside = near_image[..., None] & (coords[..., 0] >= 0) & (coords[..., 0] < w) & (coords[..., 1] >= 0) & (coords[..., 1] < h) & (weights > weight_epsilon)
    batch_index = torch.arange(b, device=device)[:, None, None].expand_as(weights)
    destination = batch_index * (h*w) + coords[..., 1] * w + coords[..., 0]
    destination = destination[inside]
    z = target[..., 2, None].expand_as(weights)[inside]
    weight = weights[inside]
    src_patch = ((yy // patch_size) * (w // patch_size) + xx // patch_size).flatten()
    source_patch = src_patch[None, :, None].expand_as(weights)[inside]
    zmin = depth.new_full((b*h*w,), float("inf"))
    zmin.scatter_reduce_(0, destination, z, reduce="amin", include_self=True)
    retain = z <= zmin[destination] + z_absolute_tolerance + z_relative_tolerance * zmin[destination]
    destination, weight, source_patch = destination[retain], weight[retain], source_patch[retain]
    mass = depth.new_zeros(b*h*w)
    mass.scatter_add_(0, destination, weight)
    support = mass.reshape(b, h, w) > weight_epsilon
    n = (h // patch_size) * (w // patch_size)
    batch_dst, pixel_dst = destination // (h*w), destination % (h*w)
    target_patch = (pixel_dst // w // patch_size) * (w // patch_size) + pixel_dst % w // patch_size
    # Normalize within each target pixel before pooling, so magnified areas
    # and collisions cannot count a pixel more than once in the patch mean.
    weight = weight / mass[destination].clamp_min(weight_epsilon)
    flat_w = depth.new_zeros(b*n*n)
    flat_w.scatter_add_(0, batch_dst*n*n + target_patch*n + source_patch, weight)
    matrix = flat_w.reshape(b, n, n)
    row_mass = matrix.sum(-1, keepdim=True)
    matrix = matrix / row_mass.clamp_min(weight_epsilon)
    coarse = _coarse_support(support, int(cleanup_kernel), int(min_hole_area))
    def pool(mask):
        return mask.float().reshape(b, h//patch_size, patch_size, w//patch_size, patch_size).mean((2, 4)).flatten(1)
    raw, rho = pool(support), pool(coarse)
    # Cleanup must never manufacture feature evidence for an empty W row.
    rho = torch.where(row_mass[..., 0] > 0, rho, 0.)
    return {"matrix": matrix, "rho_raw": raw, "rho": rho,
            "pixel_support": support, "pixel_support_coarse": coarse}
