"""CPU reference z-depth forward warp. No model/attention/training changes.

T_target_from_source transforms CV camera points (right, down, forward).
Hard projection and z-buffer are not a differentiable action-training path.
"""
import numpy as np


UNREAL_FROM_CV = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])


def cumulative_heading_motion(actions):
    """[K,4] local metre/radian actions -> current-heading-from-future-heading."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [K,4] metre/radian actions')
    motion = np.eye(4)
    for dx, dy, dz, yaw in actions:
        c, s = np.cos(yaw), np.sin(yaw)
        step = np.eye(4)
        step[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        step[:3, 3] = [dx, dy, dz]
        motion = motion @ step
    return motion


def heading_actions_to_camera_transform(actions, heading_from_camera=None):
    """Default rig is co-located; its axes differ. Offset must be explicit."""
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = UNREAL_FROM_CV
    if heading_from_camera is not None:
        extrinsic = np.asarray(heading_from_camera, dtype=float)
    motion = cumulative_heading_motion(actions)
    return np.linalg.inv(extrinsic) @ np.linalg.inv(motion) @ extrinsic


def pixel_support_to_patch_ratio(support, patch_size=14):
    support = np.asarray(support, dtype=float)
    h, w = support.shape[-2:]
    if h % patch_size or w % patch_size:
        raise ValueError('Image dimensions must be divisible by patch size')
    return support.reshape(*support.shape[:-2], h//patch_size, patch_size,
                           w//patch_size, patch_size).mean(axis=(-3, -1))


def forward_warp_zdepth(depth, intrinsics, transform, valid=None):
    """Single frame reference: nearest pixel splat, nearest camera-z wins.

    Returned source_index is shared by depth/RGB/patch feature transport.
    Unsupported depths are zero; source_index is -1.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        raise ValueError('Expected [H,W] z-depth')
    k, t = np.asarray(intrinsics, float), np.asarray(transform, float)
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    keep = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)
    indices = np.flatnonzero(keep)
    pixels = np.stack([xx.ravel()[indices], yy.ravel()[indices], np.ones(len(indices))], axis=1)
    points = (pixels @ np.linalg.inv(k).T) * depth.ravel()[indices, None]
    target = points @ t[:3, :3].T + t[:3, 3]
    front = np.isfinite(target).all(1) & (target[:, 2] > 1e-6)
    target, indices = target[front], indices[front]
    projected = target @ k.T
    uv = projected[:, :2] / projected[:, 2:3]
    # Filter before integer conversion, including extreme projections.
    inside = (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5)
    uv, target, indices = uv[inside], target[inside], indices[inside]
    rounded = np.floor(uv + .5).astype(np.int64)
    destination = rounded[:, 1]*w + rounded[:, 0]
    order = np.lexsort((indices, target[:, 2], destination))
    sorted_dest = destination[order]
    unique = np.r_[True, sorted_dest[1:] != sorted_dest[:-1]] if len(order) else np.zeros(0, bool)
    winners = order[unique]
    source = np.full(h*w, -1, dtype=np.int64)
    warped = np.zeros(h*w, dtype=np.float32)
    source[destination[winners]] = indices[winners]
    warped[destination[winners]] = target[winners, 2]
    source = source.reshape(h, w)
    support = source >= 0
    return dict(depth=warped.reshape(h, w), support=support, source_index=source,
                rho_support=pixel_support_to_patch_ratio(support))


def transport_patch_geometry(geometry, source_index, patch_size=14):
    """Expand source patch values to pixels, z-buffer winners, supported mean.

    geometry: [N,C], row-major patch order. Does not transport prefix tokens.
    """
    h, w = source_index.shape
    hg, wg = h//patch_size, w//patch_size
    if h % patch_size or w % patch_size or geometry.shape[0] != hg*wg:
        raise ValueError('Geometry/grid mismatch')
    yy, xx = np.indices((h, w))
    valid = source_index >= 0
    src = source_index[valid]
    source_patch = (src//w//patch_size)*wg + (src%w//patch_size)
    target_patch = (yy[valid]//patch_size)*wg + xx[valid]//patch_size
    result = np.zeros((hg*wg, geometry.shape[1]), dtype=np.float64)
    count = np.zeros(hg*wg, dtype=np.float64)
    np.add.at(result, target_patch, geometry[source_patch])
    np.add.at(count, target_patch, 1)
    result /= np.maximum(count[:, None], 1)
    rho = count/(patch_size*patch_size)
    return result.astype(geometry.dtype), rho
