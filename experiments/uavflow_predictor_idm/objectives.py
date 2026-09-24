"""Stage-2 forward contract, training objectives and offline evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from robot.data.dataset import ActionNormalizer
from robot.modeling.da3_giant_encoder import DA3GiantEncoder
from robot.modeling.lora import LoRALinear

from .data import EpisodePoseNormalizer, move_batch
from .vlm_conditioning import encode_stage2_condition


@contextmanager
def _base_da3_teacher_without_lora(da3: DA3GiantEncoder):
    """Temporarily evaluate the immutable base DA3, excluding student LoRA.

    Predictor shallow features stop before the LoRA insertion boundary.  The
    same DA3 instance can therefore provide a stable no-grad teacher without a
    second Giant backbone: set every LoRA residual scale to zero only while
    propagating the real image window, then restore the student adapters.
    """
    adapters = [module for module in da3.modules() if isinstance(module, LoRALinear)]
    scales = [module.scaling for module in adapters]
    try:
        for module in adapters:
            module.scaling = 0.0
        yield
    finally:
        for module, scale in zip(adapters, scales):
            module.scaling = scale


def _decode_visual_depth_levels(
    da3: DA3GiantEncoder,
    deep_levels: list[torch.Tensor],
    *,
    batch_size: int,
    steps: int,
    views: int,
) -> torch.Tensor:
    """Decode action-free `(B,T,V,N,C)` DA3 levels to `(B,T,V,H,W)`."""
    patch_start = 1 + int(getattr(da3, "num_register_tokens", 0))
    dpt_levels = []
    for level in deep_levels:
        if level.ndim != 5:
            raise ValueError(f"Expected teacher deep level (B,T,V,N,C), got {level.shape}.")
        patches = level[:, :, :, patch_start:].reshape(
            batch_size * steps * views, level.shape[3] - patch_start, level.shape[-1]
        )
        cls = level[:, :, :, 0].reshape(batch_size * steps * views, level.shape[-1])
        dpt_levels.append((patches, cls))
    decoded = da3.decode_depth_full(
        dpt_levels,
        batch_size=batch_size,
        views_per_sequence=steps * views,
        frames_chunk_size=steps * views,
    )["depth"]
    return decoded.reshape(batch_size, steps, views, *decoded.shape[-2:])


def da3_native_pseudo_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    grad_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Match the frozen base-DA3 native depth of a jointly encoded window."""
    if target.shape[-2:] != prediction.shape[-2:]:
        shape = target.shape
        target = torch.nn.functional.interpolate(
            target.flatten(0, -3)[:, None].float(),
            size=prediction.shape[-2:],
            mode="nearest",
        )[:, 0].reshape(*shape[:-2], *prediction.shape[-2:])
    prediction = prediction.float()
    target = target.float().detach()
    valid = (
        torch.isfinite(prediction) & torch.isfinite(target)
        & (prediction > 0) & (target > 0)
    )
    error = prediction - target
    depth = (error.abs() * valid).sum() / valid.float().sum().clamp_min(1.0)
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    grad_x = error[..., :, 1:] - error[..., :, :-1]
    grad_y = error[..., 1:, :] - error[..., :-1, :]
    grad = (
        (grad_x.abs() * valid_x).sum() + (grad_y.abs() * valid_y).sum()
    ) / (valid_x.float().sum() + valid_y.float().sum()).clamp_min(1.0)
    return {
        "total": depth + float(grad_weight) * grad,
        "l1": depth.detach(),
        "grad": grad.detach(),
        "valid_ratio": valid.float().mean().detach(),
    }


def scale_invariant_depth_loss(
    prediction: torch.Tensor,
    target_meters: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    grad_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Supervise true metric depth without assuming DA3's monocular scale.

    A detached per-frame median ratio aligns the GT scale to the frozen DPT
    head's native scale.  Spatial depth and depth-gradient errors still send
    gradients through frozen DPT/deeper DA3 into predicted future tokens.
    """
    if prediction.shape[-2:] != target_meters.shape[-2:]:
        size = prediction.shape[-2:]
        target_meters = torch.nn.functional.interpolate(
            target_meters.flatten(0, -3)[:, None].float(), size=size, mode="nearest"
        )[:, 0].reshape(*target_meters.shape[:-2], *size)
        valid_mask = torch.nn.functional.interpolate(
            valid_mask.flatten(0, -3)[:, None].float(), size=size, mode="nearest"
        )[:, 0].bool().reshape(*valid_mask.shape[:-2], *size)
    prediction = prediction.float()
    target_meters = target_meters.float()
    valid = (
        valid_mask.bool() & torch.isfinite(prediction) & torch.isfinite(target_meters)
        & (prediction > 0) & (target_meters > 0)
    )
    flat_pred = prediction.flatten(-2)
    flat_target = target_meters.flatten(-2)
    flat_valid = valid.flatten(-2)
    ratios = []
    for pred_row, target_row, mask_row in zip(
        flat_pred.reshape(-1, flat_pred.shape[-1]),
        flat_target.reshape(-1, flat_target.shape[-1]),
        flat_valid.reshape(-1, flat_valid.shape[-1]),
    ):
        if bool(mask_row.any()):
            ratios.append(
                (pred_row[mask_row].median() / target_row[mask_row].median().clamp_min(1e-6)).detach()
            )
        else:
            ratios.append(pred_row.new_tensor(1.0))
    ratio = torch.stack(ratios).reshape(*prediction.shape[:-2], 1, 1)
    aligned_target = target_meters * ratio
    log_error = (
        prediction.clamp_min(1e-6).log() - aligned_target.clamp_min(1e-6).log()
    )
    denom = valid.float().sum().clamp_min(1.0)
    depth = (log_error.abs() * valid).sum() / denom
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    grad_x = log_error[..., :, 1:] - log_error[..., :, :-1]
    grad_y = log_error[..., 1:, :] - log_error[..., :-1, :]
    grad = (
        (grad_x.abs() * valid_x).sum() + (grad_y.abs() * valid_y).sum()
    ) / (valid_x.float().sum() + valid_y.float().sum()).clamp_min(1.0)
    return {
        "total": depth + float(grad_weight) * grad,
        "l1": depth.detach(),
        "grad": grad.detach(),
        "valid_ratio": valid.float().mean().detach(),
    }


def gam_window_pointnorm_depth_loss(
    prediction: torch.Tensor,
    target_meters: torch.Tensor,
    valid_mask: torch.Tensor,
    scale_window_meters: torch.Tensor,
    scale_window_mask: torch.Tensor,
    *,
    semantic_mask: torch.Tensor | None = None,
    linear_weight: float = 0.5,
    log_weight: float = 1.0,
    log_epsilon_normalized: float = 1e-6,
    grad_weight: float = 1.0,
    semantic_weight: float = 3.0,
    semantic_separate_weight: float = 0.0,
    semantic_linear_weight: float | None = None,
    semantic_log_weight: float | None = None,
    semantic_grad_weight: float | None = None,
    gradient_mode: str = "tolerant_log",
    gradient_tolerance_pixels: int = 2,
    camera_fov_degrees: float = 90.0,
    fixed_scale_meters: float | None = None,
    predicted_log_scale: torch.Tensor | None = None,
    scale_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """GAM-style one-scale-per-window depth supervision.

    The scale is the mean L2 norm of valid pinhole-unprojected points over the
    complete observed+next-target window.  UAV-Flow-Sim uses one fixed 90-degree
    front camera; using camera-local point norms is sufficient for the quick
    GAM ablation and, unlike per-frame median alignment, preserves temporal
    approach/retreat inside H=1/2/3 windows.
    """
    size = prediction.shape[-2:]

    def _resize(values: torch.Tensor, *, mask: bool = False) -> torch.Tensor:
        if values.shape[-2:] == size:
            return values.bool() if mask else values.float()
        flat = values.flatten(0, -3)[:, None].float()
        resized = torch.nn.functional.interpolate(flat, size=size, mode="nearest")[:, 0]
        resized = resized.reshape(*values.shape[:-2], *size)
        return resized.bool() if mask else resized

    prediction = prediction.float()
    target_meters = _resize(target_meters)
    valid_mask = _resize(valid_mask, mask=True)
    scale_window_meters = _resize(scale_window_meters)
    scale_window_mask = _resize(scale_window_mask, mask=True)
    if semantic_mask is None:
        semantic_mask = torch.zeros_like(valid_mask)
    else:
        semantic_mask = _resize(semantic_mask, mask=True)

    height, width = size
    focal = 0.5 * float(width) / torch.tan(
        prediction.new_tensor(float(camera_fov_degrees) * torch.pi / 360.0)
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=prediction.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=prediction.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    ray_norm = torch.sqrt(
        ((xx - 0.5 * float(width)) / focal) ** 2
        + ((yy - 0.5 * float(height)) / focal) ** 2
        + 1.0
    )
    scale_valid = (
        scale_window_mask
        & torch.isfinite(scale_window_meters)
        & (scale_window_meters > 0)
    )
    point_norm = scale_window_meters * ray_norm.view(
        *([1] * (scale_window_meters.ndim - 2)), height, width
    )
    reduce_dims = tuple(range(1, point_norm.ndim))
    measured_scene_scale = (
        (point_norm * scale_valid).sum(dim=reduce_dims)
        / scale_valid.float().sum(dim=reduce_dims).clamp_min(1.0)
    ).clamp_min(1e-3).detach()
    if fixed_scale_meters is None:
        scene_scale = measured_scene_scale
    else:
        if float(fixed_scale_meters) <= 0.0:
            raise ValueError("fixed_scale_meters must be positive.")
        scene_scale = measured_scene_scale.new_full(
            measured_scene_scale.shape, float(fixed_scale_meters)
        )
    scale_view = scene_scale.view(-1, *([1] * (target_meters.ndim - 1)))
    target = target_meters / scale_view
    valid = (
        valid_mask
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    pixel_weight = 1.0 + (float(semantic_weight) - 1.0) * semantic_mask.float()
    weighted_valid = pixel_weight * valid.float()
    denom = weighted_valid.sum().clamp_min(1.0)
    error = prediction - target
    linear = (error.abs() * weighted_valid).sum() / denom
    eps = float(log_epsilon_normalized)
    if eps <= 0.0:
        raise ValueError("log_epsilon_normalized must be positive.")
    log_prediction = prediction.clamp_min(eps).log()
    log_target = target.clamp_min(eps).log()
    log_l1 = ((log_prediction - log_target).abs() * weighted_valid).sum() / denom
    gradient_mode = str(gradient_mode).lower()
    if gradient_mode == "gam_l1":
        grad = _finite_difference_gradient_l1(
            prediction, target, valid, pixel_weight
        )
    elif gradient_mode == "tolerant_log":
        grad = _tolerant_symmetric_log_gradient_loss(
            log_prediction,
            log_target,
            valid,
            pixel_weight,
            radius=int(gradient_tolerance_pixels),
        )
    else:
        raise ValueError(
            f"Unsupported depth gradient_mode={gradient_mode!r}; "
            "expected 'gam_l1' or 'tolerant_log'."
        )
    # A global pixel-weight alone barely changes optimization when semantic
    # objects occupy ~1% of an image.  This separately normalized term gives
    # every batch with a target object a non-vanishing semantic objective.
    semantic_valid = semantic_mask & valid
    semantic_denominator = semantic_valid.float().sum().clamp_min(1.0)
    semantic_linear = (
        error.abs() * semantic_valid.float()
    ).sum() / semantic_denominator
    semantic_log = (
        (log_prediction - log_target).abs() * semantic_valid.float()
    ).sum() / semantic_denominator
    if gradient_mode == "gam_l1":
        semantic_grad = _finite_difference_gradient_l1(
            prediction,
            target,
            valid,
            semantic_valid.float(),
            pair_weight_mode="max",
        )
    else:
        semantic_grad = _tolerant_symmetric_log_gradient_loss(
            log_prediction,
            log_target,
            valid,
            semantic_valid.float(),
            radius=int(gradient_tolerance_pixels),
        )
    semantic_total = (
        float(linear_weight if semantic_linear_weight is None else semantic_linear_weight)
        * semantic_linear
        + float(log_weight if semantic_log_weight is None else semantic_log_weight)
        * semantic_log
        + float(grad_weight if semantic_grad_weight is None else semantic_grad_weight)
        * semantic_grad
    )
    global_total = (
        float(linear_weight) * linear
        + float(log_weight) * log_l1
        + float(grad_weight) * grad
    )
    if predicted_log_scale is None:
        scale_l1 = global_total.new_zeros(())
    else:
        predicted_log_scale = predicted_log_scale.float().reshape(-1)
        if predicted_log_scale.shape != measured_scene_scale.shape:
            raise ValueError(
                "predicted_log_scale and scene scale disagree: "
                f"{tuple(predicted_log_scale.shape)} vs "
                f"{tuple(measured_scene_scale.shape)}"
            )
        scale_l1 = torch.nn.functional.l1_loss(
            predicted_log_scale, measured_scene_scale.log()
        )
        global_total = global_total + float(scale_loss_weight) * scale_l1
    return {
        "total": (
            global_total
            + float(semantic_separate_weight) * semantic_total
        ),
        # Keep these two differentiable so callers can assign independent
        # top-level weights to global and dynamic-object depth supervision.
        "global_total": global_total,
        "scale_l1": scale_l1,
        "scene_scale_mean": measured_scene_scale.mean().detach(),
        "l1": linear.detach(),
        "linear_l1": linear.detach(),
        "log_l1": log_l1.detach(),
        "grad": grad.detach(),
        "valid_ratio": valid.float().mean().detach(),
        "semantic_ratio": (semantic_mask & valid).float().mean().detach(),
        "semantic_total": semantic_total,
        "semantic_linear_l1": semantic_linear.detach(),
        "semantic_log_l1": semantic_log.detach(),
        "semantic_grad": semantic_grad.detach(),
        "scene_scale": scene_scale.mean().detach(),
    }


def relative_pose_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise the future cumulative pose, both poses relative to F0."""
    zero = target.new_zeros(())
    if prediction is None:
        return {
            "total": zero, "translation": zero,
            "translation_axes": target.new_zeros(3),
            "yaw": zero, "unit": zero,
        }
    if prediction.shape != target.shape:
        raise ValueError(
            f"relative_pose={tuple(prediction.shape)} but target={tuple(target.shape)}."
        )
    valid_f = valid.float()
    denominator = valid_f.sum().clamp_min(1.0)
    translation_axis_per = torch.nn.functional.smooth_l1_loss(
        prediction[..., :3].float(), target[..., :3], reduction="none"
    )
    translation_per = translation_axis_per.mean(dim=-1)
    yaw_per = torch.nn.functional.smooth_l1_loss(
        prediction[..., 3:].float(), target[..., 3:], reduction="none"
    ).mean(dim=-1)
    unit_per = (prediction[..., 3:].float().norm(dim=-1) - 1.0).abs()
    translation = (translation_per * valid_f).sum() / denominator
    translation_axes = (
        translation_axis_per * valid_f.unsqueeze(-1)
    ).sum(dim=tuple(range(translation_axis_per.ndim - 1))) / denominator
    yaw = (yaw_per * valid_f).sum() / denominator
    unit = (unit_per * valid_f).sum() / denominator
    return {
        "total": translation + yaw + 0.1 * unit,
        "translation": translation.detach(),
        "translation_axes": translation_axes.detach(),
        "yaw": yaw.detach(),
        "unit": unit.detach(),
    }


def rollout_yaw4d_actions_from_episode_pose(
    current_pose: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Roll local 4DoF actions into the episode-F0 coordinate system.

    ``current_pose`` is ``[...,5] = [x,y,z,sin(yaw),cos(yaw)]`` relative to
    F0 and ``actions`` is ``[...,C,4]`` in the policy's raw yaw-only local
    convention. The result remains relative to the same F0.
    """
    if current_pose.shape[-1] != 5 or actions.shape[-1] != 4:
        raise ValueError(
            f"Expected pose [...,5] and actions [...,C,4], got "
            f"{tuple(current_pose.shape)} and {tuple(actions.shape)}."
        )
    position = current_pose[..., :3].float()
    yaw = torch.atan2(current_pose[..., 3].float(), current_pose[..., 4].float())
    for chunk_index in range(int(actions.shape[-2])):
        step = actions[..., chunk_index, :].float()
        cosine = yaw.cos()
        sine = yaw.sin()
        position = position + torch.stack([
            cosine * step[..., 0] - sine * step[..., 1],
            sine * step[..., 0] + cosine * step[..., 1],
            step[..., 2],
        ], dim=-1)
        yaw = yaw + step[..., 3]
    return torch.cat([
        position, yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1)
    ], dim=-1)


def _finite_difference_gradient_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    *,
    pair_weight_mode: str = "mean",
) -> torch.Tensor:
    """GAM/DA3-style same-pixel finite-difference depth-gradient L1."""
    pair_weight_mode = str(pair_weight_mode).lower()
    if pair_weight_mode not in {"mean", "max"}:
        raise ValueError(
            f"Unsupported pair_weight_mode={pair_weight_mode!r}; expected 'mean' or 'max'."
        )

    def pair_weight(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if pair_weight_mode == "max":
            # Select a pair when either endpoint is semantic, explicitly
            # supervising object/background contours without tolerance.
            return torch.maximum(first, second)
        return 0.5 * (first + second)

    losses = []
    dx_valid = valid[..., :, 1:] & valid[..., :, :-1]
    if dx_valid.any():
        dx_error = (
            (prediction[..., :, 1:] - prediction[..., :, :-1])
            - (target[..., :, 1:] - target[..., :, :-1])
        ).abs()
        dx_weight = pair_weight(weight[..., :, 1:], weight[..., :, :-1])
        dx_weight = dx_weight * dx_valid.float()
        losses.append((dx_error * dx_weight).sum() / dx_weight.sum().clamp_min(1.0))
    dy_valid = valid[..., 1:, :] & valid[..., :-1, :]
    if dy_valid.any():
        dy_error = (
            (prediction[..., 1:, :] - prediction[..., :-1, :])
            - (target[..., 1:, :] - target[..., :-1, :])
        ).abs()
        dy_weight = pair_weight(weight[..., 1:, :], weight[..., :-1, :])
        dy_weight = dy_weight * dy_valid.float()
        losses.append((dy_error * dy_weight).sum() / dy_weight.sum().clamp_min(1.0))
    if not losses:
        return prediction.new_zeros(())
    # Match the original GAM helper: horizontal and vertical losses are summed.
    return torch.stack(losses).sum()


def _tolerant_symmetric_log_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Symmetric spatially tolerant Sobel loss over every valid image region.

    A small local search absorbs the known one-to-two-pixel replay/PnP
    registration noise.  The reverse term prevents a predictor from satisfying
    GT edges by hallucinating arbitrary nearby edges. Invalid-depth boundaries
    are removed by a 3x3 validity erosion before matching.
    """
    import torch.nn.functional as F

    shape = prediction.shape
    pred = prediction.reshape(-1, 1, *shape[-2:])
    gt = target.reshape(-1, 1, *shape[-2:])
    mask = valid.reshape(-1, 1, *shape[-2:]).bool()
    weights = weight.reshape(-1, 1, *shape[-2:]).float()
    fully_valid = F.avg_pool2d(mask.float(), 3, stride=1, padding=1) >= (1.0 - 1e-6)
    sobel_x = pred.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    pred_grad = torch.cat([
        F.conv2d(pred, sobel_x, padding=1),
        F.conv2d(pred, sobel_y, padding=1),
    ], dim=1)
    gt_grad = torch.cat([
        F.conv2d(gt, sobel_x, padding=1),
        F.conv2d(gt, sobel_y, padding=1),
    ], dim=1)

    def shift(values: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
        h, w = values.shape[-2:]
        padded = F.pad(
            values,
            (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)),
            value=fill,
        )
        y0 = max(-dy, 0)
        x0 = max(-dx, 0)
        return padded[..., y0 : y0 + h, x0 : x0 + w]

    def directed(source_grad, source_valid, candidate_grad, candidate_valid):
        best = torch.full_like(source_valid.float(), float("inf"))
        candidate_exists = torch.zeros_like(source_valid)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted_grad = shift(candidate_grad, dy, dx, 0.0)
                shifted_valid = shift(candidate_valid, dy, dx, 0.0).bool()
                error = (source_grad - shifted_grad).abs().mean(dim=1, keepdim=True)
                best = torch.minimum(best, torch.where(shifted_valid, error, torch.inf))
                candidate_exists |= shifted_valid
        use = source_valid & candidate_exists & torch.isfinite(best)
        weighted = weights * use.float()
        return (best.nan_to_num(posinf=0.0) * weighted).sum() / weighted.sum().clamp_min(1.0)

    forward = directed(gt_grad, fully_valid, pred_grad, fully_valid)
    backward = directed(pred_grad, fully_valid, gt_grad, fully_valid)
    return 0.5 * (forward + backward)


def deep_feature_loss(
    prediction: dict[str, object],
    target: dict[str, object],
    *,
    patch_start: int,
    patch_weight: float,
    cls_weight: float,
) -> dict[str, torch.Tensor]:
    """Cosine-distill the predicted final frame at DA3's IDM deep levels.

    Only the newly predicted frame is supervised. Historical frames are real
    observations in both branches and would otherwise dominate the loss with
    trivial exact matches. Registers are excluded because the frozen IDM head
    consumes CLS + patches, not register tokens.
    """
    pred_levels = prediction.get("deep_levels")
    target_levels = target.get("deep_levels")
    if not isinstance(pred_levels, (list, tuple)) or not isinstance(target_levels, (list, tuple)):
        raise TypeError("Deep feature loss requires deep_levels from both DA3 branches.")
    if len(pred_levels) != len(target_levels) or not pred_levels:
        raise ValueError(
            f"Deep level mismatch: prediction={len(pred_levels)}, target={len(target_levels)}"
        )
    patch_distances = []
    cls_distances = []
    for predicted_level, target_level in zip(pred_levels, target_levels):
        # (B,T,V,N,D): compare only the final future frame. Target is detached
        # by the no-grad teacher propagation in forward_batch.
        predicted_last = predicted_level[:, -1].float()
        target_last = target_level[:, -1].float()
        cls_distances.append(
            1.0 - torch.nn.functional.cosine_similarity(
                predicted_last[:, :, :1], target_last[:, :, :1], dim=-1
            ).mean()
        )
        patch_distances.append(
            1.0 - torch.nn.functional.cosine_similarity(
                predicted_last[:, :, int(patch_start) :],
                target_last[:, :, int(patch_start) :],
                dim=-1,
            ).mean()
        )
    patch = torch.stack(patch_distances).mean()
    cls = torch.stack(cls_distances).mean()
    denominator = max(float(patch_weight) + float(cls_weight), 1e-8)
    total = (float(patch_weight) * patch + float(cls_weight) * cls) / denominator
    return {
        "total": total,
        "patch": patch,
        "cls": cls,
        "cosine": 1.0 - total,
        "per_level": (
            float(patch_weight) * torch.stack(patch_distances)
            + float(cls_weight) * torch.stack(cls_distances)
        ) / denominator,
    }


def feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_start: int,
    horizon_weights: list[float],
    patch_weight: float,
    cls_weight: float,
    register_weight: float,
) -> dict[str, torch.Tensor]:
    """Separately supervise CLS/register/patch tokens across the rollout."""
    prediction = prediction.float()
    target = target.float()
    weights = torch.as_tensor(
        horizon_weights, device=prediction.device, dtype=prediction.dtype
    )
    if weights.numel() == 1 and prediction.shape[1] > 1:
        weights = weights.expand(prediction.shape[1])
    if weights.numel() != prediction.shape[1]:
        raise ValueError(
            f"feature_horizon_weights={weights.numel()} but rollout={prediction.shape[1]}"
        )

    def horizon_mse(start: int, end: int) -> torch.Tensor:
        if end <= start:
            return prediction.new_zeros(prediction.shape[1])
        return (
            prediction[:, :, :, start:end] - target[:, :, :, start:end]
        ).square().mean(dim=(0, 2, 3, 4))

    cls_per_horizon = horizon_mse(0, 1)
    register_per_horizon = horizon_mse(1, int(patch_start))
    patch_per_horizon = horizon_mse(int(patch_start), int(prediction.shape[3]))
    total_per_horizon = (
        float(patch_weight) * patch_per_horizon
        + float(cls_weight) * cls_per_horizon
        + float(register_weight) * register_per_horizon
    )
    denominator = weights.sum().clamp_min(1e-8)

    def weighted(values: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / denominator

    cls = weighted(cls_per_horizon)
    registers = weighted(register_per_horizon)
    patches = weighted(patch_per_horizon)
    total = weighted(total_per_horizon)
    patch_prediction = prediction[:, :, :, patch_start:]
    patch_target = target[:, :, :, patch_start:]
    cosine_per_horizon = torch.nn.functional.cosine_similarity(
        patch_prediction, patch_target, dim=-1
    ).mean(dim=(0, 2, 3))
    cosine = weighted(cosine_per_horizon)
    return {
        "total": total,
        "patch": patches,
        "cls": cls,
        "register": registers,
        "cosine": cosine,
        "per_horizon": total_per_horizon,
        "patch_per_horizon": patch_per_horizon,
        "cls_per_horizon": cls_per_horizon,
        "register_per_horizon": register_per_horizon,
        "cosine_per_horizon": cosine_per_horizon,
    }


def architecture_feature_loss(mode, output, observed, target_future, **kwargs):
    """Only detached real features are targets; E averages its two targets."""
    target = observed.detach() if mode in {"current_prediction", "direct_current"} else target_future.detach()
    values = feature_loss(output["future_shallow"], target, **kwargs)
    if mode == "direct_current":
        # No feature predictor exists in B. Do not fake a future reconstruction objective.
        return {key: torch.zeros_like(value) for key, value in values.items()}
    if mode == "dual_predicted":
        current = feature_loss(output["predicted_current_shallow"], observed.detach(), **kwargs)
        return {key: (value + current[key]) / 2 for key, value in values.items()}
    return values


def masked_action_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    error = (prediction.float() - target.float()).abs()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error).bool()
    return torch.where(mask, error, torch.zeros_like(error)).sum() / mask.float().sum().clamp_min(1.0)


def masked_action_l1_per_sample_step(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return one L1 value for every sample and causal transition."""
    error = (prediction.float() - target.float()).abs()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error).bool()
    reduce_dims = tuple(range(2, error.ndim))
    numerator = torch.where(mask, error, torch.zeros_like(error)).sum(dim=reduce_dims)
    denominator = mask.float().sum(dim=reduce_dims).clamp_min(1.0)
    return numerator / denominator


def masked_action_l1_per_chunk_slot(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return one L1 value for each action slot in the predicted chunk."""
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(-2)
        target = target.unsqueeze(-2)
        mask = mask.unsqueeze(-2)
    error = (prediction.float() - target.float()).abs()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error).bool()
    reduce_dims = (0, 1, 3)
    numerator = torch.where(mask, error, torch.zeros_like(error)).sum(dim=reduce_dims)
    denominator = mask.float().sum(dim=reduce_dims).clamp_min(1.0)
    return numerator / denominator


def prepare_observed_action_history(
    *,
    model: torch.nn.Module,
    normalizer: ActionNormalizer,
    batch: dict[str, Any],
    context_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Normalize only already-executed actions and preserve their validity mask."""
    model_ref = model.module if hasattr(model, "module") else model
    if not bool(getattr(model_ref, "use_action_history", False)):
        return None, None
    if "past_action_history" not in batch:
        raise KeyError("use_action_history=true but batch has no past_action_history.")
    history = normalizer.normalize(
        batch["past_action_history"][:, :context_len].float(),
        stats_keys=list(batch["action_stats_key"]),
    )
    valid = batch.get("past_action_history_mask")
    if valid is None:
        valid = torch.ones_like(history, dtype=torch.bool)
    else:
        valid = valid[:, :context_len].bool()
        broadcast_valid = valid
        while broadcast_valid.ndim < history.ndim:
            broadcast_valid = broadcast_valid.unsqueeze(-1)
        history = torch.where(broadcast_valid, history, torch.zeros_like(history))
    return history, valid


def prepare_observed_pose_history(
    *,
    model: torch.nn.Module,
    pose_normalizer: EpisodePoseNormalizer | None,
    batch: dict[str, Any],
    context_len: int,
    conditioning_generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Return normalized episode-start-relative pose for observed anchors."""
    model_ref = model.module if hasattr(model, "module") else model
    if not bool(getattr(model_ref, "use_pose_history", False)):
        return None
    if pose_normalizer is None:
        raise ValueError("use_pose_history=true requires an EpisodePoseNormalizer.")
    if "episode_pose" not in batch:
        raise KeyError("use_pose_history=true but batch has no episode_pose.")
    pose = batch["episode_pose"][:, :context_len].float()
    if pose.shape[-1] != 5:
        raise ValueError(f"Expected episode_pose (...,5), got {tuple(pose.shape)}.")
    if model_ref.training:
        xyz_noise_meters = float(
            getattr(model_ref, "pose_history_xyz_noise_meters", 0.0)
        )
        yaw_noise_degrees = float(
            getattr(model_ref, "pose_history_yaw_noise_degrees", 0.0)
        )
        if xyz_noise_meters > 0.0 or yaw_noise_degrees > 0.0:
            xyz = pose[..., :3]
            if xyz_noise_meters > 0.0:
                xyz = xyz + torch.randn(
                    xyz.shape, device=xyz.device, dtype=xyz.dtype,
                    generator=conditioning_generator,
                ) * xyz_noise_meters
            yaw = torch.atan2(pose[..., 3], pose[..., 4])
            if yaw_noise_degrees > 0.0:
                yaw = yaw + torch.randn(
                    yaw.shape, device=yaw.device, dtype=yaw.dtype,
                    generator=conditioning_generator,
                ) * (yaw_noise_degrees * torch.pi / 180.0)
            pose = torch.cat([
                xyz, yaw.sin().unsqueeze(-1), yaw.cos().unsqueeze(-1)
            ], dim=-1)
    return pose_normalizer.normalize(pose)


def forward_batch(
    *,
    model: torch.nn.Module,
    da3: DA3GiantEncoder,
    text: torch.nn.Module,
    normalizer: ActionNormalizer,
    pose_normalizer: EpisodePoseNormalizer | None,
    batch: dict[str, Any],
    context_len: int,
    rollout_steps: int,
    feature_horizon_weights: list[float],
    feature_patch_weight: float,
    feature_cls_weight: float,
    feature_register_weight: float,
    deep_feature_enabled: bool,
    deep_feature_patch_weight: float,
    deep_feature_cls_weight: float,
    stop_pos_weight: float,
    amp: bool,
    action_direct_weight: float = 1.0,
    action_refine_weight: float = 1.0,
    depth_target_source: str = "ue_gt",
    depth_scale_mode: str = "per_frame_median",
    depth_target_mode: str = "future",
    depth_fixed_scale_meters: float = 100.0,
    depth_scale_loss_weight: float = 1.0,
    depth_linear_weight: float = 0.5,
    depth_log_weight: float = 1.0,
    depth_log_epsilon_normalized: float = 1e-6,
    depth_grad_weight: float = 1.0,
    depth_semantic_weight: float = 3.0,
    depth_semantic_separate_weight: float = 0.0,
    depth_semantic_linear_weight: float | None = None,
    depth_semantic_log_weight: float | None = None,
    depth_semantic_grad_weight: float | None = None,
    depth_gradient_mode: str = "tolerant_log",
    depth_gradient_tolerance_pixels: int = 2,
    conditioning_generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Run sliding-IDM training or evaluate a legacy multi-step checkpoint."""
    images = batch["all_view_images"]
    model_ref = model.module if hasattr(model, "module") else model
    use_reference = bool(getattr(model_ref, "use_fixed_first_frame", False))
    reference_images = batch.get("episode_first_image")
    if use_reference and reference_images is None:
        raise KeyError("Fixed-F0 Stage 2 requires episode_first_image in every batch.")
    batch_size, total, views = images.shape[:3]
    feature_target_offset = int(getattr(model_ref, "feature_target_offset", 1))
    needed = int(context_len) + feature_target_offset
    if total < needed:
        raise ValueError(f"Need {needed} frames for H={context_len}, got {total}.")

    # Frozen teachers encode all images once. Future tokens leave this function
    # only as detached loss targets; the predictor receives observed tokens.
    encoded_images = (
        torch.cat([reference_images[:, None], images[:, :needed]], dim=1)
        if use_reference else images[:, :needed]
    )
    encoded_steps = needed + int(use_reference)
    flat = encoded_images.reshape(batch_size, encoded_steps * views, *images.shape[3:])
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(amp and images.is_cuda)
    ):
        encoded_shallow = da3.encode_shallow_visual_slots(flat, T=encoded_steps, V=views)["visual_tokens"]
        language = encode_stage2_condition(
            text,
            list(batch["task_description"]),
            reference_images=(
                reference_images if use_reference else images[:, 0]
            ),
            current_images=images[:, context_len - 1],
            pad_to=int(getattr(model_ref.predictor, "language_len", 77)),
            current_pose=(
                batch["openvla_prompt_pose"][:, context_len - 1]
                if "openvla_prompt_pose" in batch else None
            ),
        )
    reference_shallow = encoded_shallow[:, :1] if use_reference else None
    shallow_all = encoded_shallow[:, 1:] if use_reference else encoded_shallow
    observed = shallow_all[:, :context_len]
    dense_context = bool(getattr(model_ref, "dense_context_supervision", False))
    if dense_context:
        # Prediction at observed anchor i targets F(i+offset). For a K-action
        # chunk this is normally F(i+K), so the visual target represents the
        # state reached after executing the complete predicted plan.
        target_future = shallow_all[
            :, feature_target_offset : feature_target_offset + context_len
        ].detach()
    else:
        target_future = shallow_all[:, context_len : context_len + rollout_steps].detach()
    depth_enabled = bool(getattr(model_ref, "depth_decode_enabled", False))
    teacher_window_depth = None
    if depth_enabled and str(depth_target_source) == "da3_teacher_window":
        # Jointly propagate the complete real window [F_s, ..., F_(s+H)]
        # through the immutable base DA3.  For dense H supervision the target
        # slice below is [F_(s+1), ..., F_(s+H)].  The fixed episode-reference
        # image is deliberately not part of this pseudo-label window.
        with torch.no_grad(), _base_da3_teacher_without_lora(da3):
            teacher_features = da3.propagate_shallow_visual_slots_grad(
                shallow_all[:, :needed].detach(), gradient_checkpointing=False
            )
            teacher_window_depth = _decode_visual_depth_levels(
                da3,
                teacher_features["deep_levels"],
                batch_size=batch_size,
                steps=needed,
                views=views,
            ).detach()
    history, history_valid = prepare_observed_action_history(
        model=model, normalizer=normalizer, batch=batch, context_len=context_len
    )
    pose_history = prepare_observed_pose_history(
        model=model, pose_normalizer=pose_normalizer, batch=batch,
        context_len=context_len, conditioning_generator=conditioning_generator,
    )
    reference_pose = None
    if pose_history is not None:
        raw_start_pose = pose_history.new_tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        reference_pose = pose_normalizer.normalize(raw_start_pose).view(1, 1, 5).expand(
            batch_size, 1, 5
        )

    # Sliding mode (rollout=1) exposes one current transition. The legacy
    # rollout=3 compatibility path exposes the three original open-loop actions.
    action_start = 0 if dense_context else int(context_len) - 1
    action_steps = int(context_len) if dense_context else int(rollout_steps)
    stop_pose = None
    if getattr(model_ref, "stop_head_enabled", False) and getattr(
        model_ref, "stop_head_mode", ""
    ) == "action_hidden_pose":
        if pose_normalizer is None or "episode_pose" not in batch:
            raise ValueError("Action-hidden Stop requires episode pose and training pose statistics.")
        stop_pose = pose_normalizer.normalize(
            batch["episode_pose"][:, action_start : action_start + action_steps].float()
        )
    target_raw_full = batch["actions"][:, action_start : action_start + action_steps].float()
    action_mask_full = batch["action_loss_mask"][:, action_start : action_start + action_steps]
    chunk_size = int(getattr(model_ref, "action_chunk_size", 1))
    if chunk_size == 1:
        target_raw_full = target_raw_full[:, :, 0]
        action_mask_full = action_mask_full[:, :, 0]
    target_norm_full = normalizer.normalize(
        target_raw_full, stats_keys=list(batch["action_stats_key"])
    )
    with torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(amp and images.is_cuda)
    ):
        output = model(
            observed,
            reference_shallow=reference_shallow,
            observed_action_history=history,
            observed_action_history_valid_mask=history_valid,
            observed_pose_history=pose_history,
            reference_pose=reference_pose,
            stop_pose=stop_pose,
            lang_feats=language["last_hidden_state"],
            lang_padding_mask=language["attention_mask"],
            conditioning_generator=conditioning_generator,
        )
        geometry_architecture = getattr(model_ref, "geometry_architecture", "legacy")
        if geometry_architecture != "legacy" and deep_feature_enabled:
            raise ValueError("New geometry architectures use shallow feature targets; deep distillation is unsupported")
        feature_kwargs = dict(
            patch_start=1 + int(getattr(da3, "num_register_tokens", 0)),
            horizon_weights=feature_horizon_weights,
            patch_weight=feature_patch_weight,
            cls_weight=feature_cls_weight,
            register_weight=feature_register_weight,
        )
        feature_values = architecture_feature_loss(
            geometry_architecture, output, observed, target_future, **feature_kwargs
        )
        zero = output["actions_norm"].new_zeros(())
        direct_prediction = output.get("direct_actions_norm")
        refine_prediction = output.get("refine_actions_norm")
        action_direct = (
            masked_action_l1(direct_prediction, target_norm_full, action_mask_full)
            if isinstance(direct_prediction, torch.Tensor) else zero
        )
        action_refine = (
            masked_action_l1(refine_prediction, target_norm_full, action_mask_full)
            if isinstance(refine_prediction, torch.Tensor) else zero
        )
        if not isinstance(direct_prediction, torch.Tensor) and not isinstance(
            refine_prediction, torch.Tensor
        ):
            action_refine = masked_action_l1(
                output["actions_norm"], target_norm_full, action_mask_full
            )
        action = (
            float(action_direct_weight) * action_direct
            + float(action_refine_weight) * action_refine
        )
        pose_supervision_available = (
            pose_normalizer is not None and "episode_pose" in batch
        )
        if output.get("relative_pose") is not None and not pose_supervision_available:
            raise ValueError(
                "Relative-pose head requires episode_pose GT and a pose normalizer."
            )
        if pose_supervision_available:
            pose_target_start = (
                feature_target_offset if dense_context else int(context_len)
            )
            future_pose_target_raw = batch["episode_pose"][
                :, pose_target_start : pose_target_start + action_steps
            ].float()
            future_pose_target = pose_normalizer.normalize(future_pose_target_raw)
            pose_valid = action_mask_full.bool().all(dim=(-1, -2))
        else:
            future_pose_target_raw = target_raw_full.new_zeros(
                target_raw_full.shape[0], action_steps, 5
            )
            future_pose_target = target_raw_full.new_zeros(
                target_raw_full.shape[0], action_steps, 5
            )
            pose_valid = torch.zeros(
                target_raw_full.shape[0], action_steps,
                device=target_raw_full.device, dtype=torch.bool,
            )
        pose_values = relative_pose_loss(
            output.get("relative_pose"), future_pose_target, pose_valid
        )
        stop_logits = output.get("stop_logits")
        stop_target = batch.get("stop_target")
        if isinstance(stop_logits, torch.Tensor):
            if stop_target is None:
                raise KeyError("Stop Head is enabled but batch has no stop_target.")
            stop_target = stop_target[:, action_start : action_start + action_steps].float()
            if stop_logits.shape != stop_target.shape:
                raise ValueError(
                    f"stop_logits={tuple(stop_logits.shape)} but target={tuple(stop_target.shape)}"
                )
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                stop_logits.float(), stop_target,
                pos_weight=stop_logits.new_tensor(float(stop_pos_weight)),
            )
            stop_probability = stop_logits.float().sigmoid()
            stop_prediction = stop_probability >= 0.5
            stop_positive = stop_target >= 0.5
            stop_accuracy = (stop_prediction == stop_positive).float().mean()
            stop_recall = (
                (stop_prediction & stop_positive).float().sum()
                / stop_positive.float().sum().clamp_min(1.0)
            )
            stop_rate = stop_positive.float().mean()
            stop_positive_count = stop_positive.float().sum()
            stop_negative_count = (~stop_positive).float().sum()
            stop_positive_probability_sum = stop_probability[stop_positive].sum()
            stop_negative_probability_sum = stop_probability[~stop_positive].sum()
            stop_positive_probability = (
                stop_positive_probability_sum / stop_positive_count.clamp_min(1.0)
            )
            stop_negative_probability = (
                stop_negative_probability_sum / stop_negative_count.clamp_min(1.0)
            )
        else:
            stop_loss = zero
            # Preserve a well-defined empty/negative diagnostic tensor when
            # Stop Head is disabled; this keeps the generic evaluator usable.
            stop_probability = output["actions_norm"].new_zeros(
                output["actions_norm"].shape[:2], dtype=torch.float32
            )
            stop_target = torch.zeros_like(stop_probability)
            stop_accuracy = zero
            stop_recall = zero
            stop_rate = zero
            stop_positive_count = zero
            stop_negative_count = zero
            stop_positive_probability_sum = zero
            stop_negative_probability_sum = zero
            stop_positive_probability = zero
            stop_negative_probability = zero
        if bool(deep_feature_enabled):
            if int(rollout_steps) != 1:
                raise ValueError("Deep feature distillation currently requires rollout_steps=1.")
            if feature_target_offset != 1:
                raise ValueError(
                    "Deep sliding-IDM feature distillation assumes an adjacent F(t+1) target; "
                    "disable it for long-horizon F(t+K) prediction."
                )
            model_ref = model.module if hasattr(model, "module") else model
            target_idm_window = model_ref.build_sliding_idm_window(
                observed, target_future[:, :1]
            )
            with torch.no_grad():
                target_idm_features = da3.propagate_shallow_visual_slots_grad(
                    target_idm_window, gradient_checkpointing=False
                )
            deep_values = deep_feature_loss(
                output["idm_features"], target_idm_features,
                patch_start=1 + int(getattr(da3, "num_register_tokens", 0)),
                patch_weight=deep_feature_patch_weight,
                cls_weight=deep_feature_cls_weight,
            )
        else:
            deep_values = {
                "total": zero, "patch": zero, "cls": zero,
                "cosine": zero, "per_level": zero.new_zeros(0),
            }
        if depth_enabled:
            target_mode = str(depth_target_mode).lower()
            if geometry_architecture in {"current_prediction", "direct_current"} and target_mode != "current":
                raise ValueError("Current architectures require loss.depth_target_mode=current")
            if geometry_architecture.startswith("dual_") and target_mode != "both":
                raise ValueError("Dual architectures require loss.depth_target_mode=both")
            if target_mode not in {"future", "current", "both"}:
                raise ValueError(
                    f"Unsupported depth_target_mode={depth_target_mode!r}; "
                    "expected future, current, or both."
                )
            deep_joint = output.get("deep_joint_features")
            if not isinstance(deep_joint, dict) or "depth" not in deep_joint:
                raise RuntimeError("Frozen DPT did not decode predicted future depth.")
            future_pred_depth = deep_joint["depth"]
            if future_pred_depth.ndim == 4 and future_pred_depth.shape[1] == 1:
                future_pred_depth = future_pred_depth[:, 0]
            future_pred_depth = future_pred_depth.reshape(
                batch_size, action_steps, views, *future_pred_depth.shape[-2:]
            )

            if str(depth_target_source) == "da3_teacher_window":
                if target_mode != "future":
                    raise ValueError(
                        "current/both depth targets currently require ue_gt targets."
                    )
                if teacher_window_depth is None:
                    raise RuntimeError("DA3 teacher-window depth target was not generated.")
                depth_target_start = (
                    feature_target_offset if dense_context else int(context_len)
                )
                target_depth = teacher_window_depth[
                    :, depth_target_start : depth_target_start + action_steps
                ]
                if str(depth_scale_mode) == "gam_window_pointnorm":
                    teacher_mask = (
                        torch.isfinite(teacher_window_depth)
                        & (teacher_window_depth > 0)
                    )
                    target_mask = teacher_mask[
                        :, depth_target_start : depth_target_start + action_steps
                    ]
                    depth_values = gam_window_pointnorm_depth_loss(
                        future_pred_depth, target_depth, target_mask,
                        teacher_window_depth, teacher_mask, grad_weight=1.0,
                    )
                elif str(depth_scale_mode) == "da3_native":
                    depth_values = da3_native_pseudo_depth_loss(
                        future_pred_depth, target_depth, grad_weight=1.0
                    )
                else:
                    raise ValueError(
                        "DA3 teacher-window pseudo-depth supports "
                        "gam_window_pointnorm or da3_native."
                    )
            elif str(depth_target_source) != "ue_gt":
                raise ValueError(f"Unsupported depth_target_source={depth_target_source!r}")
            else:
                depth_gt = batch.get("gt_depth_meters")
                depth_mask = batch.get("gt_depth_mask")
                if depth_gt is None or depth_mask is None:
                    raise KeyError(
                        "depth_target_source=ue_gt requires gt_depth_meters and "
                        "gt_depth_mask from the UAV-Flow-Sim depth sidecar."
                    )
                semantic_mask = batch.get("gt_depth_semantic_mask")

                def _ue_loss(
                    prediction: torch.Tensor,
                    target_start: int,
                ) -> dict[str, torch.Tensor]:
                    target = depth_gt[:, target_start : target_start + action_steps]
                    target_mask = depth_mask[:, target_start : target_start + action_steps]
                    target_semantic = (
                        None if semantic_mask is None else
                        semantic_mask[:, target_start : target_start + action_steps]
                    )
                    scale_mode = str(depth_scale_mode)
                    if scale_mode in {
                        "gam_window_pointnorm", "fixed_metric", "scale_separated"
                    }:
                        predicted_scale = (
                            output.get("depth_log_scale")
                            if scale_mode == "scale_separated" else None
                        )
                        if scale_mode == "scale_separated" and predicted_scale is None:
                            raise RuntimeError(
                                "scale_separated depth requires depth_scale_head_enabled."
                            )
                        return gam_window_pointnorm_depth_loss(
                            prediction,
                            target,
                            target_mask,
                            depth_gt[:, :needed],
                            depth_mask[:, :needed],
                            semantic_mask=target_semantic,
                            linear_weight=depth_linear_weight,
                            log_weight=depth_log_weight,
                            log_epsilon_normalized=depth_log_epsilon_normalized,
                            grad_weight=depth_grad_weight,
                            semantic_weight=depth_semantic_weight,
                            semantic_separate_weight=depth_semantic_separate_weight,
                            semantic_linear_weight=depth_semantic_linear_weight,
                            semantic_log_weight=depth_semantic_log_weight,
                            semantic_grad_weight=depth_semantic_grad_weight,
                            gradient_mode=depth_gradient_mode,
                            gradient_tolerance_pixels=depth_gradient_tolerance_pixels,
                            fixed_scale_meters=(
                                depth_fixed_scale_meters
                                if scale_mode == "fixed_metric" else None
                            ),
                            predicted_log_scale=predicted_scale,
                            scale_loss_weight=depth_scale_loss_weight,
                        )
                    if scale_mode == "per_frame_median":
                        return scale_invariant_depth_loss(
                            prediction, target, target_mask, grad_weight=1.0
                        )
                    raise ValueError(f"Unsupported depth_scale_mode={depth_scale_mode!r}")

                losses = []
                if target_mode in {"future", "both"}:
                    future_start = (
                        feature_target_offset if dense_context else int(context_len)
                    )
                    losses.append(_ue_loss(future_pred_depth, future_start))
                if target_mode in {"current", "both"}:
                    # CA1_HB already needs this pass for action. Reuse its graph
                    # for current depth rather than executing DA3 a third time.
                    current_output = output.get("current_depth_output")
                    if current_output is not None:
                        current_pred_depth = current_output["depth"].reshape(
                            batch_size, action_steps, views, *current_output["depth"].shape[-2:]
                        )
                    else:
                        current_features = output.get("current_geometry_features")
                        if current_features is None:
                            current_features = da3.propagate_shallow_visual_slots_grad(
                                observed,
                                gradient_checkpointing=bool(
                                    getattr(model_ref, "deep_gradient_checkpointing", False)
                                ),
                            )
                        current_pred_depth = _decode_visual_depth_levels(
                            da3, current_features["deep_levels"],
                            batch_size=batch_size, steps=action_steps, views=views,
                        )
                    losses.append(_ue_loss(current_pred_depth, 0))
                depth_values = {
                    key: sum(item[key] for item in losses) / float(len(losses))
                    for key in losses[0]
                }
        else:
            depth_values = {
                "total": zero, "global_total": zero,
                "l1": zero, "grad": zero, "valid_ratio": zero,
            }
    prediction_raw = normalizer.denormalize(
        output["actions_norm"].float(), stats_keys=list(batch["action_stats_key"])
    )
    prediction_raw_chunk = (
        prediction_raw.unsqueeze(-2) if prediction_raw.ndim == 3 else prediction_raw
    )
    if pose_supervision_available:
        current_pose_raw = batch["episode_pose"][
            :, action_start : action_start + action_steps
        ].float()
        action_rollout_pose_raw = rollout_yaw4d_actions_from_episode_pose(
            current_pose_raw, prediction_raw_chunk
        )
        action_rollout_pose = pose_normalizer.normalize(action_rollout_pose_raw)
        action_pose_values = relative_pose_loss(
            action_rollout_pose, future_pose_target, pose_valid
        )
        if output.get("relative_pose") is not None:
            pose_consistency_values = relative_pose_loss(
                output["relative_pose"].float(), action_rollout_pose, pose_valid
            )
        else:
            pose_consistency_values = {
                "total": zero, "translation": zero,
                "translation_axes": zero.new_zeros(3),
                "yaw": zero, "unit": zero,
            }
        raw_axis_error = (
            action_rollout_pose_raw[..., :3] - future_pose_target_raw[..., :3]
        ).abs()
        raw_axis_weight = pose_valid.float().unsqueeze(-1)
        action_pose_values["translation_axes_meters"] = (
            raw_axis_error * raw_axis_weight
        ).sum(dim=tuple(range(raw_axis_error.ndim - 1))) / raw_axis_weight.sum().clamp_min(1.0)
    else:
        action_pose_values = {
            "total": zero, "translation": zero,
            "translation_axes": zero.new_zeros(3),
            "translation_axes_meters": zero.new_zeros(3),
            "yaw": zero, "unit": zero,
        }
        pose_consistency_values = action_pose_values
    action_per_sample_step = masked_action_l1_per_sample_step(
        output["actions_norm"], target_norm_full, action_mask_full
    )
    raw_per_sample_step = masked_action_l1_per_sample_step(
        prediction_raw, target_raw_full, action_mask_full
    )
    action_per_chunk_slot = masked_action_l1_per_chunk_slot(
        output["actions_norm"], target_norm_full, action_mask_full
    )
    raw_per_chunk_slot = masked_action_l1_per_chunk_slot(
        prediction_raw, target_raw_full, action_mask_full
    )
    raw_chunk = target_raw_full if target_raw_full.ndim == 4 else target_raw_full.unsqueeze(-2)
    mask_chunk = action_mask_full if action_mask_full.ndim == 4 else action_mask_full.unsqueeze(-2)
    valid_slot = mask_chunk.bool().all(dim=-1)
    zero_slot = (raw_chunk.abs().amax(dim=-1) <= 1.0e-8) & valid_slot
    zero_slot_fraction = zero_slot.float().sum() / valid_slot.float().sum().clamp_min(1.0)
    return {
        "feature": feature_values["total"],
        "action_direct": action_direct,
        "action_refine": action_refine,
        "feature_patch": feature_values["patch"],
        "feature_cls": feature_values["cls"],
        "feature_register": feature_values["register"],
        "feature_cos": feature_values["cosine"],
        "feature_per_horizon": feature_values["per_horizon"],
        "feature_patch_per_horizon": feature_values["patch_per_horizon"],
        "feature_cls_per_horizon": feature_values["cls_per_horizon"],
        "feature_register_per_horizon": feature_values["register_per_horizon"],
        "feature_cos_per_horizon": feature_values["cosine_per_horizon"],
        "deep_feature": deep_values["total"],
        "deep_feature_patch": deep_values["patch"],
        "deep_feature_cls": deep_values["cls"],
        "deep_feature_cos": deep_values["cosine"],
        "deep_feature_per_level": deep_values["per_level"],
        "depth": depth_values["total"],
        "depth_global": depth_values.get("global_total", depth_values["total"]),
        "depth_l1": depth_values["l1"],
        "depth_linear_l1": depth_values.get("linear_l1", depth_values["l1"]),
        "depth_log_l1": depth_values.get("log_l1", zero),
        "depth_grad": depth_values["grad"],
        "depth_scale_l1": depth_values.get("scale_l1", zero),
        "depth_scene_scale_mean": depth_values.get("scene_scale_mean", zero),
        "depth_valid_ratio": depth_values["valid_ratio"],
        "depth_semantic_ratio": depth_values.get("semantic_ratio", zero),
        "depth_semantic_total": depth_values.get("semantic_total", zero),
        "depth_semantic_linear_l1": depth_values.get("semantic_linear_l1", zero),
        "depth_semantic_log_l1": depth_values.get("semantic_log_l1", zero),
        "depth_semantic_grad": depth_values.get("semantic_grad", zero),
        "relative_pose": pose_values["total"],
        "relative_pose_translation": pose_values["translation"],
        "relative_pose_translation_axes": pose_values["translation_axes"],
        "relative_pose_yaw": pose_values["yaw"],
        "relative_pose_unit": pose_values["unit"],
        "action_rollout_pose": action_pose_values["total"],
        "action_rollout_pose_translation": action_pose_values["translation"],
        "action_rollout_pose_translation_axes": action_pose_values["translation_axes"],
        "action_rollout_pose_translation_axes_meters": action_pose_values[
            "translation_axes_meters"
        ],
        "action_rollout_pose_yaw": action_pose_values["yaw"],
        "pose_consistency": pose_consistency_values["total"],
        "action": action,
        "direct_action": action_direct,
        "refine_action": action_refine,
        "stop": stop_loss,
        "stop_accuracy": stop_accuracy,
        "stop_recall": stop_recall,
        "stop_rate": stop_rate,
        "stop_positive_probability": stop_positive_probability,
        "stop_negative_probability": stop_negative_probability,
        "stop_positive_probability_sum": stop_positive_probability_sum,
        "stop_negative_probability_sum": stop_negative_probability_sum,
        "stop_positive_count": stop_positive_count,
        "stop_negative_count": stop_negative_count,
        # Keep the detached per-example scores available to evaluation so a
        # deployment threshold can be selected from validation data instead
        # of being hard-coded to 0.5.  These tensors are not used by training.
        "stop_probabilities": stop_probability.detach(),
        "stop_targets": stop_target.detach(),
        "stop_future_gate": output.get("stop_future_gate", zero),
        "raw_l1": masked_action_l1(prediction_raw, target_raw_full, action_mask_full),
        "action_per_sample_step": action_per_sample_step,
        "raw_per_sample_step": raw_per_sample_step,
        "action_per_chunk_slot": action_per_chunk_slot.detach(),
        "raw_per_chunk_slot": raw_per_chunk_slot.detach(),
        "target_zero_slot_fraction": zero_slot_fraction.detach(),
        "gate": output["residual_gate"],
    }


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    da3: DA3GiantEncoder,
    text: torch.nn.Module,
    normalizer: ActionNormalizer,
    pose_normalizer: EpisodePoseNormalizer | None,
    loader: DataLoader,
    device: torch.device,
    contexts: list[int],
    rollout: int,
    horizon_weights: list[float],
    feature_patch_weight: float,
    feature_cls_weight: float,
    feature_register_weight: float,
    deep_feature_enabled: bool,
    deep_feature_patch_weight: float,
    deep_feature_cls_weight: float,
    stop_pos_weight: float,
    max_batches: int,
    amp: bool,
    action_direct_weight: float = 1.0,
    action_refine_weight: float = 1.0,
    depth_target_source: str = "ue_gt",
    depth_scale_mode: str = "per_frame_median",
    depth_target_mode: str = "future",
    depth_fixed_scale_meters: float = 100.0,
    depth_scale_loss_weight: float = 1.0,
    depth_linear_weight: float = 0.5,
    depth_log_weight: float = 1.0,
    depth_log_epsilon_normalized: float = 1e-6,
    depth_grad_weight: float = 1.0,
    depth_semantic_weight: float = 3.0,
    depth_semantic_separate_weight: float = 0.0,
    depth_semantic_linear_weight: float | None = None,
    depth_semantic_log_weight: float | None = None,
    depth_semantic_grad_weight: float | None = None,
    depth_gradient_mode: str = "tolerant_log",
    depth_gradient_tolerance_pixels: int = 2,
) -> dict[str, float]:
    model.eval()
    sums = {h: torch.zeros(37, device=device) for h in contexts}
    # Per rollout step: total feature MSE, patch MSE, CLS MSE, cosine.
    model_ref = model.module if hasattr(model, "module") else model
    dense_context = bool(getattr(model_ref, "dense_context_supervision", False))
    horizon_sums = {
        h: torch.zeros(int(h if dense_context else rollout), 4, device=device)
        for h in contexts
    }
    # boundary: deployed final transition for this context; start_first: the
    # true episode-start F0 -> F1 transition, rather than an arbitrary H=1 crop.
    action_detail_sums = {h: torch.zeros(6, device=device) for h in contexts}
    # Normalized pose-head xyz, normalized action-rollout xyz, and physical
    # action-rollout xyz MAE in meters.
    pose_axis_sums = {h: torch.zeros(9, device=device) for h in contexts}
    chunk_size = int(getattr(model_ref, "action_chunk_size", 1))
    chunk_slot_sums = {
        h: torch.zeros(chunk_size, 2, device=device) for h in contexts
    }
    zero_slot_sums = {h: torch.zeros(2, device=device) for h in contexts}
    # Stop scores from the imbalanced classifier tend to occupy a narrow band,
    # so use a fine deployment-oriented sweep around the default 0.5.
    stop_thresholds = tuple(value / 100.0 for value in range(45, 91))
    # Columns are TP, FP, FN, TN for each threshold.
    stop_threshold_sums = {
        h: torch.zeros(len(stop_thresholds), 4, device=device) for h in contexts
    }
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        for context in contexts:
            values = forward_batch(
                model=model,
                da3=da3,
                text=text,
                normalizer=normalizer,
                pose_normalizer=pose_normalizer,
                batch=batch,
                context_len=context,
                rollout_steps=rollout,
                feature_horizon_weights=horizon_weights,
                feature_patch_weight=feature_patch_weight,
                feature_cls_weight=feature_cls_weight,
                feature_register_weight=feature_register_weight,
                deep_feature_enabled=deep_feature_enabled,
                deep_feature_patch_weight=deep_feature_patch_weight,
                deep_feature_cls_weight=deep_feature_cls_weight,
                stop_pos_weight=stop_pos_weight,
                amp=amp,
                action_direct_weight=action_direct_weight,
                action_refine_weight=action_refine_weight,
                depth_target_source=depth_target_source,
                depth_scale_mode=depth_scale_mode,
                depth_target_mode=depth_target_mode,
                depth_fixed_scale_meters=depth_fixed_scale_meters,
                depth_scale_loss_weight=depth_scale_loss_weight,
                depth_linear_weight=depth_linear_weight,
                depth_log_weight=depth_log_weight,
                depth_log_epsilon_normalized=depth_log_epsilon_normalized,
                depth_grad_weight=depth_grad_weight,
                depth_semantic_weight=depth_semantic_weight,
                depth_semantic_separate_weight=depth_semantic_separate_weight,
                depth_semantic_linear_weight=depth_semantic_linear_weight,
                depth_semantic_log_weight=depth_semantic_log_weight,
                depth_semantic_grad_weight=depth_semantic_grad_weight,
                depth_gradient_mode=depth_gradient_mode,
                depth_gradient_tolerance_pixels=depth_gradient_tolerance_pixels,
            )
            count = float(batch["all_view_images"].shape[0])
            chunk_slot_sums[context][:, 0] += values["action_per_chunk_slot"] * count
            chunk_slot_sums[context][:, 1] += values["raw_per_chunk_slot"] * count
            zero_slot_sums[context][0] += values["target_zero_slot_fraction"] * count
            zero_slot_sums[context][1] += count
            sums[context][:15] += torch.stack([
                values["action"],
                values["feature"],
                values["feature_patch"],
                values["feature_cls"],
                values["feature_register"],
                values["feature_cos"],
                values["raw_l1"],
                values["deep_feature"],
                values["deep_feature_patch"],
                values["deep_feature_cls"],
                values["deep_feature_cos"],
                values["stop"],
                values["stop_accuracy"],
                values["stop_recall"],
                values["stop_rate"],
            ]) * count
            sums[context][15] += values["stop_positive_probability_sum"]
            sums[context][16] += values["stop_positive_count"]
            sums[context][17] += values["stop_negative_probability_sum"]
            sums[context][18] += values["stop_negative_count"]
            sums[context][19:24] += torch.stack([
                values["depth"], values["depth_linear_l1"],
                values["depth_log_l1"], values["depth_grad"],
                values["depth_semantic_ratio"],
            ]) * count
            sums[context][24:30] += torch.stack([
                values["depth_semantic_total"], values["depth_semantic_grad"],
                values["relative_pose"], values["relative_pose_translation"],
                values["relative_pose_yaw"], values["relative_pose_unit"],
            ]) * count
            sums[context][30:34] += torch.stack([
                values["action_rollout_pose"],
                values["action_rollout_pose_translation"],
                values["action_rollout_pose_yaw"],
                values["pose_consistency"],
            ]) * count
            sums[context][34:36] += torch.stack([
                values["depth_scale_l1"],
                values["depth_scene_scale_mean"],
            ]) * count
            sums[context][36] += count
            pose_axis_sums[context] += torch.cat([
                values["relative_pose_translation_axes"],
                values["action_rollout_pose_translation_axes"],
                values["action_rollout_pose_translation_axes_meters"],
            ]) * count
            horizon_sums[context] += torch.stack(
                [
                    values["feature_per_horizon"],
                    values["feature_patch_per_horizon"],
                    values["feature_cls_per_horizon"],
                    values["feature_cos_per_horizon"],
                ],
                dim=-1,
            ) * count
            action_per_step = values["action_per_sample_step"]
            raw_per_step = values["raw_per_sample_step"]
            detail = action_detail_sums[context]
            detail[0] += action_per_step[:, -1].sum()
            detail[1] += raw_per_step[:, -1].sum()
            detail[2] += float(action_per_step.shape[0])
            start_mask = batch["start_t"].reshape(-1).eq(0)
            if bool(start_mask.any()):
                detail[3] += action_per_step[start_mask, 0].sum()
                detail[4] += raw_per_step[start_mask, 0].sum()
                detail[5] += start_mask.sum()
            stop_probability = values["stop_probabilities"].reshape(-1)
            stop_positive = values["stop_targets"].reshape(-1).ge(0.5)
            for threshold_index, threshold in enumerate(stop_thresholds):
                stop_prediction = stop_probability.ge(threshold)
                threshold_counts = stop_threshold_sums[context][threshold_index]
                threshold_counts[0] += (stop_prediction & stop_positive).sum()
                threshold_counts[1] += (stop_prediction & ~stop_positive).sum()
                threshold_counts[2] += (~stop_prediction & stop_positive).sum()
                threshold_counts[3] += (~stop_prediction & ~stop_positive).sum()
    if dist.is_initialized():
        for value in sums.values():
            dist.all_reduce(value)
        for value in horizon_sums.values():
            dist.all_reduce(value)
        for value in action_detail_sums.values():
            dist.all_reduce(value)
        for value in pose_axis_sums.values():
            dist.all_reduce(value)
        for value in chunk_slot_sums.values():
            dist.all_reduce(value)
        for value in zero_slot_sums.values():
            dist.all_reduce(value)
        for value in stop_threshold_sums.values():
            dist.all_reduce(value)
    result: dict[str, float] = {}
    for context, value in sums.items():
        count = value[36].clamp_min(1.0)
        result |= {
            f"H{context}_action": float((value[0] / count).item()),
            f"H{context}_direct_action": float((value[0] / count).item()),
            f"H{context}_feature": float((value[1] / count).item()),
            f"H{context}_feat_patch": float((value[2] / count).item()),
            f"H{context}_feat_cls": float((value[3] / count).item()),
            f"H{context}_feat_reg": float((value[4] / count).item()),
            f"H{context}_cos": float((value[5] / count).item()),
            f"H{context}_raw": float((value[6] / count).item()),
            f"H{context}_direct_raw": float((value[6] / count).item()),
            f"H{context}_deep": float((value[7] / count).item()),
            f"H{context}_deep_patch": float((value[8] / count).item()),
            f"H{context}_deep_cls": float((value[9] / count).item()),
            f"H{context}_deep_cos": float((value[10] / count).item()),
            f"H{context}_stop": float((value[11] / count).item()),
            f"H{context}_stop_acc": float((value[12] / count).item()),
            f"H{context}_stop_recall": float((value[13] / count).item()),
            f"H{context}_stop_rate": float((value[14] / count).item()),
            f"H{context}_stop_pos_prob": float(
                (value[15] / value[16].clamp_min(1.0)).item()
            ),
            f"H{context}_stop_neg_prob": float(
                (value[17] / value[18].clamp_min(1.0)).item()
            ),
            f"H{context}_depth": float((value[19] / count).item()),
            f"H{context}_depth_linear": float((value[20] / count).item()),
            f"H{context}_depth_l1": float((value[20] / count).item()),
            f"H{context}_depth_log": float((value[21] / count).item()),
            f"H{context}_depth_grad": float((value[22] / count).item()),
            f"H{context}_depth_semantic_ratio": float((value[23] / count).item()),
            f"H{context}_depth_semantic": float((value[24] / count).item()),
            f"H{context}_depth_semantic_grad": float((value[25] / count).item()),
            f"H{context}_pose5": float((value[26] / count).item()),
            f"H{context}_pose5_xyz": float((value[27] / count).item()),
            f"H{context}_pose5_yaw": float((value[28] / count).item()),
            f"H{context}_pose5_unit": float((value[29] / count).item()),
            f"H{context}_action_pose5": float((value[30] / count).item()),
            f"H{context}_action_pose5_xyz": float((value[31] / count).item()),
            f"H{context}_action_pose5_yaw": float((value[32] / count).item()),
            f"H{context}_pose_consistency": float((value[33] / count).item()),
            f"H{context}_depth_scale_l1": float((value[34] / count).item()),
            f"H{context}_depth_scene_scale_m": float((value[35] / count).item()),
        }
        pose_axes = pose_axis_sums[context] / count
        result |= {
            f"H{context}_pose5_x": float(pose_axes[0].item()),
            f"H{context}_pose5_y": float(pose_axes[1].item()),
            f"H{context}_pose5_z": float(pose_axes[2].item()),
            f"H{context}_action_pose5_x": float(pose_axes[3].item()),
            f"H{context}_action_pose5_y": float(pose_axes[4].item()),
            f"H{context}_action_pose5_z": float(pose_axes[5].item()),
            f"H{context}_action_pose5_x_m": float(pose_axes[6].item()),
            f"H{context}_action_pose5_y_m": float(pose_axes[7].item()),
            f"H{context}_action_pose5_z_m": float(pose_axes[8].item()),
        }
        detail = action_detail_sums[context]
        boundary_count = detail[2].clamp_min(1.0)
        start_count = detail[5].clamp_min(1.0)
        result |= {
            f"H{context}_boundary_action": float((detail[0] / boundary_count).item()),
            f"H{context}_boundary_raw": float((detail[1] / boundary_count).item()),
            f"H{context}_start_first_action": float((detail[3] / start_count).item()),
            f"H{context}_start_first_raw": float((detail[4] / start_count).item()),
            f"H{context}_start_samples": float(detail[5].item()),
            f"H{context}_target_zero_slot_fraction": float(
                (zero_slot_sums[context][0] / zero_slot_sums[context][1].clamp_min(1.0)).item()
            ),
        }
        per_slot = chunk_slot_sums[context] / count
        for slot_index in range(chunk_size):
            result |= {
                f"H{context}_chunk_slot{slot_index + 1}_action": float(
                    per_slot[slot_index, 0].item()
                ),
                f"H{context}_chunk_slot{slot_index + 1}_raw": float(
                    per_slot[slot_index, 1].item()
                ),
            }
        for threshold_index, threshold in enumerate(stop_thresholds):
            tp, fp, fn, tn = stop_threshold_sums[context][threshold_index]
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
            accuracy = (tp + tn) / (tp + fp + fn + tn).clamp_min(1.0)
            false_positive_rate = fp / (fp + tn).clamp_min(1.0)
            label = f"{threshold:.2f}".replace(".", "p")
            result |= {
                f"H{context}_stop_thr{label}_precision": float(precision.item()),
                f"H{context}_stop_thr{label}_recall": float(recall.item()),
                f"H{context}_stop_thr{label}_f1": float(f1.item()),
                f"H{context}_stop_thr{label}_acc": float(accuracy.item()),
                f"H{context}_stop_thr{label}_fpr": float(false_positive_rate.item()),
            }
        per_step = horizon_sums[context] / count
        prediction_steps = int(context if dense_context else rollout)
        for rollout_index in range(prediction_steps):
            step_number = rollout_index + 1
            result |= {
                f"H{context}_step{step_number}_feature": float(per_step[rollout_index, 0].item()),
                f"H{context}_step{step_number}_patch": float(per_step[rollout_index, 1].item()),
                f"H{context}_step{step_number}_cls": float(per_step[rollout_index, 2].item()),
                f"H{context}_step{step_number}_cos": float(per_step[rollout_index, 3].item()),
            }
    model.train()
    return result
