"""Batch movement, temporal augmentation and context sampling for Stage 2."""

from __future__ import annotations

from typing import Any

import torch

from .runtime import rand_on_device


class EpisodePoseNormalizer:
    """GAM-style q01/q99 normalization for episode-relative UAV pose.

    Only xyz is normalized. The final sin/cos yaw channels pass through
    unchanged. Statistics are computed over unique train-split physical frames,
    never over repeated training windows or absorbing endpoint augmentation.
    """

    def __init__(self, stats: dict[str, Any]):
        self.q01 = torch.as_tensor(stats["q01"], dtype=torch.float32)
        self.q99 = torch.as_tensor(stats["q99"], dtype=torch.float32)
        self.mask = torch.as_tensor(stats.get("mask", [True, True, True, False, False]), dtype=torch.bool)
        if self.q01.shape != (5,) or self.q99.shape != (5,) or self.mask.shape != (5,):
            raise ValueError("Episode pose statistics must contain five-dimensional q01/q99/mask.")

    def normalize(self, pose: torch.Tensor) -> torch.Tensor:
        q01 = self.q01.to(device=pose.device, dtype=pose.dtype)
        scale = (self.q99 - self.q01).to(device=pose.device, dtype=pose.dtype).clamp_min(1e-8)
        mask = self.mask.to(device=pose.device)
        normalized = 2.0 * (pose - q01) / scale - 1.0
        return torch.where(mask, normalized, pose)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"q01": self.q01.cpu(), "q99": self.q99.cpu(), "mask": self.mask.cpu()}


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def temporal_color_augment(
    images: torch.Tensor,
    cfg: dict[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one mild photometric transform shared by all frames/views."""
    if not bool(cfg.get("enabled", False)) or not torch.is_grad_enabled():
        return images
    batch = images.shape[0]
    shape = (batch, 1, 1, 1, 1, 1)
    apply = rand_on_device(shape, device=images.device, generator=generator) < float(
        cfg.get("probability", 0.5)
    )
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    color = float(cfg.get("color", 0.0))
    gain = 1.0 + (rand_on_device(shape, device=images.device, generator=generator) * 2 - 1) * contrast
    bias = (rand_on_device(shape, device=images.device, generator=generator) * 2 - 1) * brightness
    channel = 1.0 + (
        rand_on_device(
            (batch, 1, 1, 3, 1, 1), device=images.device, generator=generator
        ) * 2 - 1
    ) * color
    augmented = (images * gain * channel + bias).clamp(0.0, 1.0)
    return torch.where(apply, augmented, images)


def choose_context_length(
    model_cfg: dict[str, Any],
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Sample H from the configured categorical distribution."""
    choices = torch.tensor(model_cfg["context_lengths"], device=device, dtype=torch.long)
    weights = torch.tensor(model_cfg["context_weights"], device=device, dtype=torch.float)
    if choices.numel() != weights.numel() or float(weights.sum()) <= 0:
        raise ValueError("context_lengths/context_weights must have equal positive length.")
    generator_device = torch.device(generator.device) if generator is not None else device
    sample_weights = weights.to(generator_device)
    index = torch.multinomial(sample_weights, 1, generator=generator)
    return int(choices[int(index.item())].item())
