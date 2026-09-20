"""Small runtime helpers shared by the Stage-2 training entrypoint."""

from __future__ import annotations

import math
import os
from typing import Any

from omegaconf import OmegaConf


def distributed_info() -> tuple[bool, int, int, int]:
    """Return ``(is_distributed, rank, local_rank, world_size)``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world > 1, rank, local_rank, world


def apply_overrides(cfg: Any, overrides: list[str]) -> Any:
    """Apply repeated ``--set dotted.key=value`` command-line overrides."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def lr_scale(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step <= warmup:
        return float(step) / max(1, int(warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
