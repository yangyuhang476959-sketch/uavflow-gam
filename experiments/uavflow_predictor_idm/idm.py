"""Reconstruct and freeze the exact Stage-1 inverse-dynamics head."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from robot.modeling.uav_da3_shallow_inverse import (
    DA3HierarchicalWindowInversePerceiver,
    DA3MultiLevelWindowInversePerceiver,
)


def build_frozen_idm(
    checkpoint: str,
    *,
    da3: nn.Module,
    action_dim: int,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the architecture metadata and weights stored by Stage 1."""
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    meta = payload["meta"]
    args = meta["args"]
    common = dict(
        shallow_dim=int(da3.embed_dim),
        deep_dim=int(da3.hidden_size),
        num_register_tokens=int(getattr(da3, "num_register_tokens", 0)),
        num_deep_levels=len(getattr(da3, "out_layers", [19, 27, 33, 39])),
        action_dim=int(action_dim),
        window_size=int(args["window_size"]),
        model_dim=int(args["patch_model_dim"]),
        depth=int(args["patch_depth"]),
        num_heads=int(args["patch_heads"]),
        mlp_ratio=float(args["patch_mlp_ratio"]),
        dropout=float(args["dropout"]),
    )
    head_type = str(meta["head_type"])
    if head_type == "hierarchical":
        idm = DA3HierarchicalWindowInversePerceiver(
            **common,
            num_queries_per_level=int(args["hierarchical_queries_per_level"]),
            use_pos_embed=bool(args.get("use_pos_embed", False)),
        )
    elif head_type == "worldvln_multilevel":
        idm = DA3MultiLevelWindowInversePerceiver(
            **common,
            num_latents=int(args["patch_latents"]),
        )
    else:
        raise ValueError(
            "Stage-2 supports the multi-frame hierarchical/WorldVLN IDMs; "
            f"checkpoint contains {head_type!r}."
        )
    idm.load_state_dict(payload["head"], strict=True)
    idm.eval()
    for parameter in idm.parameters():
        parameter.requires_grad = False
    setattr(idm, "head_type", head_type)
    return idm, meta
