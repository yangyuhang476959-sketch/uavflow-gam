"""Frozen DINO patch-token teacher for UAV multimodal GAM."""

import torch
import torch.nn as nn


class FrozenDinoTeacher(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_patches=196,
    ):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name).eval().requires_grad_(False)
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.num_patches = int(num_patches)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images):
        """Map RGB [B,T,V,3,H,W] in [0,1] to patch tokens [B,T,V,P,D]."""
        b, t, v, c, h, w = images.shape
        x = images.reshape(b * t * v, c, h, w).float()
        x = (x - self.mean.float()) / self.std.float()
        # DINOv3 ViT output is [CLS, 4 registers, patches]. Taking the final P
        # positions deliberately excludes CLS/register tokens from the target.
        tokens = self.encoder(pixel_values=x).last_hidden_state[:, -self.num_patches:]
        if tokens.shape[1] != self.num_patches:
            raise RuntimeError(f"DINO produced {tokens.shape[1]} patches; expected {self.num_patches}.")
        return tokens.reshape(b, t, v, self.num_patches, self.hidden_size)
