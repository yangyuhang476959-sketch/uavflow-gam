"""Native GAM's fixed pre-global encoder boundary, shared by input and teacher.

Only pre-global features are required as targets. Both branches read the same
immutable pretrained prefix; no trainable/deep weights participate in targets.
This is equivalent to a separate frozen copy, without duplicating DA3 memory.
"""
import torch

def prefix_parameters(da3):
    boundary=int(da3.backbone.pretrained.alt_start)
    for name,p in da3.backbone.pretrained.named_parameters():
        if not name.startswith('blocks.') or int(name.split('.')[1])<boundary:
            yield name,p

def freeze_prefix(da3):
    names=[]
    for name,p in prefix_parameters(da3):
        if p.requires_grad:names.append(name)
        p.requires_grad_(False)
    for block in da3.backbone.pretrained.blocks[:int(da3.backbone.pretrained.alt_start)]:block.eval()
    return names

@torch.no_grad()
def fixed_features(da3,images):
    if any(p.requires_grad for _,p in prefix_parameters(da3)):
        raise RuntimeError('Native GAM feature target is not fixed: shallow DA3 parameter is trainable')
    if any(block.training for block in da3.backbone.pretrained.blocks[:int(da3.backbone.pretrained.alt_start)]):
        raise RuntimeError('Native GAM shallow encoder must remain in eval mode')
    return da3.encode_shallow_rgb_grad(images,gradient_checkpointing=False).detach()
