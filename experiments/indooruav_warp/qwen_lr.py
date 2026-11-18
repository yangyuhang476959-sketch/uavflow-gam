"""Opt-in Qwen-only learning-rate group, preserving other predictor rates."""
import hashlib
import json
from pathlib import Path
import torch


def separate_qwen_lora(parameters,conditioner,base_lr,qwen_lr):
    selected={id(p) for n,p in conditioner.encoder.qwen.named_parameters()
              if p.requires_grad and n.split('.')[-1] in {'lora_A','lora_B'}}
    if not selected:raise ValueError('Qwen LR search requires trainable LoRA parameters')
    available={id(p) for p in parameters}
    if not selected.issubset(available):raise ValueError('Qwen LoRA outside predictor group')
    if float(base_lr)<=0 or float(qwen_lr)<=0:raise ValueError('Learning rates must be positive')
    remaining=[p for p in parameters if id(p) not in selected]
    group=dict(params=[p for p in parameters if id(p) in selected],name='qwen_lora',
               lr=float(qwen_lr),lr_multiplier=float(qwen_lr)/float(base_lr))
    return remaining,group


def audit_initial_parameters(model,destination,reference=None):
    digest=hashlib.sha256();count=0
    for name,p in sorted(model.named_parameters()):
        if not p.requires_grad:continue
        value=p.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        count+=p.numel()
    report=dict(sha256=digest.hexdigest(),trainable_elements=count)
    if reference and json.loads(Path(reference).read_text())!=report:
        raise ValueError('Search trial initial parameters differ from reference')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2))
    return report


def audit_gam_shared_initialization(model,reference,destination):
    """Verify actual GAM step0 tensors against the audited Current final path."""
    from experiments.uavflow_matched_repeat.common import tensor_hash
    bundle=torch.load(reference,map_location='cpu',weights_only=False)
    text=model.trainable_conditioner
    states=dict(pose=text.state_encoder.state_dict(),projection=model.rgbwarp_correction.proposal_projection.state_dict(),head=model.parallel_action_correction_head.decoder.state_dict())
    actual={k:tensor_hash(v) for k,v in states.items()}
    assert actual=={k:bundle['hashes'][k] for k in states},'GAM shared initial tensors mismatch'
    pred=model.rgbwarp_correction.future_predictor
    assert pred.action_slot_embed.shape==(1,1,5,pred.d_model) and pred.use_proprio_input
    report=dict(shared_hashes=actual,reference=str(reference),action_slot_embedding_shape=list(pred.action_slot_embed.shape),pose_fusion='separate SA proprio token',slot_embedding='learned position per action query',future_depth_units='metres')
    Path(destination).write_text(json.dumps(report,indent=2));return report
