"""Check shared action modules even in the control that has no Qwen encoder."""
import torch
from experiments.uavflow_matched_repeat.common import tensor_hash


def controller_head_parameter_ids(controller):
    ids={id(p) for p in controller.parameters() if p.requires_grad}
    predictor=getattr(controller,'future_predictor',None)
    if predictor is not None:ids.difference_update(id(p) for p in predictor.parameters())
    return ids


def verify_uav_initialization(model,text,path):
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    modules=dict(pose=text.state_encoder.state_dict(),
                 projection=model.rgbwarp_correction.proposal_projection.state_dict(),
                 head=model.parallel_action_head.decoder.state_dict())
    hashes={name:tensor_hash(state) for name,state in modules.items()}
    expected={name:bundle['hashes'][name] for name in modules}
    if hashes!=expected:raise ValueError('UAV-GAM public initialization changed: '+str(hashes))
    return dict(shared=hashes,qwen='absent_by_design')
