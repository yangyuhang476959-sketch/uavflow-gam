"""Strict shared initialization for the Pure/Warp ordered-action/Pose Stop MLP."""
import torch
from .common import tensor_hash

FORMAT='shared_action_hidden_pose_stop_v1'
POSE_SCALE=[1000.,1000.,1000.,180.,1.]


def load_shared_stop(module,path,pose_scale):
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if bundle.get('format')!=FORMAT or list(pose_scale)!=POSE_SCALE or bundle['pose_scale']!=POSE_SCALE:
        raise ValueError('Shared Stop format or Pose scaling differs')
    dims=(module.action_dim,module.action_slots,module.pose_dim,module.action_proj[1].out_features)
    if dims!=(1536,5,5,512):raise ValueError('Shared Stop requires ordered5x1536 hidden, Pose5 and hidden512')
    if tensor_hash(bundle['state'])!=bundle['state_hash']:raise ValueError('Shared Stop initialization checksum differs')
    module.load_state_dict(bundle['state'],strict=True)
    if tensor_hash(module.state_dict())!=bundle['state_hash']:raise AssertionError('Shared Stop tensors differ after load')
    return dict(state_hash=bundle['state_hash'],dimensions=list(dims),pose_scale=POSE_SCALE)
