"""Observation-only letterbox/pose adapter for the existing checkpoint loader."""
import math
import numpy as np
import torch
from robot.data.indooruav_dataset import (letterbox_rgb,letterbox_layout,heading_basis,
                                         cumulative_targets_to_world)


class IndoorUAVWarpPolicy:
    def __init__(self,policy):
        self.policy=policy
        config=policy.model.rgbwarp_correction.config
        if config.get('action_representation')!='chunk_anchor_local':
            raise ValueError('This adapter requires an IndoorUAV cumulative checkpoint')

    @torch.inference_mode()
    def predict(self,rgb,instruction,*,first_pose,current_pose,is_start=False):
        """Poses are the raw IndoorUAV [x,y,height,yaw_degrees]; no depth input."""
        device=next(self.policy.model.parameters()).device
        image,content=letterbox_rgb(rgb)
        _,intrinsics,_=letterbox_layout(np.asarray(rgb).shape[:2])
        first,current=np.asarray(first_pose),np.asarray(current_pose)
        xyz=(current[:3]-first[:3])@heading_basis(first[3])*100
        yaw=(current[3]-first[3]+180)%360-180
        state=torch.tensor([[*xyz,yaw,float(is_start)]],dtype=torch.float32,device=device)
        output=self.policy.predict(image[None,None].to(device),[instruction],
                    current_prompt_pose=state,stats_keys=['indooruav_vla'],
                    camera_intrinsics=torch.from_numpy(intrinsics)[None].to(device),
                    image_valid_mask=content[None].to(device))
        actions=output['actions_meters_radians'][0].float().cpu().numpy()
        output['cumulative_local_actions']=actions
        targets=cumulative_targets_to_world(current,actions)
        output['world_target_poses_degrees']=targets
        simulator_targets=targets.copy()
        simulator_targets[:,3]=np.deg2rad(simulator_targets[:,3])
        # Official load_position consumes [x,y,height,yaw_radians].
        output['simulator_target_poses']=simulator_targets
        output['action_representation']='chunk_anchor_local'
        return output


class IndoorUAVPurePolicy:
    """Same IndoorUAV observed-pose/letterbox/target convention for Pure."""
    def __init__(self,model,normalizer):
        self.model=model.eval();self.normalizer=normalizer

    @torch.inference_mode()
    def predict(self,rgb,instruction,*,first_pose,current_pose,is_start=False):
        device=next(self.model.parameters()).device
        image,_=letterbox_rgb(rgb)
        first,current=np.asarray(first_pose),np.asarray(current_pose)
        xyz=(current[:3]-first[:3])@heading_basis(first[3])*100
        yaw=(current[3]-first[3]+180)%360-180
        state=torch.tensor([[*xyz,yaw,float(is_start)]],dtype=torch.float32,device=device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            prediction=self.model(image[None,None,None].to(device),[instruction],state)
        actions=self.normalizer.denormalize(prediction.float(),stats_keys=['indooruav_vla'])[0,0].cpu().numpy()
        targets=cumulative_targets_to_world(current,actions)
        simulator=targets.copy();simulator[:,3]=np.deg2rad(simulator[:,3])
        return dict(cumulative_local_actions=actions,world_target_poses_degrees=targets,simulator_target_poses=simulator,action_representation='chunk_anchor_local')
