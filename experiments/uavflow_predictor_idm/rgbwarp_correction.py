"""Shared DA3 Current -> differentiable RGB warp -> Future correction.

No Future Predictor, feature objective, joint-view forward or uncertainty.
The controller does NOT own/register a second DA3 instance.
"""
import torch
from torch import nn

from .current_geometry import CurrentGeometryRead
from .patch_transport import nominal_intrinsics
from .rgbwarp_geometry import action_camera_transform,forward_splat,forward_alpha_splat
from .support_aware_transport import decode_levels


class CorrespondenceRead(CurrentGeometryRead):
    def __init__(self,dim,layers,width=256,heads=8):
        super().__init__(dim,width=width,heads=heads,memory_dim=dim,layer_indices=layers,gate_init=1.)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,future,current,mask,layer_index):
        if mask.shape!=(future.shape[0],future.shape[1],current.shape[1]):
            raise ValueError('Correspondence rows=future patches, columns=current patches')
        active=mask.any(-1)
        # An empty row has a safe softmax then an EXACT zero residual update.
        safe=torch.where(active[:,:,None],mask,torch.ones_like(mask))
        blocked=(~safe)[:,None].expand(-1,self.attention.num_heads,-1,-1).flatten(0,1)
        q=self.q(future);kv=self.memory(current)
        value,_=self.attention(q,kv,kv,attn_mask=blocked,need_weights=False)
        gate=self.gate+self.layer_gates[str(int(layer_index))]
        return future+gate*self.output(value)*active[:,:,None].to(value.dtype)


class RGBWarpCorrection(nn.Module):
    def __init__(self,language_dim,visual_dim,patch_size,layers,config=None):
        super().__init__();self.config=dict(config or {})
        allowed={'use_pixel_binary_mask','use_correspondence_attention','correspondence_layers',
                 'correspondence_width','correspondence_heads','depth_scale_meters','fov_degrees',
                 'heading_from_camera','warp_mode','alpha_radius','alpha_points_per_pixel'}
        if set(self.config)-allowed:raise ValueError(f'Unknown RGB warp settings: {set(self.config)-allowed}')
        self.warp_mode=str(self.config.get('warp_mode','alpha'))
        if self.warp_mode not in {'alpha','bilinear'}:raise ValueError('Choose alpha or bilinear RGB warp')
        # Persist effective defaults so old bilinear checkpoints cannot load
        # as alpha just because their saved config omitted a rasterizer tag.
        self.config['warp_mode']=self.warp_mode
        if self.warp_mode=='alpha':
            self.config.setdefault('alpha_radius',1.5)
            self.config.setdefault('alpha_points_per_pixel',8)
        self.layers=tuple(int(i) for i in self.config.get('correspondence_layers',layers))
        self.proposal_projection=nn.Sequential(nn.LayerNorm(language_dim),nn.Linear(language_dim,visual_dim))
        self.mask_projection=None
        if self.config.get('use_pixel_binary_mask',True):
            self.mask_projection=nn.Conv2d(1,visual_dim,patch_size,stride=patch_size)
            nn.init.zeros_(self.mask_projection.weight);nn.init.zeros_(self.mask_projection.bias)
        self.correspondence=(CorrespondenceRead(visual_dim,self.layers,
            width=int(self.config.get('correspondence_width',256)),heads=int(self.config.get('correspondence_heads',8)))
            if self.config.get('use_correspondence_attention',True) else None)
        if float(self.config.get('depth_scale_meters',100.))<=0:raise ValueError('Positive depth scale required')

    def forward(self,model,current_rgb,language,denormalize,intrinsics=None):
        if current_rgb.ndim!=4 or language.shape[:2]!=(current_rgb.shape[0],5) or denormalize is None:
            raise ValueError('RGB-warp needs current RGB, five native Qwen slots, ActionNormalizer')
        da3=model.da3;b,_,h,w=current_rgb.shape
        seed=self.proposal_projection(language)
        proposal=model.parallel_action_head(seed[:,None,None])
        current_shallow=da3.encode_shallow_rgb_grad(current_rgb,gradient_checkpointing=model.deep_gradient_checkpointing)
        observed=da3.propagate_shallow_visual_slots_grad(current_shallow,
            gradient_checkpointing=model.deep_gradient_checkpointing,
            return_layer_patches=self.correspondence is not None,
            layer_patch_indices=self.layers,layer_patch_mode='current')
        decoded=decode_levels(da3,observed['deep_levels'],1+int(da3.num_register_tokens))
        meters=decoded['depth'].reshape(b,h,w)*float(self.config.get('depth_scale_meters',100.))
        raw=denormalize(proposal.float())[:,0]
        transform=action_camera_transform(raw,self.config.get('heading_from_camera'))
        k=intrinsics if intrinsics is not None else nominal_intrinsics(h,w,
            fov_degrees=float(self.config.get('fov_degrees',90.)),device=current_rgb.device)
        warp=(forward_alpha_splat(current_rgb,meters,transform,k,patch_size=da3.PATCH_SIZE,
            radius=float(self.config.get('alpha_radius',1.5)),
            points_per_pixel=int(self.config.get('alpha_points_per_pixel',8)))
            if self.warp_mode=='alpha' else forward_splat(current_rgb,meters,transform,k,patch_size=da3.PATCH_SIZE))
        future_shallow=da3.encode_shallow_rgb_grad(warp['rgb'],
            invalid_mask=(1-warp['opacity'].detach() if self.warp_mode=='alpha' else (~warp['valid']).float())
                if self.mask_projection is not None else None,
            mask_projection=self.mask_projection,gradient_checkpointing=model.deep_gradient_checkpointing)
        # Preserve the VLM hidden-state path; numeric proposal values drive
        # only the warp, not a second embedding added to DA3 action tokens.
        deep=da3.propagate_shallow_with_actions_grad(future_shallow,seed[:,None,None],
            gradient_checkpointing=model.deep_gradient_checkpointing,decode_visuals=True,return_multi_level=True,
            correspondence_read=self.correspondence,
            correspondence_by_layer=observed.get('layer_patches',{}),
            correspondence_mask=warp['correspondence'])
        tokens=deep['action_tokens'].reshape(b,1,1,5,da3.embed_dim)
        z=model.parallel_action_correction_head(tokens)
        delta=model.gate_action_residual(z)
        diagnostics={'rgbwarp_valid_ratio':warp['valid'].float().mean().detach(),
                     'rgbwarp_correspondence_ratio':warp['correspondence'].float().mean().detach()}
        for i,axis in enumerate(('x','y','z','yaw')):
            diagnostics['rgbwarp_residual_abs_'+axis]=delta[...,i].abs().mean().detach()
            diagnostics['rgbwarp_residual_raw_abs_'+axis]=z[...,i].abs().mean().detach()
        if model.action_residual_gate_logit is not None:
            diagnostics['action_residual_gate']=model.action_residual_gate_logit.sigmoid().detach()
        return dict(actions_norm=proposal+delta,direct_actions_norm=proposal,refine_actions_norm=proposal+delta,
                    current_depth_output=decoded,deep_joint_features=deep,warp=warp,
                    proposal_physical=raw,residual=delta,diagnostics=diagnostics)
