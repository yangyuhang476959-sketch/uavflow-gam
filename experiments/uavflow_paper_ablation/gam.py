"""Matched GAM future-feature controls, reusing the native GAMFuturePredictor.

VLA-GAM seeds five slots from Qwen; UAV-GAM uses T5/Pose language memory and
the Predictor's existing five action slots. Neither performs RGB warping.
"""
import torch
from torch import nn
from robot.modeling.future_predictor import GAMFuturePredictor
from experiments.uavflow_predictor_idm.rgbwarp_correction import RGBWarpCorrection
from experiments.uavflow_predictor_idm.support_aware_transport import decode_levels


class MatchedFuturePredictor(GAMFuturePredictor):
    def _get_flex_block_mask(self,H,V,device,lang_len=0):
        # The installed flex backend can import on CPU but cannot compile a
        # CPU attention kernel. Keep the native dense SDPA fallback there.
        if device.type!='cuda' or getattr(self,'native_sdpa',False):return None
        return super()._get_flex_block_mask(H,V,device,lang_len)


class MatchedGAMCorrection(RGBWarpCorrection):
    def __init__(self, language_dim, visual_dim, patch_size, layers, config, *,
                 predictor_options, num_register_tokens):
        config=dict(config)
        self.mode=config.get('branch_mode')
        if (self.mode not in {'uav_gam','vla_gam'} or config.get('action_output_mode')!='direct'
                or config.get('use_correspondence_attention',True)
                or config.get('use_pixel_binary_mask',True)):
            raise ValueError('Matched GAM requires uav_gam/vla_gam, direct actions, no Warp mask or correspondence CA')
        super().__init__(language_dim,visual_dim,patch_size,layers,config)
        self.config.setdefault('pose_conditioning','sa_proprio')
        self.config.setdefault('pose_projection_init_seed',47)
        if self.config['pose_conditioning']!='sa_proprio':
            raise ValueError('Matched GAM requires the native SA proprio-token path')
        options=dict(predictor_options)
        self.config['predictor_options']=options
        # Construct both controls identically, then disable only unused paths.
        # A separate RNG stream preserves every preceding shared action tensor.
        seed=int(options.pop('init_seed',45))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.future_predictor=MatchedFuturePredictor(
                d_da3=visual_dim,language_dim=int(self.config.get('native_language_dim',language_dim)),
                num_patches_per_view=(224//patch_size)**2,
                num_register_tokens=num_register_tokens,use_language=True,
                variable_language_tokens=not bool(self.config.get('native_fixed_language_positions',False)),use_proprio_input=True,proprio_dim=language_dim,
                action_dim=4,action_chunk_size=1,num_action_slots=5,
                action_slot_position_encoding=False,input_proj_norm='ln',
                condition_mode='cross_attn',**options)
            # The public Pose MLP has already encoded native5D to2048.
            # This separate width adapter uses GAM's original SA proprio slot.
            torch.random.default_generator.manual_seed(int(self.config['pose_projection_init_seed']))
            self.future_predictor.proprio_token_proj=nn.Linear(language_dim,int(options['d_model']))
        self.future_predictor.native_sdpa=bool(self.config.get('native_sdpa',False))
        if self.config.get('gam_action_slot_position_encoding',False):
            # Restore the native slot-embedding mechanism without consuming the
            # RNG stream that initializes existing Predictor/SA Pose parameters.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.config.get('gam_action_slot_init_seed',48)))
                self.future_predictor.action_slot_embed=nn.Parameter(
                    torch.empty(1,1,5,self.future_predictor.d_model))
                nn.init.normal_(self.future_predictor.action_slot_embed,std=0.02)
        if self.config.get('vla_pose_in_vlm_only',False):
            if self.mode!='vla_gam' or not self.config.get('native_gam_path',False):
                raise ValueError('VLM-only Pose is specific to native VLA-GAM')
            # Preserve common construction/init, but remove the separate SA token.
            self.future_predictor.use_proprio_input=False
            self.future_predictor.extra_tokens_per_step=self.future_predictor.num_action_slots
            self.future_predictor.proprio_token_proj.requires_grad_(False)
        for module in (self.future_predictor.future_proprio_proj,
                       self.future_predictor.out_proprio_norm):
            module.requires_grad_(False)
        if self.mode=='uav_gam':
            self.proposal_projection.requires_grad_(False)  # no Qwen proposal
        else:
            self.future_predictor.lang_proj.requires_grad_(False)
            if self.future_predictor.lang_pos is not None:self.future_predictor.lang_pos.requires_grad_(False)
            for block in self.future_predictor.blocks:
                for module in (block.norm_cross,block.cross_attn,block.ls_cross):
                    module.requires_grad_(False)

    def forward(self,model,current_rgb,language,denormalize,intrinsics=None,
                stop_pose=None,language_mask=None,pose_embedding=None):
        if current_rgb.ndim!=4 or language.ndim!=3 or language.shape[0]!=current_rgb.shape[0]:
            raise ValueError('GAM requires current RGB and batched conditioning tokens')
        if self.future_predictor.use_proprio_input and (pose_embedding is None or pose_embedding.shape!=(current_rgb.shape[0],self.future_predictor.proprio_dim)):
            raise ValueError('GAM requires the current canonical Pose MLP embedding for its SA proprio token')
        da3=model.da3
        if self.config.get('native_frozen_shallow',False):
            from .frozen_shallow import fixed_features
            shallow=fixed_features(da3,current_rgb)
        else:
            shallow=da3.encode_shallow_rgb_grad(current_rgb,
                gradient_checkpointing=model.deep_gradient_checkpointing)
        native=bool(self.config.get('native_gam_path',False))
        current_depth=None
        if not native:
            observed=da3.propagate_shallow_visual_slots_grad(shallow,gradient_checkpointing=model.deep_gradient_checkpointing)
            current_depth=decode_levels(da3,observed['deep_levels'],1+int(da3.num_register_tokens))
        seed=None
        proposal=None
        if self.mode=='vla_gam':
            if language.shape[1]!=5:raise ValueError('VLA-GAM requires exactly five Qwen Action Slots')
            seed=self.proposal_projection(language)
            if not native:proposal=model.parallel_action_head(seed[:,None,None])
        predicted=self.future_predictor(
            past_visual_tokens=shallow,
            proprio_history=(pose_embedding[:,None] if self.future_predictor.use_proprio_input else None),
            past_action_history=shallow.new_zeros(current_rgb.shape[0],1,1,4),
            action_slot_seed=(None if seed is None else self.future_predictor.in_proj(seed)),
            lang_feats=language if self.mode=='uav_gam' else None,
            lang_padding_mask=language_mask if self.mode=='uav_gam' else None)
        future=predicted['predicted_next_visual_tokens'][:,-1:]
        action_tokens=predicted['predicted_action_tokens'][:,-1:]
        if proposal is None and not native:proposal=model.parallel_action_head(action_tokens)
        deep=da3.propagate_shallow_with_actions_grad(future,action_tokens,
            gradient_checkpointing=model.deep_gradient_checkpointing,
            decode_visuals=True,return_multi_level=True)
        tokens=deep['action_tokens'].reshape(current_rgb.shape[0],1,1,5,da3.embed_dim)
        final=model.parallel_action_correction_head(tokens)
        stop_logits=None
        if model.stop_head_enabled:
            if model.stop_head_mode!='action_hidden_pose' or stop_pose is None:
                raise ValueError('Matched GAM Stop requires action-hidden/pose MLP and normalized5D Pose')
            stop_logits=model.stop_head(tokens[:,:,0],stop_pose)
        return dict(actions_norm=final,direct_actions_norm=proposal,refine_actions_norm=final,
                    current_depth_output=current_depth,deep_joint_features=deep,
                    predicted_shallow=future,warp=None,diagnostics={},stop_logits=stop_logits)
