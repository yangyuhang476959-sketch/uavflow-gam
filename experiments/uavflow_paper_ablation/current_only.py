"""One current-image DA3 pass conditioned on the native Qwen Action Slots.

Reuse the Warp projection and two independent action heads. Numeric proposal
actions are auxiliary outputs only: no camera transform, rasterizer, future
image, memory bank, residual addition, or learned Future Predictor is used.
"""
from experiments.uavflow_predictor_idm.rgbwarp_correction import RGBWarpCorrection


class CurrentOnlyCorrection(RGBWarpCorrection):
    def __init__(self, language_dim, visual_dim, patch_size, layers, config):
        config = dict(config)
        if (config.get('branch_mode') not in {'current_only', 'future_only'}
                or config.get('action_output_mode') != 'direct'
                or config.get('use_correspondence_attention', True)
                or config.get('use_pixel_binary_mask', True)):
            raise ValueError('Current-only requires direct actions, no correspondence CA or warp mask embedding')
        super().__init__(language_dim, visual_dim, patch_size, layers, config)

    def forward(self, model, current_rgb, language, denormalize,
                intrinsics=None, stop_pose=None):
        if current_rgb.ndim != 4 or language.shape[:2] != (current_rgb.shape[0], 5):
            raise ValueError('Current-only requires current RGB and five native Qwen slots')
        da3 = model.da3
        seed = self.proposal_projection(language)
        proposal = None if self.config.get('current_final_only',False) else model.parallel_action_head(seed[:, None, None])
        shallow = da3.encode_shallow_rgb_grad(
            current_rgb, gradient_checkpointing=model.deep_gradient_checkpointing)
        deep = da3.propagate_shallow_with_actions_grad(
            shallow, seed[:, None, None],
            gradient_checkpointing=model.deep_gradient_checkpointing,
            decode_visuals=True, return_multi_level=True)
        tokens = deep['action_tokens'].reshape(current_rgb.shape[0], 1, 1, 5, da3.embed_dim)
        prediction = model.parallel_action_correction_head(tokens)
        stop_logits = None
        if model.stop_head_enabled:
            if model.stop_head_mode != 'action_hidden_pose' or stop_pose is None:
                raise ValueError('Current-only Stop requires the existing action-hidden/pose MLP and normalized5D Pose')
            stop_logits = model.stop_head(tokens[:, :, 0], stop_pose)
        future = self.config.get('branch_mode') == 'future_only'
        return dict(actions_norm=prediction, direct_actions_norm=proposal,
                    refine_actions_norm=prediction, current_depth_output=None if future else deep,
                    current_joint_features=None if future else deep, deep_joint_features=deep if future else None, warp=None,
                    diagnostics={}, stop_logits=stop_logits)
