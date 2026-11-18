"""Native GAM: final action + detached future-feature MSE + future depth only."""
import torch
from experiments.uavflow_predictor_idm.objectives import masked_action_l1,feature_loss
from experiments.uavflow_predictor_idm.runtime import autocast_context
from experiments.uavflow_predictor_idm.vlm_conditioning import encode_stage2_condition

def forward_native_gam_batch(*,model,da3,text,normalizer,batch,context_len,rollout_steps,amp,amp_dtype,
        depth_linear_weight=1.,depth_log_weight=0.,depth_grad_weight=1.,depth_log_epsilon_normalized=1e-6,
        stop_pos_weight=5.,feature_horizon_weights=(1.,),feature_patch_weight=1.,feature_cls_weight=0.,feature_register_weight=0.):
    from experiments.uavflow_predictor_idm.rgbwarp_objectives import depth_loss,pose5_state
    ref=model.module if hasattr(model,'module') else model
    images=batch['all_view_images'];assert context_len==rollout_steps==1 and images.shape[1:3]==(2,1)
    pose=pose5_state(batch['openvla_prompt_pose'],batch['frame_indices'])
    keys=list(batch['action_stats_key'])
    with autocast_context(images.device,enabled=bool(amp),dtype_name=amp_dtype):
        language=encode_stage2_condition(text,list(batch['task_description']),reference_images=images[:,0],current_images=images[:,0],pad_to=ref.predictor.language_len,current_pose=pose[:,0])
        output=model(images[:,0,0],reference_shallow=None,lang_feats=language['last_hidden_state'],lang_padding_mask=language['attention_mask'],gam_pose_embedding=(None if ref.rgbwarp_correction.config.get('vla_pose_in_vlm_only',False) else language['pose_embedding']),action_denormalizer=lambda x:normalizer.denormalize(x,stats_keys=keys))
        assert output['direct_actions_norm'] is None and output['current_depth_output'] is None
        # Future observations are target-only: evaluated AFTER policy prediction, detached.
        with torch.no_grad():
            if ref.rgbwarp_correction.config.get('native_frozen_shallow',False):
                from .frozen_shallow import fixed_features
                target_features=fixed_features(da3,images[:,1,0])
            else:
                target_features=da3.encode_shallow_rgb_grad(images[:,1,0],gradient_checkpointing=False).detach()
        features=feature_loss(output['predicted_shallow'],target_features,patch_start=1+int(da3.num_register_tokens),horizon_weights=list(feature_horizon_weights),patch_weight=feature_patch_weight,cls_weight=feature_cls_weight,register_weight=feature_register_weight)
        prediction=output['actions_norm'];truth_raw=batch['actions'][:,:1].float();mask=batch['action_loss_mask'][:,:1].bool();truth=normalizer.normalize(truth_raw,stats_keys=keys)
        action=masked_action_l1(prediction,truth,mask)
        scale=float(ref.rgbwarp_correction.config.get('depth_scale_meters',100.));future=output['deep_joint_features']['depth'].reshape_as(batch['gt_depth_meters'][:,1,0])
        depth=depth_loss(future,batch['gt_depth_meters'][:,1,0],batch['gt_depth_mask'][:,1,0],scale,linear_weight=depth_linear_weight,log_weight=depth_log_weight,grad_weight=depth_grad_weight,log_epsilon_normalized=depth_log_epsilon_normalized)
    raw=normalizer.denormalize(prediction.float(),stats_keys=keys);zero=prediction.new_zeros(())
    unused='deep_feature deep_feature_patch deep_feature_cls deep_feature_cos depth_scale_l1 depth_semantic_total depth_semantic_linear_l1 depth_semantic_log_l1 depth_semantic_grad depth_semantic_ratio relative_pose relative_pose_translation relative_pose_yaw relative_pose_unit action_rollout_pose action_rollout_pose_translation action_rollout_pose_yaw pose_consistency stop stop_accuracy stop_recall stop_rate stop_positive_probability stop_negative_probability stop_positive_probability_sum stop_negative_probability_sum stop_positive_count stop_negative_count stop_future_gate action_dual_current action_dual_future action_dual_future_gate gate'.split()
    values={k:zero for k in unused};values['deep_feature_per_level']=zero.new_zeros(0)
    for key in ('relative_pose_translation_axes','action_rollout_pose_translation_axes','action_rollout_pose_translation_axes_meters'):values[key]=zero.new_zeros(3)
    for key in ('stop_probabilities','stop_targets'):values[key]=zero.new_zeros(prediction.shape[0],1)
    error=(prediction.float()-truth).abs()*mask;raw_error=(raw-truth_raw).abs()*mask;denom=mask.float().sum((-1,-2)).clamp_min(1.)
    pred=raw.detach().reshape(-1,4);target=truth_raw.detach().reshape(-1,4);weight=mask.float().reshape(-1,4)
    moments=torch.stack((weight.sum(0),(pred*weight).sum(0),(pred.square()*weight).sum(0),(target*weight).sum(0),(target.square()*weight).sum(0),((pred-target).abs()*weight).sum(0)))
    values.update(action=action,direct_action=zero,refine_action=action,action_direct=zero,action_refine=action,depth=depth['total'],depth_global=depth['total'],depth_l1=depth['l1'],depth_linear_l1=depth['l1'],depth_log_l1=depth['log_l1'],depth_grad=depth['grad'],depth_valid_ratio=depth['valid_ratio'],depth_scene_scale_mean=zero.new_tensor(scale),raw_l1=masked_action_l1(raw,truth_raw,mask),action_raw_moments=moments,action_per_sample_step=(error.sum((-1,-2))/denom).detach(),raw_per_sample_step=(raw_error.sum((-1,-2))/denom).detach(),action_per_chunk_slot=error.sum((0,1,3))/mask.float().sum((0,1,3)).clamp_min(1.),raw_per_chunk_slot=raw_error.sum((0,1,3))/mask.float().sum((0,1,3)).clamp_min(1.),target_zero_slot_fraction=((truth_raw.abs().amax(-1)<1e-8)&mask.any(-1)).float().mean(),rgbwarp_refine_loss=action.detach())
    for name,key in [('feature','total'),('feature_patch','patch'),('feature_cls','cls'),('feature_register','register'),('feature_cos','cosine'),('feature_per_horizon','per_horizon'),('feature_patch_per_horizon','patch_per_horizon'),('feature_cls_per_horizon','cls_per_horizon'),('feature_register_per_horizon','register_per_horizon'),('feature_cos_per_horizon','cosine_per_horizon')]:values[name]=features[key]
    for key,value in depth.items():values['depth_future_'+key]=value.detach()
    values['depth_future_linear_l1']=depth['l1'].detach();values['depth_future_semantic_ratio']=zero
    for key,value in [('mean_m',future.detach().float().mean()*scale),('min_m',future.detach().float().amin()*scale),('max_m',future.detach().float().amax()*scale),('zero_fraction',(future.detach()<=1e-6).float().mean())]:values['rgbwarp_future_depth_'+key]=value
    from experiments.uavflow_predictor_idm.window_metrics import action_window_sums
    sums=action_window_sums(prediction.detach(),prediction.detach(),truth,mask,batch['frame_indices']);sums[:,0]=0;values['_window_action_stats']=sums
    count=mask.sum().to(action.dtype);values['_global_log_action_stats']=torch.stack((zero,action.detach()*count,count))
    return values
