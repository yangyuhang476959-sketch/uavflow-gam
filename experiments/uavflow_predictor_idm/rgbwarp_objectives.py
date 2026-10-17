"""Fixed REAL t+5 depth regularizes the proposal through differentiable warp.

This is intentionally a target-view CONTROL regularizer, not a claim of
pixel-aligned depth reconstruction at an arbitrary proposal camera pose.
"""
import torch
from .objectives import masked_action_l1,_finite_difference_gradient_l1
from .runtime import autocast_context
from .vlm_conditioning import encode_stage2_condition


def depth_loss(prediction,target,valid,scale,*,linear_weight=1.,log_weight=0.,grad_weight=1.,
               log_epsilon_normalized=1e-6):
    prediction=prediction.float();target=target.float()/float(scale)
    if prediction.shape!=target.shape or valid.shape!=target.shape:
        raise ValueError('Depth targets and predictions must share pixel grid')
    valid=valid.bool() & torch.isfinite(target) & (target>0)
    # Replace invalid targets before arithmetic; NaN*0 is not zero.
    target=torch.where(valid,target,torch.zeros_like(target))
    weight=valid.float();denom=weight.sum().clamp_min(1.)
    mse=((prediction-target).square()*weight).sum()/denom
    l1=((prediction-target).abs()*weight).sum()/denom
    grad=_finite_difference_gradient_l1(prediction,target,valid,torch.ones_like(prediction))
    if float(log_epsilon_normalized)<=0:
        raise ValueError('log_epsilon_normalized must be positive')
    safe_target=torch.where(valid,target,torch.ones_like(target))
    log_l1=((prediction.clamp_min(float(log_epsilon_normalized)).log()
             -safe_target.clamp_min(float(log_epsilon_normalized)).log()).abs()*weight).sum()/denom
    return dict(total=float(linear_weight)*l1+float(log_weight)*log_l1+float(grad_weight)*grad,mse=mse,l1=l1,
                log_l1=log_l1,grad=grad,valid_ratio=weight.mean())


def forward_rgbwarp_batch(*,model,da3,text,normalizer,batch,context_len,rollout_steps,amp,amp_dtype,
                         depth_linear_weight=1.,depth_log_weight=0.,depth_grad_weight=1.,
                         depth_log_epsilon_normalized=1e-6):
    ref=model.module if hasattr(model,'module') else model
    images=batch['all_view_images']
    if context_len!=1 or rollout_steps!=1 or images.shape[1:3]!=(2,1):
        raise ValueError('RGB-warp currently requires exactly Ft,F(t+5), H=1,V=1')
    pose=batch.get('episode_pose_progress' if getattr(text,'pose_progress_enabled',False) else 'openvla_prompt_pose')
    if pose is None:raise KeyError('RGB-warp requires canonical current pose [cm,deg]')
    keys=list(batch['action_stats_key'])
    with autocast_context(images.device,enabled=bool(amp),dtype_name=amp_dtype):
        language=encode_stage2_condition(text,list(batch['task_description']),
            reference_images=images[:,0],current_images=images[:,0],
            pad_to=ref.predictor.language_len,current_pose=pose[:,0])
        output=model(images[:,0,0],reference_shallow=None,
            lang_feats=language['last_hidden_state'],lang_padding_mask=language['attention_mask'],
            action_denormalizer=lambda x:normalizer.denormalize(x,stats_keys=keys),
            camera_intrinsics=batch.get('camera_intrinsics'))
        target_raw=batch['actions'][:,:1].float();mask=batch['action_loss_mask'][:,:1].bool()
        target=normalizer.normalize(target_raw,stats_keys=keys)
        proposal_loss=masked_action_l1(output['direct_actions_norm'],target,mask)
        refine_loss=masked_action_l1(output['actions_norm'],target,mask)
        scale=float(ref.rgbwarp_correction.config.get('depth_scale_meters',100.))
        gt=batch['gt_depth_meters'];valid=batch['gt_depth_mask']
        current=output['current_depth_output']['depth'].reshape_as(gt[:,0,0])
        future=output['deep_joint_features']['depth'].reshape_as(gt[:,1,0])
        # DO NOT warp/detach the target following proposal, and DO NOT exclude
        # predicted warp holes: target is the fixed real t+5 GT-valid region.
        depth_options=dict(linear_weight=depth_linear_weight,log_weight=depth_log_weight,
                           grad_weight=depth_grad_weight,log_epsilon_normalized=depth_log_epsilon_normalized)
        curr_loss=depth_loss(current,gt[:,0,0],valid[:,0,0],scale,**depth_options)
        fut_loss=depth_loss(future,gt[:,1,0],valid[:,1,0],scale,**depth_options)
    prediction=output['actions_norm'];raw=normalizer.denormalize(prediction.float(),stats_keys=keys)
    zero=prediction.new_zeros(())
    # Compatibility-only zero diagnostics for the existing trainer/evaluator.
    # No feature/pose/stop head is evaluated or supervised in this mode.
    unused='feature feature_patch feature_cls feature_register feature_cos deep_feature deep_feature_patch deep_feature_cls deep_feature_cos depth_log_l1 depth_scale_l1 depth_semantic_total depth_semantic_linear_l1 depth_semantic_log_l1 depth_semantic_grad depth_semantic_ratio relative_pose relative_pose_translation relative_pose_yaw relative_pose_unit action_rollout_pose action_rollout_pose_translation action_rollout_pose_yaw pose_consistency stop stop_accuracy stop_recall stop_rate stop_positive_probability stop_negative_probability stop_positive_probability_sum stop_negative_probability_sum stop_positive_count stop_negative_count stop_future_gate action_dual_current action_dual_future action_dual_future_gate gate'.split()
    values={k:zero for k in unused}
    for key in ('feature_per_horizon','feature_patch_per_horizon','feature_cls_per_horizon',
                'feature_register_per_horizon','feature_cos_per_horizon'):
        values[key]=zero.new_zeros(1)
    values['deep_feature_per_level']=zero.new_zeros(0)
    for key in ('relative_pose_translation_axes','action_rollout_pose_translation_axes','action_rollout_pose_translation_axes_meters'):
        values[key]=zero.new_zeros(3)
    for key in ('stop_probabilities','stop_targets'):values[key]=zero.new_zeros(prediction.shape[0],1)
    error=(prediction.float()-target).abs()*mask
    raw_error=(raw-target_raw).abs()*mask
    action_per_step=error.sum((-1,-2))/mask.float().sum((-1,-2)).clamp_min(1.)
    raw_per_step=raw_error.sum((-1,-2))/mask.float().sum((-1,-2)).clamp_min(1.)
    pred=raw.detach().reshape(-1,4);truth=target_raw.detach().reshape(-1,4);weight=mask.float().reshape(-1,4)
    moments=torch.stack((weight.sum(0),(pred*weight).sum(0),(pred.square()*weight).sum(0),
                         (truth*weight).sum(0),(truth.square()*weight).sum(0),((pred-truth).abs()*weight).sum(0)))
    total_depth=(curr_loss['total']+fut_loss['total'])/2
    values.update(action=(proposal_loss+refine_loss)/2,direct_action=proposal_loss,refine_action=refine_loss,
        action_direct=proposal_loss,action_refine=refine_loss,depth=total_depth,depth_global=total_depth,
        depth_l1=(curr_loss['l1']+fut_loss['l1'])/2,depth_linear_l1=(curr_loss['l1']+fut_loss['l1'])/2,
        depth_log_l1=(curr_loss['log_l1']+fut_loss['log_l1'])/2,
        depth_grad=(curr_loss['grad']+fut_loss['grad'])/2,depth_valid_ratio=(curr_loss['valid_ratio']+fut_loss['valid_ratio'])/2,
        depth_scene_scale_mean=zero.new_tensor(scale),
        raw_l1=masked_action_l1(raw,target_raw,mask),action_raw_moments=moments,
        action_per_sample_step=action_per_step.detach(),raw_per_sample_step=raw_per_step.detach(),
        action_per_chunk_slot=error.sum((0,1,3))/mask.float().sum((0,1,3)).clamp_min(1.),
        raw_per_chunk_slot=raw_error.sum((0,1,3))/mask.float().sum((0,1,3)).clamp_min(1.),
        target_zero_slot_fraction=((target_raw.abs().amax(-1)<1e-8)&mask.any(-1)).float().mean(),
        **output['diagnostics'])
    values['rgbwarp_proposal_loss']=proposal_loss.detach()
    values['rgbwarp_refine_loss']=refine_loss.detach()
    for branch,loss in [('current',curr_loss),('future',fut_loss)]:
        for key,value in loss.items():values[f'depth_{branch}_{key}']=value.detach()
        # MSE is diagnostic only; reconstruction uses linear/log L1.
        values[f'depth_{branch}_linear_l1']=loss['l1'].detach()
        values[f'depth_{branch}_semantic_ratio']=zero
    return values
