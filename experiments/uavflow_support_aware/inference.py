"""Checkpoint loading and observation-only B/C inference for policy services."""
from __future__ import annotations

import inspect

import torch
from omegaconf import OmegaConf

from experiments.uavflow_predictor_idm.runtime import autocast_context


def canonical_yaw4d_state(current_images,instructions,current_prompt_pose,
                          first_sim_pose,current_sim_pose):
    """Same state conversion as Dataset; never silently accept world xyz."""
    if first_sim_pose is not None or current_sim_pose is not None:
        if first_sim_pose is None or current_sim_pose is None or current_prompt_pose is not None:
            raise ValueError('Provide either both raw poses or a canonical state, not both')
        from robot.data.uavflow_dataset import episode_yaw4d_state
        import numpy as np
        first=torch.as_tensor(first_sim_pose).detach().cpu().numpy()
        current=torch.as_tensor(current_sim_pose).detach().cpu().numpy()
        if first.shape!=(len(instructions),6) or current.shape!=(len(instructions),6):
            raise ValueError('Raw simulator poses must both be [B,6]')
        current_prompt_pose=np.stack([episode_yaw4d_state(a,b) for a,b in zip(first,current)])
    if current_prompt_pose is None:
        raise ValueError('Inference requires canonical relative [B,4] cm/deg state')
    state=torch.as_tensor(current_prompt_pose,device=current_images.device,dtype=torch.float32)
    if state.shape!=(len(instructions),4) or not torch.isfinite(state).all():
        raise ValueError('Inference requires finite canonical relative [B,4] cm/deg state')
    return state


def canonical_progress_state(current_images, instructions, current_prompt_pose,
                             observed_pose_history, first_sim_pose=None, current_sim_pose=None):
    """7D state from actual observed history, never from unexecuted actions.

    History may be ragged (one [T,6] array per batch item) and must begin at
    the episode start. Alternatively callers can maintain and provide the 7D
    canonical state themselves. Two endpoints cannot recover motion progress.
    """
    from robot.data.uavflow_dataset import episode_pose_progress
    import numpy as np
    if observed_pose_history is not None:
        if current_prompt_pose is not None:
            raise ValueError('Provide history OR canonical progress state, not both')
        if len(observed_pose_history)!=len(instructions):
            raise ValueError('Expected one observed pose history per batch item')
        histories=[torch.as_tensor(h).detach().cpu().numpy() for h in observed_pose_history]
        states=[episode_pose_progress(h)[-1] for h in histories]
        for endpoints, index in ((first_sim_pose,0),(current_sim_pose,-1)):
            if endpoints is not None:
                values=torch.as_tensor(endpoints).detach().cpu().numpy()
                if values.shape!=(len(histories),6) or not np.allclose(values,np.stack([h[index] for h in histories])):
                    raise ValueError('History endpoints disagree with raw simulator poses')
        current_prompt_pose=np.stack(states)
    elif first_sim_pose is not None or current_sim_pose is not None:
        raise ValueError('Progress needs actual observed pose history, not only two endpoints')
    if current_prompt_pose is None:
        raise ValueError('Provide observed_pose_history or canonical [B,7] progress state')
    state=torch.as_tensor(current_prompt_pose,device=current_images.device,dtype=torch.float32)
    if state.shape!=(len(instructions),7) or not torch.isfinite(state).all():
        raise ValueError('Expected finite [B,7] xyz(m), sin/cos, accumulated yaw(rad), path(m)')
    if (state[:,-1]<0).any():
        raise ValueError('Travelled path must be nonnegative')
    return state


class SupportAwarePolicy:
    def __init__(self, model, conditioner, normalizer, *, amp=True, amp_dtype="auto",
                 pose_state_mode='preprocessed'):
        self.model, self.conditioner, self.normalizer = model.eval(), conditioner.eval(), normalizer
        self.amp, self.amp_dtype = amp, amp_dtype
        self.pose_state_mode=pose_state_mode

    @torch.inference_mode()
    def predict(self, current_images, instructions, *, reference_images,
                current_prompt_pose=None, stats_keys=None, camera_intrinsics=None,
                first_sim_pose=None,current_sim_pose=None,observed_pose_history=None):
        """Images [B,1,3,224,224], same preprocessing as the training dataset.

        New metric current_bank checkpoints use first-yaw-frame cm/deg state.
        Legacy preprocessed checkpoints retain their full-first-rotation state;
        they must not be silently interpreted as yaw4d checkpoints.
        """
        from experiments.uavflow_predictor_idm.vlm_conditioning import encode_stage2_condition
        if (current_images.ndim != 5 or current_images.shape != reference_images.shape
                or current_images.shape[1:3] != (1, 3)):
            raise ValueError("Current/reference must have matching one-camera image batches")
        b = current_images.shape[0]
        if getattr(self.conditioner,'pose_progress_enabled',False):
            current_prompt_pose=canonical_progress_state(current_images,instructions,current_prompt_pose,
                observed_pose_history,first_sim_pose,current_sim_pose)
        elif observed_pose_history is not None:
            raise ValueError('Observed progress history requires a progress-enabled checkpoint')
        elif self.pose_state_mode=='evaluator_yaw4d':
            current_prompt_pose=canonical_yaw4d_state(current_images,instructions,current_prompt_pose,
                                                     first_sim_pose,current_sim_pose)
        elif first_sim_pose is not None or current_sim_pose is not None:
            raise ValueError('Raw yaw4d pose conversion is only valid for evaluator_yaw4d checkpoints')
        with autocast_context(current_images.device, enabled=self.amp, dtype_name=self.amp_dtype):
            images = torch.cat((reference_images, current_images), 1)
            shallow = self.model.da3.encode_shallow_visual_slots(images, T=2, V=1)["visual_tokens"]
            language = encode_stage2_condition(
                self.conditioner, instructions, reference_images=reference_images,
                current_images=current_images, current_pose=current_prompt_pose,
                pad_to=int(self.model.predictor.language_len))
            output = self.model(
                shallow[:, 1:], reference_shallow=shallow[:, :1],
                lang_feats=language["last_hidden_state"], lang_padding_mask=language["attention_mask"],
                action_denormalizer=lambda x: self.normalizer.denormalize(x, stats_keys=stats_keys),
                camera_intrinsics=camera_intrinsics)
        output["actions_meters_radians"] = self.normalizer.denormalize(
            output["actions_norm"].float(), stats_keys=stats_keys)[:, 0]
        return output


class RGBWarpPolicy(SupportAwarePolicy):
    @torch.inference_mode()
    def predict(self,current_images,instructions,*,reference_images=None,current_prompt_pose=None,
                first_sim_pose=None,current_sim_pose=None,stats_keys=None,camera_intrinsics=None,
                observed_pose_history=None):
        """Observation only. Canonical state is first-YAW-frame cm/degrees.

        Prefer raw simulator poses, converted by EXACT Dataset helper. An
        existing service may provide already canonical current_prompt_pose.
        Never pass simulator world xyz directly as a relative state.
        """
        if getattr(self.conditioner,'pose_progress_enabled',False):
            current_prompt_pose=canonical_progress_state(current_images,instructions,current_prompt_pose,
                observed_pose_history,first_sim_pose,current_sim_pose)
        else:
            if observed_pose_history is not None:
                raise ValueError('Observed progress history requires a progress-enabled checkpoint')
            current_prompt_pose=canonical_yaw4d_state(current_images,instructions,current_prompt_pose,
                                                     first_sim_pose,current_sim_pose)
        from experiments.uavflow_predictor_idm.vlm_conditioning import encode_stage2_condition
        if (current_images.ndim!=5 or current_images.shape[1:3]!=(1,3)
                or current_images.shape[0]!=len(instructions)):
            raise ValueError('Expected current RGB [B,1,3,H,W]')
        with autocast_context(current_images.device,enabled=self.amp,dtype_name=self.amp_dtype):
            language=encode_stage2_condition(self.conditioner,instructions,
                reference_images=current_images,current_images=current_images,
                current_pose=current_prompt_pose,pad_to=self.model.predictor.language_len)
            output=self.model(current_images[:,0],reference_shallow=None,
                lang_feats=language['last_hidden_state'],lang_padding_mask=language['attention_mask'],
                action_denormalizer=lambda x:self.normalizer.denormalize(x,stats_keys=stats_keys),
                camera_intrinsics=camera_intrinsics)
        output['actions_meters_radians']=self.normalizer.denormalize(output['actions_norm'].float(),stats_keys=stats_keys)[:,0]
        return output


def load_policy(checkpoint_path, *, device="cuda:0", da3_checkpoint=None, qwen_model=None):
    from experiments.uavflow_predictor_idm.runtime import bootstrap_accelerator_plugin
    bootstrap_accelerator_plugin()
    from robot.data.dataset import ActionNormalizer
    from robot.modeling.da3_giant_encoder import DA3GiantEncoder
    from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
    from experiments.uavflow_predictor_idm.geometry_architectures import load_architecture_state
    from experiments.uavflow_predictor_idm.vlm_conditioning import build_stage2_conditioner
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    if (not cfg.model.get("support_aware_transport", {}).get("enabled", False)
            and cfg.model.get('correction_mode', 'legacy') == 'legacy'
            and not cfg.model.get('dual_stream_rgbwarp_enabled',False)):
        raise ValueError("Checkpoint is not a Candidate B/C policy")
    if da3_checkpoint is not None:
        cfg.stage1.da3_checkpoint = str(da3_checkpoint)
    if qwen_model is not None:
        cfg.stage1.qwen_model = str(qwen_model)
    da3 = DA3GiantEncoder(ckpt_path=str(cfg.stage1.da3_checkpoint), encoder_input_size=224,
                         freeze_backbone=True, n_action_steps=0, views_per_timestep=1,
                         use_temporal_embed=False).to(device).eval()
    lora = cfg.model.get("deep_lora", {}) or {}
    if cfg.model.get('dual_stream_rgbwarp_enabled',False):
        lora=dict(cfg.model.da3_lora)
        lora.update(start_block=0,target_modules=['all-linear'])
    if lora.get("enabled", False):
        da3.configure_deep_lora(block_idx=int(lora.get("start_block", 13)), rank=int(lora.get("rank", 64)),
                               alpha=float(lora.get("alpha", 128.)), dropout=float(lora.get("dropout", 0.)),
                               target_modules=lora.get("target_modules"))
    conditioner = build_stage2_conditioner(cfg.stage1, cfg.model).to(device).eval()
    options = OmegaConf.to_container(cfg.model, resolve=True)
    options = {key: value for key, value in options.items()
               if key in inspect.signature(UAVFlowPredictorIDM.__init__).parameters}
    options.update(da3=da3, idm=None, action_dim=4, language_dim=int(conditioner.hidden_size),
                   variable_language_tokens=bool(getattr(conditioner, "variable_length_tokens", False)),
                   depth_decode_enabled=True)
    model = UAVFlowPredictorIDM(**options).to(device)
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    model.semantic_geometry_action.load_state_dict(checkpoint["semantic_geometry_action"], strict=True)
    model.parallel_action_head.load_state_dict(checkpoint["parallel_action_head"], strict=True)
    model.parallel_action_correction_head.load_state_dict(checkpoint["parallel_action_correction_head"], strict=True)
    load_architecture_state(model, checkpoint)
    from experiments.uavflow_predictor_idm.current_geometry import load_current_geometry_read
    load_current_geometry_read(model, checkpoint, required=model.current_geometry_read is not None)
    for name in ("reference_step_embed", "missing_action_embed", "missing_pose_embed", "residual_gate_logit"):
        if name in checkpoint:
            with torch.no_grad():
                getattr(model, name).copy_(checkpoint[name].to(device))
    strict_metric=(model.dual_stream_rgbwarp_enabled or (
        cfg.dataset.get('openvla_prompt_pose_mode')=='evaluator_yaw4d'
        and bool(cfg.stage1.get('qwen_numeric_pose_token',False))))
    def load_trainable(module,state,label):
        if strict_metric:
            expected={n for n,p in module.named_parameters() if p.requires_grad}
            missing=expected-set(state)
            if missing:
                raise ValueError(f'Missing {label} trainable weights: {sorted(missing)}')
        status=module.load_state_dict(state,strict=False)
        if strict_metric and status.unexpected_keys:
            raise ValueError(f'Unexpected {label} weights: {status.unexpected_keys}')
    load_trainable(da3,checkpoint.get('da3_trainable',{}),'DA3')
    if checkpoint.get("conditioner_trainable") is not None:
        load_trainable(conditioner,checkpoint['conditioner_trainable'],'Qwen')
    elif strict_metric:
        raise KeyError('Metric correction checkpoint must carry Qwen LoRA/slot/state weights')
    if "action_normalizer" not in checkpoint:
        raise KeyError("Candidate B checkpoint must carry its action_normalizer")
    normalizer = ActionNormalizer.from_state_dict(checkpoint["action_normalizer"])
    policy_type=RGBWarpPolicy if model.dual_stream_rgbwarp_enabled else SupportAwarePolicy
    return policy_type(model, conditioner, normalizer, amp=bool(cfg.training.amp),
                              pose_state_mode=str(cfg.dataset.get('openvla_prompt_pose_mode','preprocessed')),
                              amp_dtype=str(cfg.training.get("amp_dtype", "auto")))
