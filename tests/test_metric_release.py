import torch
import pytest

from experiments.uavflow_predictor_idm.metric_train import build_config,VARIANTS
from experiments.uavflow_support_aware.inference import canonical_yaw4d_state
from robot.data.uavflow_dataset import episode_yaw4d_state


@pytest.mark.parametrize('variant',VARIANTS)
def test_three_metric_release_contracts(variant,tmp_path,monkeypatch):
    monkeypatch.setenv('MODEL_ROOT',str(tmp_path/'models'))
    monkeypatch.setenv('OUTPUT_ROOT',str(tmp_path/'output'))
    cfg=build_config(variant)
    assert cfg.loss.depth_scale_mode=='fixed_metric'
    assert cfg.loss.depth_fixed_scale_meters==100
    assert cfg.dataset.openvla_prompt_pose_mode=='evaluator_yaw4d'
    assert cfg.dataset.pose_progress_enabled and cfg.model.pose_progress_enabled
    assert cfg.stage1.qwen_pose_progress_enabled==(variant!='gam')
    assert cfg.dataset.visual_anchor_stride==5 and cfg.model.action_chunk_size==5
    assert not cfg.model.train_deep_backbone
    assert not cfg.model.stop_head_enabled and cfg.loss.stop_weight==0
    assert cfg.training.max_epochs==5 and cfg.training.lr_schedule=='constant'
    assert cfg.loss.action_weight==3 and cfg.loss.depth_weight==3
    assert cfg.loss.depth_linear_weight==cfg.loss.depth_log_weight==cfg.loss.depth_grad_weight==pytest.approx(1./3.)
    assert cfg.stage1.t5_model==str(tmp_path/'models/t5-base')
    assert cfg.stage1.da3_checkpoint==str(tmp_path/'models/track4world_da3.pth')
    assert cfg.stage1.qwen_model==str(tmp_path/'models/qwen3.5-2b')
    if variant=='gam':
        assert cfg.loss.feature_weight==1
        assert cfg.loss.action_direct_weight==0 and cfg.loss.action_refine_weight==1
        assert cfg.model.get('action_residual_gate_init') is None
        assert cfg.model.use_pose_history and cfg.model.use_fixed_first_frame
        assert not cfg.model.parallel_vla_gfm_enabled
        assert cfg.model.correction_mode=='legacy' and cfg.stage1.text_encoder_type=='t5'
        assert cfg.model.deep_lora.enabled and cfg.model.deep_lora.start_block==13
        assert cfg.loss.depth_target_mode=='future'
    elif variant=='current_bank':
        assert cfg.loss.feature_weight==0
        assert cfg.loss.action_direct_weight==cfg.loss.action_refine_weight==.5
        assert cfg.model.action_residual_gate_init==.1
        assert cfg.model.correction_mode=='pure'
        assert cfg.model.parallel_current_geometry_read_enabled
        assert cfg.model.current_geometry_read_mode=='per_layer'
        assert cfg.model.current_geometry_bank_mode=='output_current'
        assert cfg.stage1.qwen_numeric_pose_token
        assert cfg.model.deep_lora.enabled and cfg.model.deep_lora.start_block==13
        assert cfg.loss.depth_target_mode=='both'
    else:
        assert cfg.loss.feature_weight==0
        assert cfg.loss.action_direct_weight==cfg.loss.action_refine_weight==.5
        assert cfg.model.action_residual_gate_init==.1
        assert cfg.model.dual_stream_rgbwarp_enabled
        assert cfg.model.da3_lora.enabled and cfg.model.da3_lora.target_modules==['all-linear']
        assert cfg.model.rgbwarp_correction.use_correspondence_attention
        assert cfg.model.rgbwarp_correction.warp_mode=='alpha'
        assert cfg.model.rgbwarp_correction.alpha_radius==1.5
        assert cfg.model.rgbwarp_correction.alpha_points_per_pixel==8
        assert not cfg.model.deep_lora.enabled


def test_online_pose_uses_exact_dataset_conversion():
    first=[[100.,200.,300.,15.,179.,-10.]]
    now=[[110.,210.,350.,5.,-179.,3.]]
    result=canonical_yaw4d_state(torch.zeros(1,1,3,2,2),['Turn'],None,first,now)
    torch.testing.assert_close(result[0],torch.tensor(episode_yaw4d_state(first[0],now[0])))
    with pytest.raises(ValueError,match='either'):
        canonical_yaw4d_state(torch.zeros(1,1,3,2,2),['Turn'],torch.zeros(1,4),first,now)
    with pytest.raises(ValueError,match='finite'):
        canonical_yaw4d_state(torch.zeros(1,1,3,2,2),['Turn'],[[float('nan')]*4],None,None)


@pytest.mark.parametrize('weights', [(1./3., 1./3., 1./3.), (1., 0., 1.)])
def test_rgbwarp_startup_accepts_current_and_legacy_depth_recipes(weights, tmp_path, monkeypatch):
    import sys
    from omegaconf import OmegaConf
    from experiments.uavflow_predictor_idm import train

    cfg = build_config('rgbwarp')
    cfg.loss.depth_linear_weight, cfg.loss.depth_log_weight, cfg.loss.depth_grad_weight = weights
    path = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, path)
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path)])
    monkeypatch.setattr(train, 'distributed_info', lambda: (False, 0, 0, 1))

    class PassedValidation(Exception):
        pass

    def reached_accelerator(**kwargs):
        raise PassedValidation

    monkeypatch.setattr(train, 'configure_accelerator', reached_accelerator)
    with pytest.raises(PassedValidation):
        train.main()
