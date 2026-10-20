from experiments.uavflow_predictor_idm.metric_train import build_config, VARIANTS
from experiments.uavflow_qwen_pure.train import _cfg as pure_qwen_config


def test_four_metric_handoff_jobs_use_constant_lr_and_have_correct_action_flow():
    assert set(VARIANTS)=={'gam','current_bank','rgbwarp','rgbwarp_no_action'}
    c3=build_config('rgbwarp')
    c4=build_config('rgbwarp_no_action')
    assert c3.training.lr_schedule=='constant' and c3.training.warmup_steps==0
    assert c4.training.lr_schedule=='constant' and c4.training.warmup_steps==0
    assert c3.model.rgbwarp_correction.get('inject_proposal_action_tokens',True)
    assert not c4.model.rgbwarp_correction.inject_proposal_action_tokens
    assert c3.dataset.visual_anchor_stride==5 and c4.dataset.chunk_size==5


def test_pure_qwen_handoff_is_yaw4d_pose_token_and_constant_lr():
    cfg=pure_qwen_config()
    assert cfg.dataset.openvla_prompt_pose_mode=='evaluator_yaw4d'
    assert cfg.model.pose_mode=='numeric_token'
    assert cfg.training.lr_schedule=='constant' and cfg.training.warmup_steps==0
    assert cfg.training.max_epochs==10 and cfg.training.batch_size==4
