import torch
import pytest
from test_native_gam_path import make_native

@pytest.fixture(autouse=True)
def single_thread():
 old=torch.get_num_threads();torch.set_num_threads(1)
 yield
 torch.set_num_threads(old)

def test_vla_uses_qwen_seed_no_text_ca_no_pose_sa_no_current_decode(monkeypatch):
 net=make_native('vla_gam',vlm_pose_only=True)
 def forbidden(*a,**k):raise AssertionError('Extra Pose SA, text CA, Current decode or Proposal head called')
 pred=net.rgbwarp_correction.future_predictor
 monkeypatch.setattr(pred,'_embed_proprio_history',forbidden)
 monkeypatch.setattr(net.da3,'propagate_shallow_visual_slots_grad',forbidden)
 monkeypatch.setattr(net.parallel_action_head,'forward',forbidden)
 for block in pred.blocks:monkeypatch.setattr(block.cross_attn,'forward',forbidden)
 assert not pred.use_proprio_input
 assert not any(p.requires_grad for p in pred.proprio_token_proj.parameters())
 seen={};original=pred.forward
 def capture(*args,**kw):seen.update(kw);return original(*args,**kw)
 monkeypatch.setattr(pred,'forward',capture)
 qwen=torch.rand(2,5,16,requires_grad=True)
 out=net(torch.rand(2,3,224,224),reference_shallow=None,lang_feats=qwen,lang_padding_mask=None,gam_pose_embedding=None,action_denormalizer=forbidden)
 assert seen['lang_feats'] is None and seen['proprio_history'] is None
 assert seen['action_slot_seed'].shape==(2,5,32)
 assert out['direct_actions_norm'] is None and out['current_depth_output'] is None
 assert net.da3.dpt_head.counts==[256]
 (out['actions_norm'].square().mean()+out['predicted_shallow'].square().mean()+out['deep_joint_features']['depth'].square().mean()).backward()
 assert qwen.grad.abs().sum()>0
 assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]

def test_shared_predictor_weights_keep_uav_initial_values():
 torch.manual_seed(42);uav=make_native()
 torch.manual_seed(42);vla=make_native('vla_gam',vlm_pose_only=True)
 a=uav.rgbwarp_correction.future_predictor;b=vla.rgbwarp_correction.future_predictor
 for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)
 assert a.use_proprio_input and not b.use_proprio_input
 assert not b.lang_proj.weight.requires_grad

def test_config_has_vlm_pose_and_future_only_supervision():
 from experiments.uavflow_paper_ablation.gam_train import build_config
 cfg=build_config('vla_gam')
 assert cfg.model.rgbwarp_correction.vla_pose_in_vlm_only
 assert cfg.stage1.qwen_numeric_pose_token and cfg.stage1.qwen_pose5_enabled
 assert not cfg.stage1.get('gam_t5_language',False)
 assert cfg.loss.depth_target_mode=='future' and cfg.loss.action_direct_weight==0
 assert cfg.loss.feature_weight==1 and cfg.loss.action_weight==3 and cfg.loss.depth_weight==3


def test_corrected_gam_slot_embeddings_can_be_matched_without_changing_other_settings():
 from omegaconf import OmegaConf
 from experiments.uavflow_paper_ablation.gam_train import build_config

 original=build_config('vla_gam')
 matched=build_config('vla_gam',match_gam_slot_embeddings=True)
 correction=matched.model.rgbwarp_correction
 assert correction.pop('gam_action_slot_position_encoding') is True
 assert correction.pop('gam_action_slot_init_seed')==48
 assert OmegaConf.to_container(matched,resolve=True)==OmegaConf.to_container(original,resolve=True)

 torch.manual_seed(42)
 uav=make_native(slot_positions=True)
 torch.manual_seed(42)
 vla=make_native('vla_gam',vlm_pose_only=True,slot_positions=True)
 torch.testing.assert_close(
  vla.rgbwarp_correction.future_predictor.action_slot_embed,
  uav.rgbwarp_correction.future_predictor.action_slot_embed,rtol=0,atol=0,
 )
