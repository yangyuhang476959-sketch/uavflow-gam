"""Equal linear/log/GAM-gradient depth terms for the remote release."""
from types import SimpleNamespace

import pytest
import torch

from experiments.uavflow_predictor_idm import rgbwarp_objectives as rgb
from experiments.uavflow_predictor_idm.objectives import forward_batch, gam_window_pointnorm_depth_loss


def test_log_recipe_matches_existing_gam_and_keeps_legacy_no_log():
    pred=torch.tensor([[[.2,.4],[.3,.6]]],requires_grad=True)
    target=torch.tensor([[[10.,30.],[40.,50.]]])
    valid=torch.ones_like(target,dtype=torch.bool)
    old=rgb.depth_loss(pred,target,valid,100.)
    new=rgb.depth_loss(pred,target,valid,100.,log_weight=1.)
    torch.testing.assert_close(old['total'],old['l1']+old['grad'])
    torch.testing.assert_close(new['total'],new['l1']+new['log_l1']+new['grad'])
    torch.testing.assert_close(new['log_l1'],(pred.log()-(target/100).log()).abs().mean())
    gam=gam_window_pointnorm_depth_loss(pred,target,valid,target,valid,
        linear_weight=1.,log_weight=1.,grad_weight=1.,gradient_mode='gam_l1',
        fixed_scale_meters=100.,semantic_weight=1.)
    torch.testing.assert_close(new['total'],gam['total'])
    torch.testing.assert_close(torch.autograd.grad(new['total'],pred,retain_graph=True)[0],
                               torch.autograd.grad(gam['total'],pred)[0])


def test_log_invalid_targets_and_empty_mask_are_finite():
    pred=torch.ones(1,2,2,requires_grad=True)
    target=torch.tensor([[[100.,float('nan')],[0.,float('inf')]]])
    result=rgb.depth_loss(pred,target,torch.ones_like(pred,dtype=torch.bool),100.,log_weight=1.)
    assert result['total'].item()==0
    result['total'].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.count_nonzero()==0
    result=rgb.depth_loss(pred,target,torch.zeros_like(pred,dtype=torch.bool),100.,log_weight=1.)
    assert result['total'].item()==0
    with pytest.raises(ValueError,match='positive'):
        rgb.depth_loss(pred,target,torch.ones_like(pred,dtype=torch.bool),100.,log_epsilon_normalized=0.)


def test_trainer_dispatch_preserves_explicit_log_weight(monkeypatch):
    monkeypatch.setattr(rgb,'forward_rgbwarp_batch',lambda **kw:kw)
    for weight in [0.,1.]:
        result=forward_batch(model=SimpleNamespace(dual_stream_rgbwarp_enabled=True),
            da3=None,text=None,normalizer=None,pose_normalizer=None,batch={},context_len=1,rollout_steps=1,
            feature_horizon_weights=[1.],feature_patch_weight=1.,feature_cls_weight=0.,
            feature_register_weight=0.,deep_feature_enabled=False,deep_feature_patch_weight=1.,
            deep_feature_cls_weight=0.,stop_pos_weight=1.,amp=False,
            depth_log_weight=weight,depth_log_epsilon_normalized=1e-5)
        assert result['depth_log_weight']==weight
        assert result['depth_log_epsilon_normalized']==1e-5


def test_current_future_log_terms_are_both_used_and_logged(monkeypatch):
    from test_rgbwarp_correction import make_model
    from robot.data.dataset import ActionNormalizer
    net=make_model(corr=False)
    normalizer=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    slots=torch.rand(1,5,16)
    monkeypatch.setattr(rgb,'encode_stage2_condition',lambda *a,**k:dict(last_hidden_state=slots,attention_mask=None))
    batch=dict(all_view_images=torch.rand(1,2,1,3,224,224),openvla_prompt_pose=torch.zeros(1,2,4),
        action_stats_key=['sim'],task_description=['Approach the car'],
        actions=torch.zeros(1,2,5,4),action_loss_mask=torch.ones(1,2,5,4,dtype=torch.bool),
        gt_depth_meters=torch.full((1,2,1,224,224),3.),
        gt_depth_mask=torch.ones(1,2,1,224,224,dtype=torch.bool))
    args=dict(model=net,da3=net.da3,text=None,normalizer=normalizer,batch=batch,
              context_len=1,rollout_steps=1,amp=False,amp_dtype='fp32')
    new=rgb.forward_rgbwarp_batch(**args,depth_log_weight=1.)
    old=rgb.forward_rgbwarp_batch(**args,depth_log_weight=0.)
    assert new['depth_log_l1']>0
    for branch in ['current','future']:
        torch.testing.assert_close(new[f'depth_{branch}_total'],
            new[f'depth_{branch}_linear_l1']+new[f'depth_{branch}_log_l1']+new[f'depth_{branch}_grad'])
    torch.testing.assert_close(new['depth']-old['depth'],new['depth_log_l1'])
    torch.testing.assert_close(3*new['depth'],1.5*(new['depth_current_total']+new['depth_future_total']))
