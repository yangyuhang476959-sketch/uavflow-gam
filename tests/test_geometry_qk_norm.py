import copy
import torch
from torch import nn
from torch.nn import functional as F
import pytest

from experiments.uavflow_predictor_idm.qk_normalized_attention import QKNormalizedAttention
from experiments.uavflow_predictor_idm.current_geometry import CurrentGeometryRead
from experiments.uavflow_predictor_idm.rgbwarp_correction import CorrespondenceRead
from experiments.uavflow_predictor_idm.metric_train import build_config, preserve_resume_qk_norm

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_legacy_weights_outputs_and_gradients_are_unchanged():
    torch.manual_seed(1)
    old=nn.MultiheadAttention(8,2,batch_first=True)
    new=QKNormalizedAttention(8,2,qk_norm=False)
    new.load_state_dict(old.state_dict(),strict=True)
    assert set(old.state_dict())==set(new.state_dict())
    q=torch.randn(2,3,8,requires_grad=True);k=torch.randn(2,5,8,requires_grad=True)
    mask=torch.zeros(4,3,5,dtype=torch.bool);mask[:,:,-1]=True
    a=old(q,k,k,attn_mask=mask,need_weights=False)[0]
    b=new(q,k,k,attn_mask=mask,need_weights=False)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga=torch.autograd.grad(a.square().sum(),(q,k),retain_graph=True)
    gb=torch.autograd.grad(b.square().sum(),(q,k))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_post_projection_per_head_norm_and_mask_match_explicit_reference():
    torch.manual_seed(2)
    a=QKNormalizedAttention(8,2,qk_norm=True)
    q=torch.randn(2,3,8);k=torch.randn(2,5,8);v=torch.randn(2,5,8)
    mask=torch.zeros(4,3,5,dtype=torch.bool);mask[:,:,-2:]=True
    out=a(q,k,v,attn_mask=mask)[0]
    projected=[F.linear(x,w,b).reshape(2,-1,2,4).transpose(1,2)
               for x,w,b in zip((q,k,v),a.in_proj_weight.chunk(3),a.in_proj_bias.chunk(3))]
    qp,kp,vp=projected;qp=a.q_norm(qp);kp=a.k_norm(kp)
    scores=qp@kp.transpose(-1,-2)/2
    weights=scores.masked_fill(mask.reshape(2,2,3,5),float('-inf')).softmax(-1)
    expected=F.linear((weights@vp).transpose(1,2).reshape(2,3,8),a.out_proj.weight,a.out_proj.bias)
    torch.testing.assert_close(out,expected)
    changed=v.clone();changed[:,-2:]=1e6
    torch.testing.assert_close(out,a(q,k,changed,attn_mask=mask)[0],rtol=0,atol=0)


@pytest.mark.parametrize('kind',['bank','warp'])
def test_large_inputs_finite_backward_and_norm_parameters_receive_gradients(kind):
    torch.manual_seed(3)
    if kind=='bank':
        m=CurrentGeometryRead(8,width=8,heads=2,memory_dim=8,qk_norm=True)
        q=(torch.randn(2,1,1,5,8)*1e4).requires_grad_();k=(torch.randn(2,1,1,7,8)*1e4).requires_grad_()
        out=m(q,k)
    else:
        m=CorrespondenceRead(8,[19],width=8,heads=2,qk_norm=True)
        nn.init.normal_(m.output.weight,std=.1)
        q=(torch.randn(2,5,8)*1e4).requires_grad_();k=(torch.randn(2,7,8)*1e4).requires_grad_()
        mask=torch.ones(2,5,7,dtype=torch.bool);mask[:,0]=False
        out=m(q,k,mask,19)
        torch.testing.assert_close(out[:,0],q[:,0],rtol=0,atol=0)
    out.square().mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    for norm in (m.attention.q_norm,m.attention.k_norm):
        assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
        assert norm.weight.grad.abs().sum()>0


@pytest.mark.parametrize('variant',['rgbwarp','rgbwarp_no_action','current_bank'])
def test_new_defaults_on_old_resume_preserved_and_explicit_choice_not_overwritten(tmp_path,variant):
    cfg=build_config(variant)
    key='qk_norm' if variant.startswith('rgbwarp') else 'current_geometry_qk_norm'
    model=cfg.model.rgbwarp_correction if variant.startswith('rgbwarp') else cfg.model
    assert model[key] is True
    path=tmp_path/'old.pt';torch.save({'config':{'model':{}}},path)
    restored=preserve_resume_qk_norm(copy.deepcopy(cfg),path)
    branch=restored.model.rgbwarp_correction if variant.startswith('rgbwarp') else restored.model
    assert branch[key] is False
    prefix='model.rgbwarp_correction.' if variant.startswith('rgbwarp') else 'model.'
    restored=preserve_resume_qk_norm(copy.deepcopy(cfg),path,[prefix+key+'=true'])
    branch=restored.model.rgbwarp_correction if variant.startswith('rgbwarp') else restored.model
    assert branch[key] is True
    saved_model={'rgbwarp_correction':{'qk_norm':True}} if variant.startswith('rgbwarp') else {'current_geometry_qk_norm':True}
    torch.save({'config':{'model':saved_model}},path)
    restored=preserve_resume_qk_norm(copy.deepcopy(cfg),path)
    branch=restored.model.rgbwarp_correction if variant.startswith('rgbwarp') else restored.model
    assert branch[key] is True
