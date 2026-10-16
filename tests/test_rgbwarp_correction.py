import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from robot.modeling.da3_giant_encoder import DA3GiantEncoder
from robot.modeling.lora import LoRALinear
from experiments.uavflow_predictor_idm.rgbwarp_geometry import forward_splat,action_camera_transform
from experiments.uavflow_predictor_idm.rgbwarp_correction import CorrespondenceRead
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from test_provenance_correction import Block,Transformer


@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(42)
    yield
    torch.set_num_threads(n)


def camera(size=28):
    return torch.tensor([[size/2,0,(size-1)/2],[0,size/2,(size-1)/2],[0,0,1.]])


def test_identity_and_correspondence_direction():
    rgb=torch.rand(1,3,28,28);depth=torch.ones(1,28,28)*3
    out=forward_splat(rgb,depth,torch.eye(4)[None],camera())
    torch.testing.assert_close(out['rgb'],rgb,atol=5e-6,rtol=1e-5)
    assert out['valid'].all()
    assert torch.equal(out['correspondence'][0],torch.eye(4,dtype=torch.bool))
    # Camera moves right (agent +y), so source pixels move LEFT in future.
    actions=torch.zeros(1,5,4);actions[0,-1,1]=3/14
    moved=forward_splat(rgb,depth,action_camera_transform(actions),camera())
    assert not moved['valid'][0,0,:,-1].any()
    assert moved['correspondence'][0,0,1]  # Future-left reads Current-right


def test_warp_nonzero_action_depth_gradients_and_invalid_depth():
    yy,xx=torch.meshgrid(torch.arange(28),torch.arange(28),indexing='ij')
    rgb=torch.stack((xx/28,yy/28,(xx+yy)/56))[None]
    depth=torch.full((1,28,28),3.,requires_grad=True)
    actions=torch.zeros(1,5,4,requires_grad=True)
    values=actions+torch.tensor([.01,.02,.0,.002])
    out=forward_splat(rgb,depth,action_camera_transform(values),camera())
    loss=(out['rgb']*torch.rand_like(out['rgb'])).sum()
    gd,ga=torch.autograd.grad(loss,(depth,actions))
    assert torch.isfinite(gd).all() and torch.isfinite(ga).all()
    assert gd.abs().sum()>0 and ga.abs().sum()>0
    invalid=forward_splat(rgb,torch.zeros_like(depth),torch.eye(4)[None],camera())
    assert not invalid['valid'].any() and not invalid['correspondence'].any()
    assert torch.isfinite(invalid['rgb']).all() and invalid['rgb'].count_nonzero()==0


def test_zbuffer_nearest_surface_wins():
    # Translate so two pixels on different surfaces compete near one target.
    rgb=torch.rand(1,3,28,28);depth=torch.ones(1,28,28)*5
    depth[:,:,14:]=2
    action=torch.zeros(1,5,4);action[0,-1,1]=.4
    out=forward_splat(rgb,depth,action_camera_transform(action),camera())
    assert torch.isfinite(out['rgb']).all()
    assert (out['depth'][out['valid'][:,0]]>0).all()


def test_correspondence_empty_rows_and_initial_identity():
    read=CorrespondenceRead(16,[3],width=16,heads=4)
    q,k=torch.rand(2,8,16),torch.rand(2,8,16)
    mask=torch.zeros(2,8,8,dtype=torch.bool);mask[:,0,2]=True
    torch.testing.assert_close(read(q,k,mask,3),q)
    with torch.no_grad():read.output.weight.normal_(0,.1)
    out=read(q,k,mask,3)
    torch.testing.assert_close(out[:,1:],q[:,1:]);assert torch.isfinite(out).all()
    assert not torch.allclose(out[:,0],q[:,0])


class FrontTransformer(Transformer):
    def __init__(self):
        super().__init__();self.embed_dim=16;self.num_register_tokens=2
        self.patch_embed=nn.Conv2d(3,16,14,stride=14)
        self.register_tokens=nn.Parameter(torch.zeros(1,2,16))
        self.cls_token=nn.Parameter(torch.zeros(1,1,16))

    def prepare_tokens_with_masks(self,images):
        b,v,c,h,w=images.shape
        patches=self.patch_embed(images.flatten(0,1)).flatten(2).transpose(1,2)
        x=torch.cat((self.cls_token.expand(b*v,-1,-1),self.register_tokens.expand(b*v,-1,-1),patches),1)
        return x.reshape(b,v,-1,16)


class TinyDPT(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(32,1);self.counts=[]
    def forward(self,features,h,w,patch_start_idx,**kwargs):
        patches=features[-1][0];self.counts.append(patches.shape[-2])
        assert patches.shape[-2]==256  # Camera/register/FIVE actions excluded.
        value=F.softplus(self.proj(patches)[...,0])*.03+.02
        b,v,n=value.shape
        return {'depth':F.interpolate(value.reshape(b*v,1,16,16),size=(h,w),mode='bilinear',align_corners=False).reshape(b,v,h,w)}


class TinyDA3(DA3GiantEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.embed_dim=16;self.hidden_size=32;self.num_register_tokens=2;self.h_patches=16
        self.shallow_target_layer=0;self.temporal_embed=None;self.out_layers=[3,5,7,9]
        self.backbone=nn.Module();self.backbone.pretrained=FrontTransformer();self.dpt_head=TinyDPT()
        self.register_buffer('encoder_mean',torch.zeros(1,3,1,1));self.register_buffer('encoder_std',torch.ones(1,3,1,1))


def make_model(corr=True,mask=True):
    da3=TinyDA3()
    da3.configure_deep_lora(0,rank=2,alpha=4,dropout=0,target_modules=['all-linear'])
    return UAVFlowPredictorIDM(da3=da3,idm=None,action_dim=4,action_chunk_size=5,
        d_model=32,depth=1,num_heads=4,language_dim=16,language_len=8,
        direct_action_enabled=True,deep_action_enabled=True,compute_idm_branch=False,
        geometry_architecture='dual_vla_gfm',parallel_vla_gfm_enabled=True,
        parallel_vla_gfm_mode='qwen_tokens',parallel_action_decode_mode='geometry_residual',
        parallel_action_position_mode='decoder_only',parallel_vla_gfm_width=32,parallel_vla_gfm_heads=4,
        action_residual_gate_init=.1,
        parallel_current_geometry_read_enabled=False,parallel_current_depth_enabled=True,
        depth_decode_enabled=True,gradient_checkpointing=True,train_deep_backbone=True,deep_train_start_block=0,
        dual_stream_rgbwarp_enabled=True,rgbwarp_correction={'correspondence_layers':[3,5,7,9],
            'correspondence_width':16,'correspondence_heads':4,'use_correspondence_attention':corr,
            'use_pixel_binary_mask':mask})


def run(model):
    rgb=torch.rand(1,3,224,224);slots=torch.rand(1,5,16,requires_grad=True)
    result=model(rgb,reference_shallow=None,lang_feats=slots,lang_padding_mask=None,
                 action_denormalizer=lambda x:x*.05)
    return result,slots


@pytest.mark.parametrize('corr,mask',[(True,True),(False,True),(True,False)])
def test_full_two_stream_shared_lora_frozen_dpt_and_gradients(corr,mask):
    net=make_model(corr,mask).train()
    result,slots=run(net)
    assert isinstance(net.predictor,nn.Identity)
    assert not list(net.predictor.parameters())
    assert result['actions_norm'].shape==(1,1,5,4)
    torch.testing.assert_close(result['actions_norm'],result['direct_actions_norm'])
    assert net.da3.dpt_head.counts==[256,256]
    assert all(not p.requires_grad for p in net.da3.dpt_head.parameters())
    assert len([m for m in net.da3.modules() if isinstance(m,LoRALinear)])==40
    assert len(list(net.parameters()))==len({id(p) for p in net.parameters()})
    for branch in ['current_depth_output','deep_joint_features']:
        net.zero_grad(set_to_none=True)
        loss=result[branch]['depth'].square().mean()
        loss.backward(retain_graph=True)
        assert all(p.grad is None for p in net.da3.dpt_head.parameters())
        assert net.da3.backbone.pretrained.blocks[0].attn.qkv.lora_B.grad.abs().sum()>0
        assert net.da3.backbone.pretrained.blocks[-1].attn.qkv.lora_B.grad.abs().sum()>0
    # Fixed future depth, NO action loss: verify proposal numeric warp path.
    proposal=result['direct_actions_norm'];proposal.retain_grad()
    result['deep_joint_features']['depth'].square().mean().backward()
    assert proposal.grad is not None and proposal.grad.abs().sum()>0
    assert slots.grad is not None and torch.isfinite(slots.grad).all()


def test_mask_zero_init_and_pixel_pattern_not_only_coverage():
    net=make_model();da3=net.da3;projection=net.rgbwarp_correction.mask_projection
    image=torch.rand(1,3,224,224)
    a=da3.encode_shallow_rgb_grad(image,gradient_checkpointing=False)
    b=da3.encode_shallow_rgb_grad(image,invalid_mask=torch.ones(1,1,224,224),mask_projection=projection,gradient_checkpointing=False)
    torch.testing.assert_close(a,b)
    with torch.no_grad():projection.weight.normal_()
    left=torch.zeros(1,1,224,224);right=left.clone()
    left[:,:,:,:7]=1;right[:,:,:,7:14]=1
    assert left.sum()==right.sum()
    assert not torch.allclose(projection(left),projection(right))


def test_pose_helper_same_units_wrap_and_roll_pitch_contract():
    from robot.data.uavflow_dataset import episode_yaw4d_state
    first=np.array([100.,200.,300.,15.,179.,-10.])
    now=np.array([110.,210.,350.,5.,-179.,3.])
    state=episode_yaw4d_state(first,now)
    assert abs(state[-1]-2)<1e-5 and state[2]==50
    yaw=np.deg2rad(first[4]);world=now[:3]-first[:3]
    expected=[np.cos(yaw)*world[0]+np.sin(yaw)*world[1],-np.sin(yaw)*world[0]+np.cos(yaw)*world[1],50.,2.]
    np.testing.assert_allclose(state,expected,atol=1e-5)
    np.testing.assert_array_equal(episode_yaw4d_state(first,first),np.zeros(4))


def test_future_depth_gradient_through_warp_not_action_value_shortcut():
    net=make_model(corr=False)
    with torch.no_grad():
        net.rgbwarp_correction.action_values.weight.zero_()
    result,_=run(net)
    proposal=result['direct_actions_norm']
    current=result['current_depth_output']['depth']
    loss=result['deep_joint_features']['depth'].square().mean()
    gp,gc=torch.autograd.grad(loss,(proposal,current),retain_graph=True)
    assert gp.abs().sum()>0 and gc.abs().sum()>0
    assert torch.isfinite(gp).all() and torch.isfinite(gc).all()
    (loss+result['actions_norm'].square().mean()+current.square().mean()).backward()
    assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]


def test_fixed_target_objective_and_eval_proposal_final_metrics(monkeypatch):
    from robot.data.dataset import ActionNormalizer
    from experiments.uavflow_predictor_idm import rgbwarp_objectives as obj
    from experiments.uavflow_predictor_idm.objectives import evaluate
    net=make_model(corr=False)
    normalizer=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    slots=torch.rand(1,5,16)
    monkeypatch.setattr(obj,'encode_stage2_condition',lambda *a,**k:dict(last_hidden_state=slots,attention_mask=None))
    batch=dict(all_view_images=torch.rand(1,2,1,3,224,224),openvla_prompt_pose=torch.zeros(1,2,4),
        action_stats_key=['sim'],task_description=['Pass the car on the right'],
        actions=torch.rand(1,2,5,4)*.05,action_loss_mask=torch.ones(1,2,5,4,dtype=torch.bool),
        gt_depth_meters=torch.full((1,2,1,224,224),3.),
        gt_depth_mask=torch.ones(1,2,1,224,224,dtype=torch.bool),start_t=torch.tensor([1]))
    args=dict(model=net,da3=net.da3,text=None,normalizer=normalizer,batch=batch,
              context_len=1,rollout_steps=1,amp=False,amp_dtype='fp32')
    values=obj.forward_rgbwarp_batch(**args)
    torch.testing.assert_close(values['action'],values['direct_action']+values['refine_action'])
    torch.testing.assert_close(values['depth'],values['depth_current_total']+values['depth_future_total'])
    # Future observation and GT actions are not inputs to the forward policy.
    output=net(batch['all_view_images'][:,0,0],reference_shallow=None,lang_feats=slots,
               lang_padding_mask=None,action_denormalizer=lambda x:normalizer.denormalize(x,stats_keys=['sim']))
    expected=obj.depth_loss(output['deep_joint_features']['depth'].reshape(1,224,224),
                           batch['gt_depth_meters'][:,1,0],batch['gt_depth_mask'][:,1,0],100.)
    torch.testing.assert_close(values['depth_future_total'],expected['total'])
    batch['all_view_images'][:,1].zero_()
    changed=obj.forward_rgbwarp_batch(**args)
    torch.testing.assert_close(values['depth'],changed['depth'])
    torch.testing.assert_close(values['action'],changed['action'])
    metrics=evaluate(model=net,da3=net.da3,text=None,normalizer=normalizer,pose_normalizer=None,
        loader=[batch],contexts=[1],rollout=1,device=torch.device('cpu'),amp=False,
        horizon_weights=[1.],feature_patch_weight=1.,feature_cls_weight=0.,feature_register_weight=0.,
        deep_feature_enabled=False,deep_feature_patch_weight=1.,deep_feature_cls_weight=0.,stop_pos_weight=5.,
        max_batches=1,amp_dtype='fp32',depth_target_source='ue_gt',depth_target_mode='both')
    assert metrics['H1_direct_action']==pytest.approx(values['direct_action'].item())
    assert metrics['H1_refine_action']==pytest.approx(values['refine_action'].item())


def test_gate_checkpoint_and_optimizer_roundtrip():
    from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state,load_architecture_state
    net=make_model(corr=False)
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4)
    result,_=run(net)
    (result['actions_norm'].square().mean()+result['current_depth_output']['depth'].square().mean()
        +result['deep_joint_features']['depth'].square().mean()).backward()
    optimizer.step()
    saved={'geometry_architecture_state':copy.deepcopy(architecture_state(net))}
    restored=make_model(corr=False)
    load_architecture_state(restored,saved)
    for a,b in zip(net.rgbwarp_correction.parameters(),restored.rgbwarp_correction.parameters()):
        torch.testing.assert_close(a,b)
    torch.testing.assert_close(net.action_residual_gate_logit,restored.action_residual_gate_logit)
    clone=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4)
    clone.load_state_dict(optimizer.state_dict())
    assert len(clone.state)==len(optimizer.state)
    incompatible=copy.deepcopy(saved)
    incompatible['geometry_architecture_state'].pop('action_residual_gate_logit')
    with pytest.raises(ValueError,match='gating changed'):
        load_architecture_state(restored,incompatible)


def test_entrypoint_explicit_defaults():
    from experiments.uavflow_predictor_idm.rgbwarp_train import build_config
    cfg=build_config()
    assert cfg.model.dual_stream_rgbwarp_enabled and not cfg.model.support_aware_transport
    assert not cfg.model.use_fixed_first_frame and not cfg.model.stop_head_enabled
    assert cfg.stage1.qwen_numeric_pose_token
    assert cfg.model.da3_lora.target_modules==['all-linear']
    assert cfg.loss.action_weight==3 and cfg.loss.depth_weight==1 and cfg.loss.feature_weight==0
    assert cfg.training.max_epochs==5


def test_numeric_state_hook_and_observation_only_service(monkeypatch):
    from experiments.uavflow_predictor_idm.vlm_conditioning import FrozenQwen35Conditioner
    from experiments.uavflow_predictor_idm import vlm_conditioning
    from experiments.uavflow_support_aware.inference import RGBWarpPolicy
    from robot.data.dataset import ActionNormalizer

    class Encoder(nn.Module):
        prompt_mode='current_image_action_question'
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(8,16).requires_grad_(False)
            self.qwen=SimpleNamespace(get_input_embeddings=lambda:self.embedding)
        def forward(self,images,texts,*,omit_prompt_pose,state_placeholder):
            assert omit_prompt_pose and state_placeholder
            value=self.embedding(torch.tensor([[7,1,2,3,4,5]]))
            return dict(joint_layers=(value[:,1:]+value[:,:1])[:,None],joint_mask=torch.ones(1,5,dtype=torch.bool))

    conditioner=FrozenQwen35Conditioner.__new__(FrozenQwen35Conditioner);nn.Module.__init__(conditioner)
    conditioner.encoder=Encoder();conditioner.numeric_pose_token=True;conditioner.state_token_id=7
    conditioner.register_buffer('state_scale',torch.tensor([1000.,1000.,1000.,180.]))
    conditioner.state_encoder=nn.Sequential(nn.Linear(4,256),nn.GELU(),nn.Linear(256,16))
    a=conditioner.encode_tokens(['Turn'],images=torch.rand(1,2,1,3,224,224),current_pose=torch.zeros(1,4))
    b=conditioner.encode_tokens(['Turn'],images=torch.rand(1,2,1,3,224,224),current_pose=torch.tensor([[100.,0.,0.,45.]]))
    assert not torch.allclose(a['last_hidden_state'],b['last_hidden_state'])
    b['last_hidden_state'].sum().backward()
    assert conditioner.state_encoder[0].weight.grad.abs().sum()>0
    assert not conditioner.encoder.embedding._forward_hooks

    net=make_model(corr=False).eval();slots=torch.rand(1,5,16)
    monkeypatch.setattr(vlm_conditioning,'encode_stage2_condition',lambda *a,**k:dict(last_hidden_state=slots,attention_mask=None))
    norm=ActionNormalizer({'sim':dict(q01=[-.1]*4,q99=[.1]*4,mask=[True]*4)},default_key='sim')
    policy=RGBWarpPolicy(net,conditioner,norm,amp=False)
    images=torch.rand(1,1,3,224,224)
    expected=net(images[:,0],reference_shallow=None,lang_feats=slots,lang_padding_mask=None,
                 action_denormalizer=lambda x:norm.denormalize(x))
    result=policy.predict(images,['Turn'],first_sim_pose=[[0.]*6],current_sim_pose=[[0.,0.,0.,0.,30.,0.]])
    torch.testing.assert_close(result['actions_meters_radians'],norm.denormalize(expected['actions_norm'])[:,0])
    with pytest.raises(ValueError,match=r'\[B,6\]'):
        policy.predict(images,['Turn'],first_sim_pose=[],current_sim_pose=[[0.]*6])


def test_gated_residual_has_no_tanh_or_fixed_bound_and_trains_gate():
    net=make_model(corr=False)
    assert net.action_residual_gate_logit.sigmoid().item()==pytest.approx(.1)
    with torch.no_grad():net.parallel_action_correction_head.decoder.output.bias.fill_(20.)
    output,_=run(net)
    torch.testing.assert_close(output['residual'],torch.full_like(output['residual'],2.))
    torch.testing.assert_close(output['actions_norm'],output['direct_actions_norm']+output['residual'])
    output['residual'].sum().backward()
    assert net.action_residual_gate_logit.grad.abs()>0
    assert net.parallel_action_correction_head.decoder.output.bias.grad.abs().sum()>0
    assert not any('residual_max_abs' in k for k in net.rgbwarp_correction.state_dict())
    assert not net.residual_gate_logit.requires_grad  # old feature gate stays unused


def test_invalid_nan_target_has_no_gradient_and_gam_axis_mean_sum():
    from experiments.uavflow_predictor_idm.rgbwarp_objectives import depth_loss
    pred=torch.tensor([[[0.,1.],[0.,1.]]],requires_grad=True)
    target=torch.zeros_like(pred)+100.
    valid=torch.ones_like(pred,dtype=torch.bool)
    values=depth_loss(pred,target,valid,100.)
    assert values['grad'].item()==pytest.approx(1.)  # x mean + y mean, not joint mean
    target[:]=float('nan')
    loss=depth_loss(pred,target,valid,100.)['total']
    assert torch.isfinite(loss) and loss.item()==0
    loss.backward();assert pred.grad.count_nonzero()==0
