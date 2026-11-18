import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from test_rgbwarp_correction import make_model,TinyDA3
from experiments.uavflow_paper_ablation.gam import MatchedGAMCorrection
from experiments.uavflow_paper_ablation.gam_train import build_config
from experiments.uavflow_paper_ablation.initialization import controller_head_parameter_ids
from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state,load_architecture_state


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def make_gam(mode):
    return make_model(corr=False,mask=False,output_mode='direct',branch_mode=mode)


def call(model,language=None,mask=None):
    mode=model.rgbwarp_correction.mode
    if language is None:language=torch.rand(2,7 if mode=='uav_gam' else 5,16,requires_grad=True)
    return model(torch.rand(2,3,224,224),reference_shallow=None,lang_feats=language,
                 lang_padding_mask=mask,gam_pose_embedding=torch.rand(2,16),
                 action_denormalizer=lambda x:x),language


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_native_predictor_direct_action_depth_and_complete_gradient_paths(mode,monkeypatch):
    import experiments.uavflow_predictor_idm.rgbwarp_correction as warp
    def forbidden(*a,**k):raise AssertionError('GAM called RGB Warp')
    monkeypatch.setattr(warp,'forward_splat',forbidden)
    monkeypatch.setattr(warp,'forward_alpha_splat',forbidden)
    net=make_gam(mode)
    assert isinstance(net.rgbwarp_correction,MatchedGAMCorrection)
    assert net.rgbwarp_correction.correspondence is None
    assert net.rgbwarp_correction.mask_projection is None
    assert net.action_residual_gate_logit is None
    out,language=call(net)
    assert out['actions_norm'].shape==(2,1,5,4)
    assert out['predicted_shallow'].shape==(2,1,1,259,16)
    assert net.da3.dpt_head.counts==[256,256]
    assert out['warp'] is None
    # This is a complete action head, regardless of the pre-geometry output.
    with torch.no_grad():
        net.parallel_action_correction_head.decoder.output.weight.zero_()
        net.parallel_action_correction_head.decoder.output.bias.fill_(.25)
    direct,_=call(net)
    torch.testing.assert_close(direct['actions_norm'],torch.full_like(direct['actions_norm'],.25))
    # Recompute after the intentional head edit before backward.
    out,language=call(net)
    loss=(out['actions_norm'].square().mean()+out['direct_actions_norm'].square().mean()
          +out['current_depth_output']['depth'].square().mean()+out['deep_joint_features']['depth'].square().mean())
    loss.backward()
    assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert all(p.grad is None for p in net.da3.dpt_head.parameters())
    assert net.rgbwarp_correction.future_predictor.future_visual_proj.weight.grad.abs().sum()>0


def test_common_heads_and_predictor_initialization_are_identical():
    torch.manual_seed(42);warp=make_model(corr=False,mask=True,output_mode='direct')
    controls=[]
    for mode in ['uav_gam','vla_gam']:
        torch.manual_seed(42);gam=make_gam(mode);controls.append(gam)
        for name,value in warp.state_dict().items():
            if name in gam.state_dict():torch.testing.assert_close(value,gam.state_dict()[name],rtol=0,atol=0)
    a,b=[m.rgbwarp_correction.future_predictor.state_dict() for m in controls]
    assert a.keys()==b.keys()
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    for m in controls:
        ids=controller_head_parameter_ids(m.rgbwarp_correction)
        assert not ids & {id(p) for p in m.rgbwarp_correction.future_predictor.parameters()}


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_shared_loss_and_global_logging_no_future_rgb_input(mode,monkeypatch):
    from robot.data.dataset import ActionNormalizer
    from experiments.uavflow_predictor_idm import rgbwarp_objectives as obj
    from experiments.uavflow_predictor_idm.global_loss_logging import GlobalLossWindow
    net=make_gam(mode)
    language=torch.rand(2,7 if mode=='uav_gam' else 5,16)
    pose_embedding=torch.rand(2,16)
    monkeypatch.setattr(obj,'encode_stage2_condition',lambda *a,**k:dict(
        last_hidden_state=language,attention_mask=torch.ones(language.shape[:2],dtype=torch.bool),
        pose_embedding=pose_embedding))
    normalizer=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    batch=dict(all_view_images=torch.rand(2,2,1,3,224,224),
        openvla_prompt_pose=torch.zeros(2,2,4),frame_indices=torch.tensor([[0,5],[10,15]]),start_t=torch.tensor([0,10]),
        action_stats_key=['sim','sim'],task_description=['Turn','Land'],actions=torch.rand(2,2,5,4),
        action_loss_mask=torch.ones(2,2,5,4,dtype=torch.bool),
        gt_depth_meters=torch.ones(2,2,1,224,224)*3,gt_depth_mask=torch.ones(2,2,1,224,224,dtype=torch.bool))
    options=dict(model=net,da3=net.da3,text=SimpleNamespace(pose5_enabled=True),normalizer=normalizer,batch=batch,
        context_len=1,rollout_steps=1,amp=False,amp_dtype='fp32',
        depth_linear_weight=1/3,depth_log_weight=1/3,depth_grad_weight=1/3)
    a=obj.forward_rgbwarp_batch(**options)
    batch['all_view_images'][:,1].fill_(float('nan'))
    batch['actions'][:,1].fill_(float('nan'))
    b=obj.forward_rgbwarp_batch(**options)
    for key in ['action','depth','direct_action','refine_action']:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    torch.testing.assert_close(a['depth'],(a['depth_current_total']+a['depth_future_total'])/2)
    torch.testing.assert_close(a['action'],(a['direct_action']+a['refine_action'])/2)
    logger=GlobalLossWindow();logger.add(a,3*a['action']+3*a['depth'])
    report,loss,metadata=logger.finish()
    torch.testing.assert_close(report['depth'].float(),a['depth'])
    assert metadata['log_valid_elements']==40
    from experiments.uavflow_predictor_idm.objectives import evaluate
    evaluated=evaluate(model=net,da3=net.da3,text=options['text'],normalizer=normalizer,
        pose_normalizer=None,loader=[batch],contexts=[1],rollout=1,device=torch.device('cpu'),
        amp=False,amp_dtype='fp32',horizon_weights=[1.],feature_patch_weight=1.,feature_cls_weight=0.,
        feature_register_weight=0.,deep_feature_enabled=False,deep_feature_patch_weight=1.,
        deep_feature_cls_weight=0.,stop_pos_weight=5.,max_batches=0,
        depth_target_source='ue_gt',depth_target_mode='both',depth_linear_weight=1/3,
        depth_log_weight=1/3,depth_grad_weight=1/3)
    assert evaluated['H1_evaluated_windows']==2
    assert 'H1_depth_current_total' in evaluated and 'H1_depth_future_total' in evaluated
    assert 'H1_rgbwarp_valid_ratio' not in evaluated
    assert evaluated['H1_refine_action']==pytest.approx(a['refine_action'].item())


def test_t5_pose_exact_architecture_scaling_initialization_and_online_information(monkeypatch):
    import experiments.uavflow_paper_ablation.t5_pose as module
    from experiments.uavflow_predictor_idm.vlm_conditioning import encode_stage2_condition
    class Text(nn.Module):
        hidden_size=8
        def __init__(self,**kwargs):super().__init__();self.weight=nn.Parameter(torch.ones(1))
        def encode_tokens(self,texts,pad_to):
            return dict(last_hidden_state=torch.ones(len(texts),pad_to,8),
                        attention_mask=torch.ones(len(texts),pad_to,dtype=torch.bool))
    monkeypatch.setattr(module,'TextConditioner',Text)
    text=module.T5PoseConditioner('unused')
    assert text.state_encoder[0].in_features==5 and text.state_encoder[0].out_features==256
    assert isinstance(text.state_encoder[1],nn.GELU)
    assert text.state_encoder[2].out_features==2048
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43);first=nn.Linear(5,256)
        torch.manual_seed(44);last=nn.Linear(256,2048)
    for a,b in zip(text.state_encoder[0].parameters(),first.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(text.state_encoder[2].parameters(),last.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    pose=torch.tensor([[1000.,-500.,250.,90.,0.]])
    out=encode_stage2_condition(text,['Turn'],reference_images=torch.rand(1,1,3,224,224),
        current_images=torch.rand(1,1,3,224,224),pad_to=8,current_pose=pose)
    expected=text.state_encoder(torch.tensor([[1.,-.5,.25,.5,0.]]))
    torch.testing.assert_close(out['pose_embedding'],expected)
    assert out['last_hidden_state'].shape==(1,8,2048)
    changed=encode_stage2_condition(text,['Turn'],reference_images=torch.rand(1,1,3,224,224),
        current_images=torch.rand(1,1,3,224,224),pad_to=8,current_pose=torch.zeros_like(pose))
    torch.testing.assert_close(out['last_hidden_state'],changed['last_hidden_state'],rtol=0,atol=0)
    assert not torch.equal(out['pose_embedding'],changed['pose_embedding'])
    (out['last_hidden_state'].square().mean()+out['pose_embedding'].square().mean()).backward()
    assert all(p.grad is None for p in text.text.parameters())
    assert text.state_encoder[0].weight.grad.abs().sum()>0
    assert text.text_projection.weight.grad.abs().sum()>0
    text.train();assert not text.text.training


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_optimizer_checkpoint_roundtrip_and_topology_rejection(mode):
    net=make_gam(mode)
    result,_=call(net)
    loss=sum(result[key]['depth'].square().mean() for key in ['current_depth_output','deep_joint_features'])
    loss+=result['actions_norm'].square().mean()+result['direct_actions_norm'].square().mean()
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-5)
    loss.backward();optimizer.step()
    state=copy.deepcopy(dict(geometry_architecture_state=architecture_state(net),optimizer=optimizer.state_dict()))
    other=make_gam(mode);load_architecture_state(other,state)
    restored=torch.optim.AdamW([p for p in other.parameters() if p.requires_grad],lr=1e-5)
    restored.load_state_dict(state['optimizer'])
    assert len(restored.state)==len(optimizer.state)>0
    for name,value in net.rgbwarp_correction.state_dict().items():
        torch.testing.assert_close(value,other.rgbwarp_correction.state_dict()[name])
    with pytest.raises(ValueError,match='settings changed'):
        load_architecture_state(make_gam('uav_gam' if mode=='vla_gam' else 'vla_gam'),state)


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_existing_checkpoint_loader_and_observation_only_policy(mode,tmp_path,monkeypatch):
    from robot.data.dataset import ActionNormalizer
    from robot.modeling import da3_giant_encoder
    from experiments.uavflow_predictor_idm import vlm_conditioning
    from experiments.uavflow_support_aware.inference import load_policy
    class Conditioner(nn.Module):
        hidden_size=16
        pose5_enabled=True
        pose_progress_enabled=False
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.rand(1))
            self.register_buffer('state_scale',torch.tensor([1000.,1000.,1000.,180.,1.]))
    torch.manual_seed(42);net=make_gam(mode).eval();text=Conditioner()
    norm=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    cfg=build_config(mode)
    cfg.model.update(d_model=32,depth=1,num_heads=4,language_len=8,
        parallel_vla_gfm_width=32,parallel_vla_gfm_heads=4,stop_model_dim=32,
        train_deep_backbone=True,deep_train_start_block=0)
    cfg.model.da3_lora.update(rank=2,alpha=4)
    cfg.model.rgbwarp_correction={k:v for k,v in net.rgbwarp_correction.config.items() if k!='predictor_options'}
    cfg.training.amp=False
    checkpoint=dict(config=OmegaConf.to_container(cfg),predictor=net.predictor.state_dict(),
        semantic_geometry_action=net.semantic_geometry_action.state_dict(),
        parallel_action_head=net.parallel_action_head.state_dict(),
        parallel_action_correction_head=net.parallel_action_correction_head.state_dict(),
        geometry_architecture_state=architecture_state(net),
        da3_trainable={k:p for k,p in net.da3.named_parameters() if p.requires_grad},
        conditioner_trainable=text.state_dict(),action_normalizer=norm.state_dict())
    path=tmp_path/'gam.pt';torch.save(checkpoint,path)
    def da3_factory(**kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);return TinyDA3()
    monkeypatch.setattr(da3_giant_encoder,'DA3GiantEncoder',da3_factory)
    monkeypatch.setattr(vlm_conditioning,'build_stage2_conditioner',lambda *a:Conditioner())
    policy=load_policy(path,device='cpu')
    assert isinstance(policy.model.rgbwarp_correction,MatchedGAMCorrection)
    assert policy.model.rgbwarp_correction.mode==mode
    rgb=torch.rand(1,1,3,224,224)
    language=torch.rand(1,7 if mode=='uav_gam' else 5,16)
    pose_embedding=torch.rand(1,16)
    mask=torch.ones(language.shape[:2],dtype=torch.bool)
    seen=[]
    def encode(*a,**kwargs):
        seen.append(kwargs['current_pose'])
        return dict(last_hidden_state=language,attention_mask=mask,pose_embedding=pose_embedding)
    monkeypatch.setattr(vlm_conditioning,'encode_stage2_condition',encode)
    pose=torch.tensor([[100.,-30.,20.,90.,0.]])
    actual=policy.predict(rgb,['Turn'],current_prompt_pose=pose)
    expected=net(rgb[:,0],reference_shallow=None,lang_feats=language,lang_padding_mask=mask,
                 gam_pose_embedding=pose_embedding,action_denormalizer=lambda x:x)
    torch.testing.assert_close(actual['actions_norm'],expected['actions_norm'],rtol=0,atol=0)
    torch.testing.assert_close(seen[0],pose)
    assert actual['warp'] is None


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_pose_enters_native_sa_proprio_slot_and_backpropagates(mode,monkeypatch):
    net=make_gam(mode)
    predictor=net.rgbwarp_correction.future_predictor
    assert predictor.use_proprio_input
    assert isinstance(predictor.proprio_token_proj,nn.Linear)
    assert predictor.proprio_token_proj.in_features==16
    assert predictor.proprio_token_proj.out_features==32
    # Native LayerScale starts at zero. Open the existing SA branch to test
    # connectivity after the gate has learned, without changing production init.
    with torch.no_grad():
        for block in predictor.blocks:block.ls_self.gamma.fill_(.1)
    captured={}
    original=predictor.forward
    def record(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(predictor,'forward',record)
    language=torch.rand(2,7 if mode=='uav_gam' else 5,16)
    pose=torch.rand(2,16,requires_grad=True)
    result=net(torch.rand(2,3,224,224),reference_shallow=None,lang_feats=language,
        lang_padding_mask=torch.ones(language.shape[:2],dtype=torch.bool),gam_pose_embedding=pose,
        action_denormalizer=lambda x:x)
    torch.testing.assert_close(captured['proprio_history'],pose[:,None],rtol=0,atol=0)
    if mode=='uav_gam':assert captured['lang_feats'].shape==language.shape
    else:assert captured['lang_feats'] is None
    result['deep_joint_features']['depth'].square().mean().backward()
    assert pose.grad is not None and pose.grad.abs().sum()>0
    assert predictor.proprio_token_proj.weight.grad.abs().sum()>0


def test_configs_inherit_identical_dataset_loss_pose_and_training_contracts():
    from experiments.uavflow_paper_ablation.current_train import CONFIG
    baseline=OmegaConf.load(CONFIG.parent/'warp_reference.yaml')
    for mode in ['uav_gam','vla_gam']:
        cfg=build_config(mode)
        assert cfg.dataset==baseline.dataset
        expected_loss=copy.deepcopy(baseline.loss)
        expected_loss.feature_weight=1.;expected_loss.action_direct_weight=0.;expected_loss.action_refine_weight=1.;expected_loss.depth_target_mode='future'
        assert cfg.loss==expected_loss
        assert cfg.model.rgbwarp_correction.native_gam_path and cfg.model.rgbwarp_correction.native_language_dim==768
        assert cfg.training.batch_size==8 and cfg.training.gradient_accumulation_steps==1
        assert cfg.training.preserve_warp24_microgroups
        assert cfg.training.lr==5e-5 and cfg.training.predictor_lr_mult==.2
        assert cfg.training.direct_action_head_lr_mult==10
        assert cfg.training.max_epochs==20 and cfg.training.eval_max_batches==0
        assert cfg.training.eval_every==5000 and cfg.training.eval_every_epoch
        assert cfg.model.da3_lora==baseline.model.da3_lora
        assert cfg.model.rgbwarp_correction.action_output_mode=='direct'
        assert cfg.model.stop_head_enabled is False and cfg.training.epoch_weights_only is False
    assert build_config('vla_gam').stage1==baseline.stage1
    assert build_config('uav_gam').stage1.text_encoder_type=='t5'
    assert not build_config('uav_gam').stage1.qwen_lora_enabled
    assert build_config('uav_gam').stage1.paper_t5_native_text
