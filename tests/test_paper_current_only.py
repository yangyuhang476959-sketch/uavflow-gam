import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from test_rgbwarp_correction import make_model
from experiments.uavflow_paper_ablation.current_only import CurrentOnlyCorrection
from experiments.uavflow_paper_ablation.current_train import build_config
from experiments.uavflow_paper_ablation.sampling import Warp24MicrogroupSampler
from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state, load_architecture_state


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


def current_model():
    return make_model(corr=False, mask=False, output_mode='direct', branch_mode='current_only')


def test_current_only_single_pass_no_warp_no_residual_and_gradients(monkeypatch):
    net = current_model().train()
    assert isinstance(net.rgbwarp_correction, CurrentOnlyCorrection)
    assert isinstance(net.predictor, nn.Identity)
    assert net.rgbwarp_correction.mask_projection is None
    assert net.rgbwarp_correction.correspondence is None
    assert net.action_residual_gate_logit is None
    def forbidden(*args, **kwargs):
        raise AssertionError('Current-only invoked a Warp-only path')
    monkeypatch.setattr(net.da3, 'propagate_shallow_visual_slots_grad', forbidden)
    import experiments.uavflow_predictor_idm.rgbwarp_correction as warp
    monkeypatch.setattr(warp, 'forward_splat', forbidden)
    monkeypatch.setattr(warp, 'forward_alpha_splat', forbidden)
    monkeypatch.setattr(warp, 'action_camera_transform', forbidden)
    slots = torch.rand(2, 5, 16, requires_grad=True)
    rgb = torch.rand(2, 3, 224, 224)
    out = net(rgb, reference_shallow=None, lang_feats=slots, lang_padding_mask=None,
              action_denormalizer=forbidden)
    assert out['warp'] is None and out['deep_joint_features'] is None
    assert out['actions_norm'].shape == (2, 1, 5, 4)
    assert net.da3.dpt_head.counts == [256]
    final_before = out['actions_norm'].detach().clone()
    # Numeric proposal does not feed the current DA3 or get added to final.
    with torch.no_grad():
        net.parallel_action_head.decoder.output.bias.add_(10.)
    again = net(rgb, reference_shallow=None, lang_feats=slots, lang_padding_mask=None,
                action_denormalizer=forbidden)
    torch.testing.assert_close(again['actions_norm'], final_before, rtol=0, atol=0)
    depth_to_slots=torch.autograd.grad(again['current_depth_output']['depth'].mean(),slots,retain_graph=True)[0]
    assert torch.isfinite(depth_to_slots).all() and depth_to_slots.abs().sum()>0
    loss = again['actions_norm'].square().mean() + again['direct_actions_norm'].square().mean()
    loss += again['current_depth_output']['depth'].square().mean()
    loss.backward()
    assert slots.grad.abs().sum() > 0
    assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert all(p.grad is None for p in net.da3.dpt_head.parameters())
    assert not ({id(p) for p in net.parallel_action_head.parameters()}
                & {id(p) for p in net.parallel_action_correction_head.parameters()})


def objective_fixture(monkeypatch):
    from robot.data.dataset import ActionNormalizer
    from experiments.uavflow_predictor_idm import rgbwarp_objectives as obj
    net = current_model()
    slots = torch.rand(2, 5, 16, requires_grad=True)
    seen_pose = []
    def encode(*args, **kwargs):
        seen_pose.append(kwargs['current_pose'])
        return dict(last_hidden_state=slots, attention_mask=None)
    monkeypatch.setattr(obj, 'encode_stage2_condition', encode)
    text = SimpleNamespace(pose5_enabled=True, pose_progress_enabled=False)
    norm = ActionNormalizer({'sim': dict(q01=[-1.]*4, q99=[1.]*4, mask=[True]*4)}, default_key='sim')
    batch = dict(all_view_images=torch.rand(2,2,1,3,224,224),
                 openvla_prompt_pose=torch.rand(2,2,4), frame_indices=torch.tensor([[0,5],[9,14]]),
                 start_t=torch.tensor([0,9]),
                 action_stats_key=['sim','sim'], task_description=['Turn','Land'],
                 actions=torch.rand(2,2,5,4)*.05,
                 action_loss_mask=torch.ones(2,2,5,4,dtype=torch.bool),
                 gt_depth_meters=torch.full((2,2,1,224,224),3.),
                 gt_depth_mask=torch.ones(2,2,1,224,224,dtype=torch.bool))
    args = dict(model=net, da3=net.da3, text=text, normalizer=norm, batch=batch,
                context_len=1, rollout_steps=1, amp=False, amp_dtype='fp32',
                depth_linear_weight=1/3, depth_log_weight=1/3, depth_grad_weight=1/3)
    return obj, net, batch, args, seen_pose


def test_loss_current_gt_only_pose_and_invalid_mask(monkeypatch):
    obj, net, batch, args, seen = objective_fixture(monkeypatch)
    a = obj.forward_rgbwarp_batch(**args)
    torch.testing.assert_close(a['depth'], a['depth_current_total'])
    torch.testing.assert_close(a['action'], (a['direct_action']+a['refine_action'])/2)
    assert not any(k.startswith('depth_future_') for k in a)
    assert seen[-1][:,-1].tolist() == [1.,0.]
    torch.testing.assert_close(seen[-1][:,:4], batch['openvla_prompt_pose'][:,0])
    batch['all_view_images'][:,1].fill_(float('nan'))
    batch['gt_depth_meters'][:,1].fill_(float('nan'))
    batch['gt_depth_mask'][:,1].zero_()
    batch['actions'][:,1].fill_(float('nan'))
    b = obj.forward_rgbwarp_batch(**args)
    for key in ('action','depth','direct_action','refine_action'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    batch['gt_depth_meters'][:,0].fill_(float('nan'))
    batch['gt_depth_mask'][:,0].zero_()
    c = obj.forward_rgbwarp_batch(**args)
    assert c['depth'].item() == 0 and torch.isfinite(c['depth'])


def test_global_interval_logging_uses_current_depth_without_future(monkeypatch):
    from experiments.uavflow_predictor_idm.global_loss_logging import GlobalLossWindow
    obj,net,batch,args,seen=objective_fixture(monkeypatch)
    values=obj.forward_rgbwarp_batch(**args)
    logger=GlobalLossWindow(current_only=True)
    loss=3*values['action']+3*values['depth']
    logger.add(values,loss)
    result,total,metadata=logger.finish()
    torch.testing.assert_close(result['depth'].float(),values['depth'].float())
    torch.testing.assert_close(result['depth_linear_l1'].float(),values['depth_linear_l1'].float())
    torch.testing.assert_close(total.float(),loss.float())
    assert metadata['log_valid_elements']==40
    assert not any(k.startswith('depth_future_') for k in result)


def test_full_evaluate_reports_current_not_fake_future_or_warp_metrics(monkeypatch):
    from experiments.uavflow_predictor_idm.objectives import evaluate
    obj, net, batch, args, seen = objective_fixture(monkeypatch)
    result = evaluate(model=net, da3=net.da3, text=args['text'], normalizer=args['normalizer'],
        pose_normalizer=None, loader=[batch], contexts=[1], rollout=1, device=torch.device('cpu'),
        amp=False, amp_dtype='fp32', horizon_weights=[1.], feature_patch_weight=1.,
        feature_cls_weight=0., feature_register_weight=0., deep_feature_enabled=False,
        deep_feature_patch_weight=1., deep_feature_cls_weight=0., stop_pos_weight=5.,
        depth_target_source='ue_gt', depth_target_mode='current', max_batches=0)
    assert result['H1_evaluated_windows'] == 2
    assert 'H1_depth_current_total' in result
    assert not any(k.startswith('H1_depth_future_') for k in result)
    assert 'H1_rgbwarp_valid_ratio' not in result


def test_checkpoint_topology_and_full_optimizer_roundtrip():
    net = current_model()
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=5e-4)
    out = net(torch.rand(1,3,224,224),reference_shallow=None,lang_feats=torch.rand(1,5,16),
              lang_padding_mask=None,action_denormalizer=lambda x:x)
    (out['actions_norm'].square().mean()+out['direct_actions_norm'].square().mean()
     +out['current_depth_output']['depth'].square().mean()).backward()
    optimizer.step()
    state = copy.deepcopy(dict(geometry_architecture_state=architecture_state(net),optimizer=optimizer.state_dict()))
    other = current_model()
    load_architecture_state(other,state)
    restored = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad],lr=5e-4)
    restored.load_state_dict(state['optimizer'])
    assert len(restored.state)==len(optimizer.state)>0
    for key, tensor in net.rgbwarp_correction.state_dict().items():
        torch.testing.assert_close(tensor,other.rgbwarp_correction.state_dict()[key])
    warp = make_model(corr=False,mask=False,output_mode='direct')
    with pytest.raises(ValueError, match='settings changed'):
        load_architecture_state(warp,state)


@pytest.mark.parametrize('size',[23,24,25,50,309145])
def test_sampler_preserves_exact_baseline_rank_groups_and_resume(size):
    data=range(size)
    old=[]
    for rank in range(3):
        sampler=DistributedSampler(data,num_replicas=3,rank=rank,seed=42)
        sampler.set_epoch(2)
        indices=list(sampler)
        old.append(indices[:len(indices)//8*8])
    sampler=Warp24MicrogroupSampler(data,1,0,42,resume_epoch=2)
    sampler.set_epoch(2)
    actual=list(sampler)
    expected=[]
    for update in range(sampler.updates):
        for rank in range(3):
            expected.extend(old[rank][update*8:(update+1)*8])
    assert actual==expected and len(actual)==len(sampler)
    sampler.resume_batch=1
    assert list(sampler)==expected[24:]
    for rank in range(3):
        distributed=Warp24MicrogroupSampler(data,3,rank,42,batch_size=8)
        distributed.set_epoch(2)
        assert list(distributed)==old[rank]


def test_config_changes_are_explicit_and_public_recipe_preserved():
    from omegaconf import OmegaConf
    from experiments.uavflow_paper_ablation.current_train import CONFIG
    baseline=OmegaConf.load(CONFIG.parent/'warp_reference.yaml')
    current=build_config()
    assert current.model.rgbwarp_correction.branch_mode=='current_only'
    assert current.training.batch_size*current.training.gradient_accumulation_steps==24
    assert current.training.preserve_warp24_microgroups
    assert current.loss.depth_target_mode=='current'
    assert current.training.epoch_weights_only is False
    assert current.dataset==baseline.dataset and current.stage1==baseline.stage1
    assert current.model.da3_lora==baseline.model.da3_lora
    assert current.training.lr==baseline.training.lr==5e-5
    assert current.training.predictor_lr_mult==.2
    assert current.training.direct_action_head_lr_mult==10
    assert current.training.max_epochs==20


def test_observation_only_inference_uses_current_path_and_same_pose(monkeypatch):
    from experiments.uavflow_support_aware.inference import RGBWarpPolicy
    from experiments.uavflow_predictor_idm import vlm_conditioning
    from robot.data.dataset import ActionNormalizer
    class Conditioner(nn.Module):
        pose5_enabled=True
        pose_progress_enabled=False
        def __init__(self):
            super().__init__()
            self.state_encoder=nn.Sequential(nn.Linear(5,256),nn.GELU(),nn.Linear(256,16))
            self.register_buffer('state_scale',torch.tensor([1000.,1000.,1000.,180.,1.]))
    conditioner=Conditioner()
    net=current_model().eval()
    norm=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    slots=torch.rand(2,5,16)
    poses=[]
    def encode(*args,**kwargs):
        poses.append(kwargs['current_pose'])
        return dict(last_hidden_state=slots,attention_mask=None)
    monkeypatch.setattr(vlm_conditioning,'encode_stage2_condition',encode)
    policy=RGBWarpPolicy(net,conditioner,norm,amp=False)
    rgb=torch.rand(2,1,3,224,224)
    pose=torch.tensor([[0.,0.,0.,0.,1.],[150.,-30.,20.,90.,0.]])
    actual=policy.predict(rgb,['Turn','Land'],current_prompt_pose=pose)
    expected=net(rgb[:,0],reference_shallow=None,lang_feats=slots,lang_padding_mask=None,
                 action_denormalizer=lambda x:norm.denormalize(x))
    torch.testing.assert_close(actual['actions_meters_radians'],norm.denormalize(expected['actions_norm'])[:,0])
    torch.testing.assert_close(poses[0],pose)
    assert actual['warp'] is None and actual['deep_joint_features'] is None


def test_public_parameters_have_same_constructor_initialization_as_warp():
    torch.manual_seed(42)
    warp=make_model(corr=False,mask=True,output_mode='direct')
    torch.manual_seed(42)
    current=current_model()
    common=set(warp.state_dict()) & set(current.state_dict())
    assert common
    for name in common:
        torch.testing.assert_close(warp.state_dict()[name],current.state_dict()[name],rtol=0,atol=0)


def test_real_checkpoint_loader_selects_current_adapter(tmp_path,monkeypatch):
    from robot.data.dataset import ActionNormalizer
    from robot.modeling import da3_giant_encoder
    from experiments.uavflow_predictor_idm import vlm_conditioning
    from experiments.uavflow_support_aware.inference import load_policy
    from test_rgbwarp_correction import TinyDA3
    from omegaconf import OmegaConf
    class Conditioner(nn.Module):
        hidden_size=16
        pose5_enabled=True
        pose_progress_enabled=False
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.rand(1))
            self.register_buffer('state_scale',torch.tensor([1000.,1000.,1000.,180.,1.]))
    torch.manual_seed(42)
    model=current_model().eval()
    conditioner=Conditioner()
    norm=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    cfg=build_config()
    cfg.model.update(d_model=32,depth=1,num_heads=4,language_len=8,
        parallel_vla_gfm_width=32,parallel_vla_gfm_heads=4,
        stop_model_dim=32,train_deep_backbone=True,deep_train_start_block=0)
    cfg.model.da3_lora.update(rank=2,alpha=4)
    cfg.model.rgbwarp_correction=dict(model.rgbwarp_correction.config)
    cfg.training.amp=False
    checkpoint=dict(config=OmegaConf.to_container(cfg),predictor=model.predictor.state_dict(),
        semantic_geometry_action=model.semantic_geometry_action.state_dict(),
        parallel_action_head=model.parallel_action_head.state_dict(),
        parallel_action_correction_head=model.parallel_action_correction_head.state_dict(),
        geometry_architecture_state=architecture_state(model),
        da3_trainable={k:p for k,p in model.da3.named_parameters() if p.requires_grad},
        conditioner_trainable=conditioner.state_dict(),action_normalizer=norm.state_dict())
    path=tmp_path/'current.pt'
    torch.save(checkpoint,path)
    def da3_factory(**kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            return TinyDA3()
    monkeypatch.setattr(da3_giant_encoder,'DA3GiantEncoder',da3_factory)
    monkeypatch.setattr(vlm_conditioning,'build_stage2_conditioner',lambda *a:Conditioner())
    policy=load_policy(path,device='cpu')
    assert isinstance(policy.model.rgbwarp_correction,CurrentOnlyCorrection)
    rgb=torch.rand(1,3,224,224)
    slots=torch.rand(1,5,16)
    inputs=dict(reference_shallow=None,lang_feats=slots,lang_padding_mask=None,
                action_denormalizer=lambda x:x)
    expected=model(rgb,**inputs)
    actual=policy.model(rgb,**inputs)
    torch.testing.assert_close(actual['actions_norm'],expected['actions_norm'],rtol=0,atol=0)
    torch.testing.assert_close(actual['current_depth_output']['depth'],expected['current_depth_output']['depth'],rtol=0,atol=0)
