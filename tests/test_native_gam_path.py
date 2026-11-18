import copy
from types import SimpleNamespace
import torch
from torch import nn
import pytest
from test_rgbwarp_correction import TinyDA3
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.rgbwarp_objectives import forward_rgbwarp_batch
from robot.data.dataset import ActionNormalizer

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1);torch.manual_seed(42)

def make_native(mode='uav_gam',fixed=False,vlm_pose_only=False,slot_positions=False):
    da3=TinyDA3();da3.configure_deep_lora(0,rank=2,alpha=4,dropout=0,target_modules=['all-linear'])
    return UAVFlowPredictorIDM(da3=da3,idm=None,action_dim=4,action_chunk_size=5,d_model=32,depth=1,num_heads=4,language_dim=16,language_len=8,direct_action_enabled=True,deep_action_enabled=True,compute_idm_branch=False,geometry_architecture='dual_vla_gfm',parallel_vla_gfm_enabled=True,parallel_vla_gfm_mode='qwen_tokens',parallel_action_decode_mode='geometry_residual',parallel_action_position_mode='decoder_only',parallel_vla_gfm_width=32,parallel_vla_gfm_heads=4,action_residual_gate_init=None,parallel_current_geometry_read_enabled=False,parallel_current_depth_enabled=True,depth_decode_enabled=True,gradient_checkpointing=True,train_deep_backbone=True,deep_train_start_block=0,dual_stream_rgbwarp_enabled=True,rgbwarp_correction=dict(action_output_mode='direct',use_correspondence_attention=False,use_pixel_binary_mask=False,branch_mode=mode,native_gam_path=True,native_language_dim=8,native_fixed_language_positions=True,native_frozen_shallow=fixed,native_sdpa=fixed,vla_pose_in_vlm_only=vlm_pose_only,gam_action_slot_position_encoding=slot_positions))

def test_native_t5_has_no_extra_text_linear_and_pose_unchanged(monkeypatch):
    from experiments.uavflow_paper_ablation import t5_pose
    class Text(nn.Module):
        hidden_size=8
        def __init__(self,**kw):super().__init__();self.weight=nn.Parameter(torch.ones(1))
        def encode_tokens(self,texts,pad_to):return dict(last_hidden_state=torch.ones(len(texts),pad_to,8),attention_mask=torch.ones(len(texts),pad_to,dtype=torch.bool))
    monkeypatch.setattr(t5_pose,'TextConditioner',Text)
    old=t5_pose.T5PoseConditioner('unused');native=t5_pose.T5PoseConditioner('unused',native_text=True)
    assert isinstance(native.text_projection,nn.Identity) and native.hidden_size==8 and native.action_interface_dim==2048
    out=native.encode_tokens(['Turn'],pad_to=8,current_pose=torch.zeros(1,5));assert out['last_hidden_state'].shape==(1,8,8)
    for k,v in old.state_encoder.state_dict().items():torch.testing.assert_close(v,native.state_encoder.state_dict()[k],rtol=0,atol=0)
    assert not [n for n,p in native.named_parameters() if n.startswith('text_projection')]

@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_no_current_or_pre_action_and_all_trainables_receive_grad(mode,monkeypatch):
    net=make_native(mode)
    def forbidden(*a,**k):raise AssertionError('current deep decode or auxiliary action head called')
    monkeypatch.setattr(net.da3,'propagate_shallow_visual_slots_grad',forbidden)
    monkeypatch.setattr(net.parallel_action_head,'forward',forbidden)
    language=torch.rand(2,8 if mode=='uav_gam' else 5,8 if mode=='uav_gam' else 16)
    out=net(torch.rand(2,3,224,224),reference_shallow=None,lang_feats=language,lang_padding_mask=None,gam_pose_embedding=torch.rand(2,16),action_denormalizer=forbidden)
    assert out['current_depth_output'] is None and out['direct_actions_norm'] is None
    assert net.da3.dpt_head.counts==[256]
    loss=out['actions_norm'].square().mean()+out['deep_joint_features']['depth'].square().mean()+out['predicted_shallow'].square().mean();loss.backward()
    assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert all(not p.requires_grad for p in net.parallel_action_head.parameters())


@pytest.mark.parametrize('mode',['uav_gam','vla_gam'])
def test_future_only_loss_no_gt_input_and_logging(monkeypatch,mode):
    from experiments.uavflow_predictor_idm import rgbwarp_objectives as obj
    from experiments.uavflow_predictor_idm.global_loss_logging import GlobalLossWindow
    net=make_native(mode,vlm_pose_only=(mode=='vla_gam'));language=torch.rand(2,8 if mode=='uav_gam' else 5,8 if mode=='uav_gam' else 16);pose=torch.rand(2,16) if mode=='uav_gam' else None
    monkeypatch.setattr(obj,'encode_stage2_condition',lambda *a,**k:dict(last_hidden_state=language,attention_mask=torch.ones(language.shape[:2],dtype=torch.bool),pose_embedding=pose))
    # Native objective imports the shared encoder entry; patch that binding too.
    import experiments.uavflow_paper_ablation.native_objectives as native
    monkeypatch.setattr(native,'encode_stage2_condition',obj.encode_stage2_condition)
    normalizer=ActionNormalizer({'sim':dict(q01=[-1.]*4,q99=[1.]*4,mask=[True]*4)},default_key='sim')
    images=torch.rand(2,2,1,3,224,224,requires_grad=True)
    batch=dict(all_view_images=images,openvla_prompt_pose=torch.zeros(2,2,4),frame_indices=torch.tensor([[0,5],[10,15]]),start_t=torch.tensor([0,10]),action_stats_key=['sim','sim'],task_description=['Turn','Land'],actions=torch.rand(2,2,5,4)*.05,action_loss_mask=torch.ones(2,2,5,4,dtype=torch.bool),gt_depth_meters=torch.full((2,2,1,224,224),3.),gt_depth_mask=torch.ones(2,2,1,224,224,dtype=torch.bool))
    args=dict(model=net,da3=net.da3,text=SimpleNamespace(pose5_enabled=True,pose_progress_enabled=False),normalizer=normalizer,batch=batch,context_len=1,rollout_steps=1,amp=False,amp_dtype='fp32',depth_linear_weight=1/3,depth_log_weight=1/3,depth_grad_weight=1/3)
    a=forward_rgbwarp_batch(**args);assert a['feature']>0;torch.testing.assert_close(a['action'],a['refine_action']);torch.testing.assert_close(a['depth'],a['depth_future_total'])
    assert not any(k.startswith('depth_current_') for k in a)
    changed=dict(batch,all_view_images=torch.cat((images[:,:1],torch.zeros_like(images[:,1:])),1),gt_depth_meters=batch['gt_depth_meters'].clone());changed['gt_depth_meters'][:,0]=float('nan')
    b=forward_rgbwarp_batch(**dict(args,batch=changed))
    for k in ['action','depth','refine_action']:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert not torch.equal(a['feature'],b['feature'])
    loss=3*a['action']+a['feature']+3*a['depth'];loss.backward();assert images.grad[:,0].abs().sum()>0 and images.grad[:,1].abs().sum()==0
    logger=GlobalLossWindow(future_only=True,final_only=True);logger.add(a,loss);r,total,_=logger.finish()
    for k in ['action','depth','feature']:torch.testing.assert_close(r[k],a[k].double())
    assert 'rgbwarp_proposal_loss' not in r and not any(k.startswith('depth_current_') for k in r)

    from experiments.uavflow_predictor_idm.objectives import evaluate
    result=evaluate(model=net,da3=net.da3,text=args['text'],normalizer=normalizer,pose_normalizer=None,loader=[batch],contexts=[1],rollout=1,device=torch.device('cpu'),amp=False,amp_dtype='fp32',horizon_weights=[1.],feature_patch_weight=1.,feature_cls_weight=0.,feature_register_weight=0.,deep_feature_enabled=False,deep_feature_patch_weight=1.,deep_feature_cls_weight=0.,stop_pos_weight=5.,depth_target_source='ue_gt',depth_target_mode='future',max_batches=0)
    assert result['H1_evaluated_windows']==2 and result['H1_feature']>0
    assert 'H1_depth_future_total' in result
    assert not any('depth_current_' in key or 'proposal_loss' in key for key in result)



def test_uav_vla_native_predictors_same_initial_state():
    torch.manual_seed(42);a=make_native();torch.manual_seed(42);b=make_native('vla_gam')
    for name,value in a.rgbwarp_correction.future_predictor.state_dict().items():torch.testing.assert_close(value,b.rgbwarp_correction.future_predictor.state_dict()[name],rtol=0,atol=0)
    assert a.rgbwarp_correction.future_predictor.lang_proj.in_features==8
    assert a.rgbwarp_correction.future_predictor.lang_pos.shape==(8,32)
    assert not b.rgbwarp_correction.future_predictor.lang_pos.requires_grad


def test_final_head_initialization_matches_warp_and_full_resume():
    from test_rgbwarp_correction import make_model
    from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state,load_architecture_state
    torch.manual_seed(42);warp=make_model(corr=False,mask=True,output_mode='direct')
    torch.manual_seed(42);native=make_native()
    for key,value in warp.parallel_action_correction_head.state_dict().items():torch.testing.assert_close(value,native.parallel_action_correction_head.state_dict()[key],rtol=0,atol=0)
    out=native(torch.rand(1,3,224,224),reference_shallow=None,lang_feats=torch.rand(1,8,8),lang_padding_mask=None,gam_pose_embedding=torch.rand(1,16),action_denormalizer=lambda x:x)
    optimizer=torch.optim.AdamW([p for p in native.parameters() if p.requires_grad],lr=1e-5)
    (out['actions_norm'].square().mean()+out['deep_joint_features']['depth'].square().mean()+out['predicted_shallow'].square().mean()).backward();optimizer.step()
    state=copy.deepcopy(dict(geometry_architecture_state=architecture_state(native),optimizer=optimizer.state_dict()))
    restored=make_native();load_architecture_state(restored,state)
    other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-5);other.load_state_dict(state['optimizer'])
    assert len(other.state)==len(optimizer.state)>0


def test_active_da3_path_matches_git_reference(monkeypatch):
    import subprocess,textwrap
    from pathlib import Path
    import robot.modeling.da3_giant_encoder as encoder
    try:original=subprocess.check_output(['git','show','HEAD:src/robot/modeling/da3_giant_encoder.py'],stderr=subprocess.DEVNULL,text=True)
    except (subprocess.CalledProcessError,FileNotFoundError):pytest.skip('Git reference unavailable in code-only export')
    import ast
    tree=ast.parse(original);names=['propagate_shallow_with_actions_grad','_propagate_shallow_with_actions_impl']
    namespace=dict(vars(encoder));reference={}
    for name in names:
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(textwrap.dedent(ast.get_source_segment(original,node)),'<git-reference>','exec'),namespace)
        reference[name]=namespace[name]
    net=make_native().eval();rgb=torch.rand(1,3,224,224);language=torch.rand(1,8,8);pose=torch.rand(1,16)
    def run():return net(rgb,reference_shallow=None,lang_feats=language,lang_padding_mask=None,gam_pose_embedding=pose,action_denormalizer=lambda x:x)
    with torch.no_grad():current=run()
    with monkeypatch.context() as patch:
        for name,value in reference.items():patch.setattr(encoder.DA3GiantEncoder,name,value)
        with torch.no_grad():old=run()
    for key in ['actions_norm','predicted_shallow']:torch.testing.assert_close(current[key],old[key],rtol=0,atol=0)
    torch.testing.assert_close(current['deep_joint_features']['depth'],old['deep_joint_features']['depth'],rtol=0,atol=0)


def test_fixed_teacher_is_immutable_and_only_deep_lora_updates():
    from experiments.uavflow_paper_ablation.frozen_shallow import fixed_features,prefix_parameters
    net=make_native(fixed=True).train();rgb=torch.rand(2,3,224,224,requires_grad=True)
    teacher_before=fixed_features(net.da3,rgb).clone()
    prefix_before={n:p.detach().clone() for n,p in prefix_parameters(net.da3)}
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-4)
    for _ in range(3):
        opt.zero_grad();out=net(rgb,reference_shallow=None,lang_feats=torch.rand(2,8,8),lang_padding_mask=None,gam_pose_embedding=torch.rand(2,16),action_denormalizer=lambda x:x)
        (out['actions_norm'].square().mean()+out['deep_joint_features']['depth'].square().mean()+(out['predicted_shallow']-teacher_before).square().mean()).backward();opt.step()
    assert rgb.grad is None
    assert all(not p.requires_grad and p.grad is None and torch.equal(p,prefix_before[n]) for n,p in prefix_parameters(net.da3))
    assert torch.equal(fixed_features(net.da3,rgb),teacher_before)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in net.da3.named_parameters() if p.requires_grad)
    assert all(p.grad is not None for p in net.rgbwarp_correction.future_predictor.parameters() if p.requires_grad)


def test_native_sdpa_accepts_changing_batch_and_has_original_mask(monkeypatch):
    net=make_native(fixed=True);pred=net.rgbwarp_correction.future_predictor
    assert pred.native_sdpa and pred._get_flex_block_mask(1,1,torch.device('cuda')) is None
    mask=pred._build_dense_block_causal_mask(1,1,torch.device('cpu'))
    assert mask.all() # One timestep retains native within-block bidirectionality.
    for training in [True,False,True]:
        net.train(training)
        for b in [2,1,3]:
            out=net(torch.rand(b,3,224,224),reference_shallow=None,lang_feats=torch.rand(b,8,8),lang_padding_mask=torch.ones(b,8,dtype=torch.bool),gam_pose_embedding=torch.rand(b,16),action_denormalizer=lambda x:x)
            assert torch.isfinite(out['actions_norm']).all()
