import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from robot.modeling.provenance_attention import routed_block, LocalReliability, shared_residual
from robot.modeling.da3_giant_encoder import DA3GiantEncoder
from experiments.uavflow_predictor_idm.provenance_correction import checkpoint_state, load_checkpoint_state
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.semantic_geometry_action import ParallelContinuousActionHead


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    import robot.modeling.future_predictor as predictor
    monkeypatch.setattr(predictor, '_HAS_FLEX', False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ls1, self.ls2 = nn.Identity(), nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, 3*d)
        self.attn.q_norm, self.attn.k_norm = nn.Identity(), nn.Identity()
        self.attn.num_heads, self.attn.rope = 4, None
        self.attn.proj, self.attn.proj_drop, self.attn.attn_drop = nn.Linear(d, d), nn.Identity(), nn.Dropout(0.)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.attn.qkv(self.norm1(x)).reshape(b, n, 3, 4, d//4).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x+self.attn.proj(y)
        return x+self.mlp(self.norm2(x))


def data():
    torch.manual_seed(12)
    x = torch.randn(2, 1, 23, 16)  # camera + 5 actions + register + 16 patches
    return x, x[:, :, 1:6].clone(), torch.rand(2, 16), Block()


def test_view_trunk_not_duplicated_and_parameters_shared():
    x, u, rho, block = data()
    ids = [id(p) for p in block.parameters()]
    legacy = block(x[:, 0])
    routed, _, _ = routed_block(x, u, block, None, action_count=5, rho=rho)
    # Initially equal action states => exact old visual update and key count.
    torch.testing.assert_close(routed[:, 0, :1], legacy[:, :1])
    torch.testing.assert_close(routed[:, 0, 6:], legacy[:, 6:])
    assert ids == [id(p) for p in block.parameters()]


@pytest.mark.parametrize('r,which', [(1., 'unsupported'), (0., 'supported')])
def test_zero_direct_visual_contribution(r, which):
    x, u, _, block = data()
    # With zero action values, the zero-membership branch has zero attention
    # read. Kill affine biases so its remaining change is only the FFN.
    block.norm1 = nn.Identity()
    with torch.no_grad():
        block.attn.qkv.bias.zero_()
        x[:, :, 1:6].zero_(); u.zero_()
        block.attn.proj.bias.zero_()
    out, uns, stats = routed_block(x, u, block, None, action_count=5, rho=torch.full((2,16), r))
    if which == 'unsupported':
        assert stats['unsupported_read_before'].item() == 0.
        torch.testing.assert_close(uns[:, 0], block.mlp(block.norm2(u[:, 0])))
    else:
        torch.testing.assert_close(out[:, 0, 1:6], block.mlp(block.norm2(x[:, 0, 1:6])))


def test_confidence_one_equivalence_and_supported_unaffected():
    x, u, rho, block = data()
    base, bu, _ = routed_block(x, u, block, None, action_count=5, rho=rho)
    one, ou, _ = routed_block(x, u, block, None, action_count=5, rho=rho, confidence=torch.ones_like(rho))
    torch.testing.assert_close(base, one); torch.testing.assert_close(bu, ou)
    zero, zu, stats = routed_block(x, u, block, None, action_count=5, rho=rho, confidence=torch.zeros_like(rho))
    # Supported action read is unchanged in THIS block; geometry changes
    # because unsupported geometry->action-source conditioning is attenuated.
    torch.testing.assert_close(base[:,:,1:6], zero[:,:,1:6])
    assert not torch.allclose(bu, zu)
    assert stats['unsupported_read_after'].item() == 0.


def test_uncertainty_gradients_are_independent():
    x, u, rho, block = data()
    x.requires_grad_(); u.requires_grad_()
    head = LocalReliability(16, [9], width=8)
    unc, conf = head(x[:, 0, -16:], 9)
    y, z, _ = routed_block(x, u, block, None, action_count=5, rho=rho, confidence=conf)
    (y.square().mean()+z.square().mean()).backward()
    assert all(p.grad is None for p in head.parameters())
    assert block.attn.qkv.weight.grad is not None
    assert x.grad is not None and u.grad is not None
    x.grad = None
    loss, _ = head.loss({9: {'patches':x[:,0,-16:], 'uncertainty':unc, 'confidence':conf}},
        {'layer_patches':{8:torch.randn(2,1,1,16,16)}}, rho, torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())
    assert x.grad is None


def test_shared_residual_bias_counted_once():
    head = ParallelContinuousActionHead(input_dim=16, action_dim=4, chunk_size=5, decoder_step_embedding=True)
    s, u = torch.randn(2,1,1,5,16), torch.randn(2,1,1,5,16)
    rho = torch.rand(2,16)
    ds, du, alpha = shared_residual(head, s, u, rho)
    torch.testing.assert_close(ds+du, alpha*head(s)+(1-alpha)*head(u))
    (ds+du).sum().backward()
    torch.testing.assert_close(head.decoder.output.bias.grad, torch.full((4,), 10.))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(10)])
        self.norm = nn.LayerNorm(16)
        self.camera_token = nn.Parameter(torch.randn(1,2,16))
        self.alt_start, self.rope_start, self.rope, self.cat_token = 1, 100, None, True
        self.num_register_tokens = 0

    def _prepare_rope(self, *args):
        return None, None

    def process_attention(self, x, block, **kwargs):
        b, v, n, d = x.shape
        return block(x.reshape(b*v,n,d)).reshape(b,v,n,d)


class TinyDA3(DA3GiantEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        self.embed_dim, self.num_register_tokens = 16, 0
        self.out_layers = [3,5,7,9]
        self.temporal_embed = None
        self.backbone = nn.Module(); self.backbone.pretrained = Transformer()
        self.dpt_head = nn.Identity()

    def encode_shallow_visual_slots(self, images, T, V):
        value = images.reshape(images.shape[0],T,V,-1).mean(-1)
        return {'visual_tokens':value[...,None,None].expand(-1,-1,-1,257,16)}

    def propagate_shallow_with_actions_grad(self, *args, **kwargs):
        kwargs['decode_visuals'] = False
        kwargs['return_multi_level'] = True
        out = super().propagate_shallow_with_actions_grad(*args, **kwargs)
        out.update(self.decode_depth_full(out['level_feats']))
        return out

    def decode_depth_full(self, pairs, **kwargs):
        p = pairs[-1][0]
        depth = p.square().mean(-1).reshape(p.shape[0],1,16,16)*.01+.04
        return {'depth': F.interpolate(depth, size=(224,224), mode='nearest')}


def model(mode, bank=False):
    return UAVFlowPredictorIDM(da3=TinyDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16, language_len=8,
        direct_action_enabled=True, deep_action_enabled=True, compute_idm_branch=False,
        geometry_architecture='dual_vla_gfm', parallel_vla_gfm_enabled=True,
        parallel_vla_gfm_mode='qwen_tokens', parallel_action_decode_mode='geometry_residual',
        parallel_action_position_mode='decoder_only', parallel_vla_gfm_width=32,
        parallel_vla_gfm_heads=4, parallel_current_geometry_read_enabled=bank,
        parallel_current_depth_enabled=True, depth_decode_enabled=True,
        gradient_checkpointing=True, correction_mode=mode,
        provenance_correction={'uncertainty_layers':[5,9], 'cleanup_kernel':1,'min_hole_area':0})


def forward(net, language=None):
    visual = torch.randn(2,1,1,257,16)
    return net(visual, reference_shallow=visual, lang_feats=(torch.randn(2,5,16) if language is None else language),
        lang_padding_mask=torch.ones(2,5,dtype=torch.bool), action_denormalizer=lambda x:x,
        current_depth_meters=torch.ones(2,224,224)*4)


def test_proposal_is_native_vlm_not_predictor():
    net = model('pure').eval()
    language = torch.randn(2,5,16)
    first = forward(net, language)['direct_actions_norm']
    with torch.no_grad():
        for p in net.predictor.parameters(): p.add_(torch.randn_like(p)*.1)
    second = forward(net, language)['direct_actions_norm']
    torch.testing.assert_close(first, second)
    assert first.shape == (2,1,5,4)
    assert not torch.allclose(second, forward(net, language+torch.randn_like(language))['direct_actions_norm'])


def test_full_stream_shapes_backward_checkpoint_migration():
    pure = model('pure')
    sup = model('support_aware')
    load_checkpoint_state(sup, checkpoint_state(pure), initialize=True)
    out = forward(sup)
    info = out['provenance_correction']
    assert info['rho'].shape == (2,256)
    assert info['delta_sup'].shape == info['delta_unsup'].shape == out['actions_norm'].shape == (2,1,5,4)
    out['actions_norm'].square().mean().backward()
    assert sup.da3.backbone.pretrained.blocks[-1].attn.qkv.weight.grad is not None
    post = model('uncertain')
    load_checkpoint_state(post, checkpoint_state(sup), initialize=True)
    post.train()
    assert any(p.requires_grad for p in post.da3.parameters())
    result = forward(post)
    records = result['provenance_correction']['reliability_records']
    assert set(records) == {5,9}
    assert result['actions_norm'].requires_grad
    assert all(r['uncertainty'].requires_grad for r in records.values())
    assert post.predictor.training and post.provenance_correction.reliability.training
    with pytest.raises(ValueError, match='Legacy'):
        load_checkpoint_state(sup, None, initialize=False)
    load_checkpoint_state(sup, None, initialize=True)
    saved = checkpoint_state(post)
    loaded = model('uncertain')
    load_checkpoint_state(loaded, saved)
    for k,v in post.provenance_correction.state_dict().items():
        torch.testing.assert_close(v, loaded.provenance_correction.state_dict()[k])


def test_pure_legacy_da3_path_unchanged():
    da3 = TinyDA3().eval()
    v, a = torch.randn(2,1,1,257,16), torch.randn(2,1,1,5,16)
    default = da3.propagate_shallow_with_actions_grad(v,a)
    explicit = da3.propagate_shallow_with_actions_grad(v,a,provenance_rho=None)
    torch.testing.assert_close(default['action_tokens'], explicit['action_tokens'])
    torch.testing.assert_close(default['depth'], explicit['depth'])


def test_real_objectives_uncertainty_only_head_gradient_and_joint_training():
    from experiments.uavflow_predictor_idm.objectives import forward_batch
    from experiments.uavflow_predictor_idm.provenance_correction import JointDeepTeacher
    from robot.data.dataset import ActionNormalizer
    class Conditioner(nn.Module):
        hidden_size, is_multimodal, variable_length_tokens, use_reference_image = 16, True, True, False
        def encode_tokens(self, texts, **kwargs):
            b = len(texts)
            return {'last_hidden_state':torch.ones(b,5,16), 'attention_mask':torch.ones(b,5,dtype=torch.bool)}
    net = model('uncertain')
    with torch.no_grad():
        net.parallel_action_head.decoder.output.weight.zero_()
        net.parallel_action_head.decoder.output.bias[:] = torch.tensor([0.,.4,0.,0.])
    net.train()
    net.da3._transport_teacher = JointDeepTeacher(net.da3, [5,9])
    assert net.da3._transport_teacher.da3 is net.da3
    stats = ActionNormalizer({'sim':{'q01':[-.1]*4,'q99':[.1]*4,'mask':[True]*4}},default_key='sim')
    batch = {'all_view_images':torch.rand(2,2,1,3,224,224),
        'episode_first_image':torch.rand(2,1,3,224,224),'task_description':['pass the car']*2,
        'actions':torch.zeros(2,2,5,4),'action_stats_key':['sim']*2,
        'action_loss_mask':torch.ones(2,2,5,4,dtype=torch.bool),
        'gt_depth_meters':torch.full((2,2,1,224,224),10.),
        'gt_depth_mask':torch.ones(2,2,1,224,224,dtype=torch.bool),'start_t':torch.zeros(2,dtype=torch.long)}
    args = dict(model=net,da3=net.da3,text=Conditioner(),normalizer=stats,pose_normalizer=None,
        batch=batch,context_len=1,rollout_steps=1,feature_horizon_weights=[1.],feature_patch_weight=1.,
        feature_cls_weight=0.,feature_register_weight=0.,deep_feature_enabled=False,
        deep_feature_patch_weight=1.,deep_feature_cls_weight=0.,stop_pos_weight=5.,amp=False,
        depth_target_mode='both',depth_scale_mode='fixed_metric',depth_gradient_mode='gam_l1')
    values = forward_batch(**args)
    assert values['provenance_uncertainty_loss'] > 0
    before = {n:p.detach().clone() for n,p in net.named_parameters()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001)
    values['provenance_uncertainty_loss'].backward(); optimizer.step()
    changed = [n for n,p in net.named_parameters() if not torch.equal(before[n],p)]
    assert changed and all(n.startswith('provenance_correction.reliability.') for n in changed)
    # Confidence is active during single-stage training, with no action-loss
    # gradient to reliability. The head's separate loss updates only the head.
    for layer in [5,9]:
        assert values[f'unsupported_read_after_L{layer}'] <= values[f'unsupported_read_before_L{layer}']
    optimizer.zero_grad(set_to_none=True)
    joint = forward_batch(**args)
    (joint['action']+joint['depth']+joint['provenance_uncertainty_loss']).backward()
    assert net.da3.backbone.pretrained.blocks[-1].attn.qkv.weight.grad is not None
    assert any(p.grad is not None for p in net.provenance_correction.reliability.parameters())


def test_post_confidence_one_matches_full_support_policy(monkeypatch):
    support = model('support_aware').eval()
    post = model('uncertain').eval()
    loaded = post.load_state_dict(support.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith('provenance_correction.reliability.') for k in loaded.missing_keys)
    original = post.provenance_correction.reliability.forward
    called = []
    def ones(patches, layer):
        called.append(layer)
        u, c = original(patches, layer)
        return u, torch.ones_like(c)
    monkeypatch.setattr(post.provenance_correction.reliability, 'forward', ones)
    torch.manual_seed(111); reference = forward(support)
    torch.manual_seed(111); result = forward(post)
    torch.testing.assert_close(reference['actions_norm'], result['actions_norm'])
    torch.testing.assert_close(reference['deep_joint_features']['depth'], result['deep_joint_features']['depth'])
    assert called == [5,9]


def test_no_direct_cross_branch_but_within_branch_action_reasoning():
    x, u, rho, block = data()
    s1, u1, _ = routed_block(x, u, block, None, action_count=5, rho=rho)
    s2, _, _ = routed_block(x, u+torch.randn_like(u), block, None, action_count=5, rho=rho)
    torch.testing.assert_close(s1[:,:,1:6], s2[:,:,1:6])
    changed = x.clone(); changed[:,:,1:6] += torch.randn_like(u)
    _, u2, _ = routed_block(changed, u, block, None, action_count=5, rho=rho)
    torch.testing.assert_close(u1, u2)
    within = u.clone(); within[:,:,0] += torch.randn_like(within[:,:,0])
    _, u3, _ = routed_block(x, within, block, None, action_count=5, rho=rho)
    assert not torch.allclose(u1[:,:,1:], u3[:,:,1:])


def test_paired_geometry_numerator_denominator_matches_explicit_formula():
    x, u, rho, block = data()
    u = u+torch.randn_like(u)
    out, _, _ = routed_block(x, u, block, None, action_count=5, rho=rho)
    b, _, n, d = x.shape
    q,k,v = block.attn.qkv(block.norm1(x[:,0])).reshape(b,n,3,4,4).permute(2,0,3,1,4).unbind(0)
    _,uk,uv = block.attn.qkv(block.norm1(u[:,0])).reshape(b,5,3,4,4).permute(2,0,3,1,4).unbind(0)
    eg = (q@k.transpose(-1,-2)/2).exp()
    eu = (q@uk.transpose(-1,-2)/2).exp()
    membership = torch.cat((rho.mean(-1,keepdim=True).expand(-1,7),rho),-1)[:,None,:,None]
    es = eg[...,1:6]*membership
    eu = eu*(1-membership)
    normal = torch.cat((eg[...,:1],eg[...,6:]),-1)
    normal_v = torch.cat((v[...,:1,:],v[...,6:,:]),-2)
    denom = normal.sum(-1,keepdim=True)+es.sum(-1,keepdim=True)+eu.sum(-1,keepdim=True)
    y = (normal@normal_v+es@v[...,1:6,:]+eu@uv)/denom
    expected = x[:,0]+block.attn.proj(y.transpose(1,2).reshape(b,n,d))
    expected = expected+block.mlp(block.norm2(expected))
    torch.testing.assert_close(out[:,0,:1],expected[:,:1])
    torch.testing.assert_close(out[:,0,6:],expected[:,6:])


@pytest.mark.parametrize('r',[0.,1.])
def test_residual_extreme_is_one_shared_head(r):
    head = ParallelContinuousActionHead(input_dim=16,action_dim=4,chunk_size=5)
    s,u = torch.randn(2,1,1,5,16),torch.randn(2,1,1,5,16)
    ds,du,alpha = shared_residual(head,s,u,torch.full((2,16),r))
    torch.testing.assert_close(ds+du,head(s if r else u))
    assert torch.count_nonzero(du if r else ds) == 0
    assert torch.all(alpha+(1-alpha)==1)


def test_rho_detached_and_zero_confidence_backward_finite():
    x,u,_,block = data(); x.requires_grad_(); u.requires_grad_()
    rho = torch.zeros(2,16,requires_grad=True)
    conf = torch.zeros_like(rho,requires_grad=True)
    y,z,_ = routed_block(x,u,block,None,action_count=5,rho=rho,confidence=conf)
    (y.square().mean()+z.square().mean()).backward()
    assert rho.grad is None and conf.grad is None
    assert torch.isfinite(x.grad).all() and torch.isfinite(u.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)


def test_warp_only_detaches_proposal_and_current_depth():
    net = model('support_aware')
    visual = torch.randn(2,1,1,257,16)
    proposal = torch.randn(2,1,5,4,requires_grad=True)*.01
    metric = torch.full((2,224,224),4.,requires_grad=True)
    transport,decoded,_ = net.provenance_correction.support(net,visual,proposal,lambda x:x,metric=metric)
    assert not transport['rho'].requires_grad
    assert decoded['depth'].requires_grad
    final = proposal+torch.ones_like(proposal)
    assert torch.autograd.grad(final.sum(),proposal)[0].abs().sum() > 0
    assert metric.grad is None


def test_three_variants_are_independent_single_stage_configs():
    from experiments.uavflow_predictor_idm.correction_train import build_config
    for mode in ['pure','support_aware','uncertain']:
        cfg = build_config(mode)
        assert cfg.model.correction_mode == mode
        assert cfg.model.provenance_correction.support_detach_depth
        assert not cfg.model.provenance_correction.support_routing.cross_action_branch
    # New uncertainty parameters only; pure -> uncertain is allowed directly.
    load_checkpoint_state(model('uncertain'),checkpoint_state(model('pure')),initialize=True)
    # Pre-refactor pure has identical numerical weights and remains loadable.
    old_pure = checkpoint_state(model('pure'))
    old_pure.pop('routing_version')
    load_checkpoint_state(model('pure'),old_pure)
    old_support = checkpoint_state(model('support_aware'))
    old_support.pop('routing_version')
    with pytest.raises(ValueError,match='average-bridge'):
        load_checkpoint_state(model('support_aware'),old_support)


@pytest.mark.parametrize('mode',['pure','support_aware','uncertain'])
def test_current_bank_all_modes_shared_ca_and_single_current_pass(mode,monkeypatch):
    net = model(mode,bank=True).train()
    calls, reads = [], []
    original = net.da3.propagate_shallow_visual_slots_grad
    def current(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(net.da3,'propagate_shallow_visual_slots_grad',current)
    def record(module,args,kwargs,result):
        reads.append((args[0],args[1],kwargs['layer_index'],result))
    handle = net.current_geometry_read.register_forward_hook(record,with_kwargs=True)
    result = forward(net)
    handle.remove()
    assert len(calls)==1 and calls[0]['return_layer_patches']
    assert result['current_geometry_features']['layer_patches'].keys() == {3,5,7,9}
    assert len(reads)==(4 if mode=='pure' else 8)
    if mode!='pure':
        for first,second in zip(reads[::2],reads[1::2]):
            assert first[1] is second[1]  # same bank, same CA module
            assert first[2]==second[2]
            assert first[0].shape==second[0].shape==(2,1,1,5,16)
    loss = result['actions_norm'].square().mean()
    loss.backward()
    assert net.current_geometry_read.memory[1].weight.grad is not None
    assert net.da3.backbone.pretrained.blocks[-1].attn.qkv.weight.grad is not None
    if mode=='uncertain':
        assert all(p.grad is None for p in net.provenance_correction.reliability.parameters())


def test_current_bank_shared_read_does_not_write_observation_or_couple_queries():
    from experiments.uavflow_predictor_idm.current_geometry import CurrentGeometryRead
    reader = CurrentGeometryRead(16,width=16,heads=4,memory_dim=16,gate_init=.1)
    memory = torch.randn(2,1,1,16,16,requires_grad=True)
    saved = memory.detach().clone()
    sup,uns = torch.randn(2,1,1,5,16),torch.randn(2,1,1,5,16)
    first = reader(sup,memory)
    _ = reader(uns,memory)
    torch.testing.assert_close(first,reader(sup,memory))
    torch.testing.assert_close(memory.detach(),saved)
    first.sum().backward()
    assert memory.grad is not None  # not a frozen-bank detach
