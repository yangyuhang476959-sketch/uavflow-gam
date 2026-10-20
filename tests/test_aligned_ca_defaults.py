"""Check new-comparison CA initialization and the actual optimizer routing."""
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from experiments.uavflow_predictor_idm.current_geometry import CurrentGeometryRead
from experiments.uavflow_predictor_idm.rgbwarp_correction import CorrespondenceRead
from experiments.uavflow_predictor_idm.metric_train import build_config
from experiments.uavflow_predictor_idm import train


def test_correspondence_and_bank_ca_start_with_equal_gate_and_zero_update():
    layers=[19,27,33,39]
    bank=CurrentGeometryRead(8,width=8,heads=2,layer_indices=layers,memory_dim=8,gate_init=1.,zero_init=True)
    warp=CorrespondenceRead(8,layers,width=8,heads=2)
    for module in (bank,warp):
        assert module.gate.item()==1
        assert all(p.item()==0 for p in module.layer_gates.values())
        assert module.output.weight.count_nonzero()==module.output.bias.count_nonzero()==0
    q=torch.randn(2,1,1,5,8);memory=torch.randn(2,1,1,7,8)
    torch.testing.assert_close(bank(q,memory,layer_index=19),q,rtol=0,atol=0)
    bank(q,memory,layer_index=19).sum().backward()
    assert bank.output.weight.grad.abs().sum()>0


@pytest.mark.parametrize('enabled,expected',[(False,1e-5),(True,5e-4)])
def test_actual_optimizer_routes_all_bank_ca_parameters(enabled,expected):
    model=nn.Module()
    model.predictor=nn.Linear(8,8);model.da3=nn.Linear(8,8);model.direct_action_head=nn.Linear(8,4)
    model.current_geometry_read=CurrentGeometryRead(8,width=8,heads=2)
    for name in ('causal_action_decoder','parallel_action_head','parallel_action_correction_head',
                 'transport_policy','provenance_correction','rgbwarp_correction','action_residual_gate_logit'):
        setattr(model,name,None)
    for name in ('stop_head','relative_pose_head','depth_scale_head'):
        setattr(model,name,nn.Linear(8,1).requires_grad_(False))
    cfg=OmegaConf.create({'training':{'lr':5e-5,'predictor_lr_mult':.2,'direct_action_head_lr_mult':10.,'current_geometry_head_lr':enabled}})
    source=inspect.getsource(train.main)
    start=source.index('    trainable = [p for p in model.parameters()')
    end=source.index('    optimizer = torch.optim.AdamW',start)
    scope={'model':model,'raw_model':model,'cfg':cfg}
    exec(compile(textwrap.dedent(source[start:end]),'<actual trainer optimizer routing>','exec'),scope)
    groups=scope['optimizer_groups']
    wanted={id(p) for p in model.current_geometry_read.parameters()}
    rates={group['lr'] for group in groups for p in group['params'] if id(p) in wanted}
    assert rates=={expected}
    assert sum(id(p) in wanted for g in groups for p in g['params'])==len(wanted)


def test_fair_comparison_config_and_legacy_constructor_are_distinct():
    c5=build_config('current_bank')
    assert c5.model.current_geometry_gate_init==1.
    assert c5.model.current_geometry_zero_init
    assert c5.training.current_geometry_head_lr
    assert c5.training.lr*c5.training.direct_action_head_lr_mult==pytest.approx(5e-4)
    for name in ('current_bank','rgbwarp','rgbwarp_no_action'):
        assert build_config(name).model.action_residual_gate_init==.1
    legacy=CurrentGeometryRead(8,width=8,heads=2)
    assert legacy.gate.item()==pytest.approx(1e-3)
    assert legacy.output.weight.count_nonzero()>0


def test_model_constructor_applies_configured_bank_initialization():
    from test_provenance_correction import TinyDA3
    from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
    net=UAVFlowPredictorIDM(da3=TinyDA3(),idm=None,action_dim=4,action_chunk_size=5,
        d_model=32,depth=1,num_heads=4,language_dim=16,language_len=8,
        direct_action_enabled=True,deep_action_enabled=True,compute_idm_branch=False,
        geometry_architecture='dual_vla_gfm',parallel_vla_gfm_enabled=True,
        parallel_vla_gfm_mode='qwen_tokens',parallel_action_decode_mode='geometry_residual',
        parallel_action_position_mode='decoder_only',parallel_vla_gfm_width=32,
        parallel_vla_gfm_heads=4,parallel_current_geometry_read_enabled=True,
        current_geometry_read_mode='per_layer',current_geometry_bank_mode='output_current',
        current_geometry_gate_init=1.,current_geometry_zero_init=True,
        parallel_current_depth_enabled=True,depth_decode_enabled=True,
        correction_mode='pure',action_residual_gate_init=.1)
    assert net.current_geometry_read.gate.item()==1.
    assert net.current_geometry_read.output.weight.count_nonzero()==0
    assert net.action_residual_gate_logit.sigmoid().item()==pytest.approx(.1)
