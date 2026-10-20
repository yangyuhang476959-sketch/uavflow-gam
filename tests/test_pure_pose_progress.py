"""Verify Pure's seven-state token and optimizer groups without Qwen weights."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.uavflow_qwen_pure import model as pure
from experiments.uavflow_qwen_pure.train import _cfg, _forward


class Tokenizer:
    def __len__(self): return 16
    def add_special_tokens(self,*a): pass
    def convert_tokens_to_ids(self,token): return 7


class Qwen(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(16,8);self.output=nn.Linear(8,16)
        self.adapter=nn.Parameter(torch.zeros(8));self.config=SimpleNamespace(use_cache=True)
    def gradient_checkpointing_enable(self,**k): pass
    def resize_token_embeddings(self,n): pass
    def get_input_embeddings(self): return self.embedding
    def get_output_embeddings(self): return self.output


class Encoder(nn.Module):
    def __init__(self,*a,**k):
        super().__init__();self.qwen=Qwen();self.hidden_size=8
        self.processor=SimpleNamespace(tokenizer=Tokenizer());self.action_placeholder_tokens=['a']*5
    def forward(self,images,instructions,**kwargs):
        assert kwargs['omit_prompt_pose'] and kwargs['state_placeholder']
        ids=torch.tensor([[7,1,2,3,4,5]]).expand(len(instructions),-1)
        e=self.qwen.embedding(ids)
        return {'joint_layers':(e[:,1:]+e[:,:1]+self.qwen.adapter)[:,None]}


def test_seven_dimensional_token_responds_to_progress_and_has_aligned_lrs(monkeypatch):
    monkeypatch.setattr(pure,'FrozenQwen35SemanticEncoder',Encoder)
    net=pure.PureQwenActionPolicy('unused',decoder_width=8,pose_mode='numeric_token',pose_progress_enabled=True)
    assert net.state_encoder[0].in_features==7
    torch.testing.assert_close(net.state_scale,torch.tensor([10.,10.,10.,1.,1.,2*torch.pi,10.]))
    state=torch.tensor([[0.,0.,0.,0.,1.,0.,0.]])
    changed=state.clone();changed[0,-1]=5.
    a=net(None,['Approach'],state);b=net(None,['Approach'],changed)
    assert not torch.allclose(a,b)
    b.sum().backward();assert net.state_encoder[0].weight.grad.abs().sum()>0
    assert not net.encoder.qwen.embedding._forward_hooks
    groups=net.optimizer_groups(1e-5,5e-4,state_lr=1e-5)
    assert [g['lr'] for g in groups]==[1e-5,5e-4,1e-5]
    assert {id(p) for p in groups[2]['params']}=={id(p) for p in net.state_encoder.parameters()}
    assert len([p for g in groups for p in g['params']])==len({id(p) for g in groups for p in g['params']})
    with pytest.raises(ValueError,match=r'\[B,7\]'):net(None,['Approach'],torch.zeros(1,4))


def test_training_dispatch_uses_progress_not_legacy_pose():
    class Policy(nn.Module):
        pose_progress_enabled=True
        def forward(self,images,instructions,states):
            torch.testing.assert_close(states,torch.arange(7).float()[None])
            return torch.zeros(1,1,5,4)
    batch=dict(all_view_images=torch.zeros(1,2,1,3,2,2),task_description=['Approach'],
               episode_pose_progress=torch.arange(14).reshape(1,2,7).float(),
               openvla_prompt_pose=torch.full((1,2,4),999.),actions=torch.zeros(1,2,5,4),
               action_stats_key=['sim'],action_loss_mask=torch.ones(1,2,5,4,dtype=torch.bool))
    normalizer=SimpleNamespace(normalize=lambda value,**kwargs:value)
    assert _forward(Policy(),batch,normalizer,torch.device('cpu'),False,torch.bfloat16)[0]==0


def test_pure_config_has_same_progress_input_and_state_lr_as_corrections():
    from experiments.uavflow_predictor_idm.metric_train import build_config
    cfg=_cfg()
    assert cfg.dataset.pose_progress_enabled and cfg.model.pose_progress_enabled
    assert cfg.stage1.qwen_pose_progress_enabled and cfg.stage1.qwen_numeric_pose_token
    for variant in ['rgbwarp','rgbwarp_no_action','current_bank']:
        other=build_config(variant)
        assert cfg.training.lora_lr==pytest.approx(other.training.lr*other.training.predictor_lr_mult)
        assert cfg.training.state_lr==pytest.approx(other.training.lr*other.training.predictor_lr_mult)
        assert cfg.dataset.action_delta_mode==other.dataset.action_delta_mode=='openvla_yaw_only'
