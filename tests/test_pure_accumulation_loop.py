"""Run the real Pure trainer loop with a tiny CPU policy and uneven tail."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import pytest
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from experiments.uavflow_qwen_pure import train


class TinyDataset(Dataset):
    def __len__(self): return 10
    def __getitem__(self,index):
        return {'x':torch.tensor(float(index+1)/10),
                'actions':torch.full((1,5,4),float(index)/20),
                'action_loss_mask':torch.ones(1,5,4,dtype=torch.bool)}


class TinyPolicy(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.encoder=nn.Module();self.encoder.qwen=nn.Linear(1,1)
        self.projection=nn.Linear(1,1);self.head=nn.Linear(1,1);self.state_encoder=nn.Linear(1,1)
    def trainable_state(self): return {k:v.detach().clone() for k,v in self.state_dict().items()}
    def load_trainable_state(self,state): self.load_state_dict(state)


@pytest.mark.parametrize("eval_every", [0, 1, 2])
@pytest.mark.parametrize("progress", [False, True])
def test_real_pure_loop_accumulates_tail_and_resumes_exact_cursor(tmp_path,monkeypatch,progress,eval_every):
    normalizer=SimpleNamespace(state_dict=lambda:{'test':True})
    monkeypatch.setattr(train,'build_robot_dataset',lambda *a,**k:TinyDataset())
    monkeypatch.setattr(train,'ActionNormalizer',SimpleNamespace(from_stats_dir=lambda *a:normalizer))
    monkeypatch.setattr(train,'PureQwenActionPolicy',TinyPolicy)
    monkeypatch.setattr(train,'distributed_info',lambda:(False,0,0,1))
    monkeypatch.setattr(train,'configure_accelerator',lambda **k:SimpleNamespace(device=torch.device('cpu'),kind='cpu'))
    monkeypatch.setattr(train,'manual_seed_all',lambda *a:None)
    def forward(model,batch,*args):
        weight=sum(p.sum() for p in model.parameters())
        prediction=(weight*batch['x'][:,None,None]).expand(-1,5,4)
        target=batch['actions'][:,0];mask=batch['action_loss_mask'][:,0]
        loss=(prediction-target).abs().mean()
        return loss,prediction.detach(),target,mask
    monkeypatch.setattr(train,'_forward',forward)
    evaluations=[]
    real_eval=train._eval
    def evaluate(*args):
        evaluations.append(True)
        return real_eval(*args)
    monkeypatch.setattr(train,'_eval',evaluate)
    def config(_):
        return OmegaConf.create(dict(seed=42,model={'pose_progress_enabled':progress},dataset={'action_stats_key':'sim'},stage1={'action_stats_dir':'unused'},training=dict(batch_size=3,gradient_accumulation_steps=2,max_epochs=1,lr_schedule='constant',warmup_steps=0,lora_lr=1e-3,head_lr=1e-3,num_workers=0,weight_decay=0,log_every=1,save_every=1,eval_every=eval_every,eval_max_batches=0,results_dir=str(tmp_path/'full'))))
    monkeypatch.setattr(train,'_cfg',config)
    original_save=train.atomic_save
    def save(payload,path):
        original_save(payload,path)
        if payload['epoch']==0 and payload['step']==1:
            original_save(payload,tmp_path/'mid.pt')
    monkeypatch.setattr(train,'atomic_save',save)
    monkeypatch.setattr(sys,'argv',['pure'])
    train.main()
    assert len(evaluations)==(2 if eval_every==1 else 1)
    evaluations.clear()
    full=torch.load(tmp_path/'full/last.pt',weights_only=False)
    mid=torch.load(tmp_path/'mid.pt',weights_only=False)
    assert (mid['step'],mid['epoch'],mid['batch'],mid['gradient_accumulation_steps'])==(1,0,1,2)
    assert full['step']==2  # update sizes6 and4; the final1-sample microbatch is retained
    monkeypatch.setattr(sys,'argv',['pure','--resume',str(tmp_path/'mid.pt'),'--set',f'training.results_dir={tmp_path / "resumed"}'])
    train.main()
    assert len(evaluations)==1  # periodic + epoch-end at the same step run once
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert (resumed['step'],resumed['epoch'],resumed['batch'])==(2,1,0)
    for name,value in full['trainable_state'].items():
        torch.testing.assert_close(value,resumed['trainable_state'][name],rtol=0,atol=0)

    if not progress:
        monkeypatch.setattr(sys,'argv',['pure','--resume',str(tmp_path/'mid.pt'),'--set','model.pose_progress_enabled=true'])
        with pytest.raises(ValueError,match='4D pose and 7D pose-progress'):
            train.main()
