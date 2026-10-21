import datetime
from pathlib import Path
import pytest
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.uavflow_predictor_idm.eval_sampler import evaluation_sampler
from experiments.uavflow_predictor_idm.metric_train import build_config
from experiments.uavflow_qwen_pure.train import _cfg

@pytest.mark.parametrize('size,world',[(17,2),(7,3),(2,4),(0,2)])
def test_full_shards_cover_each_example_once_without_padding(size,world):
    ds=list(range(size))
    shards=[list(evaluation_sampler(ds,world=world,rank=r,max_batches=0,shuffle=True,seed=42)) for r in range(world)]
    assert sorted(x for shard in shards for x in shard)==list(range(size))
    for r,shard in enumerate(shards):
        assert len(shard)==len(evaluation_sampler(ds,world=world,rank=r,max_batches=0,shuffle=True,seed=42))
        assert shard==list(evaluation_sampler(ds,world=world,rank=r,max_batches=0,shuffle=True,seed=42))


def test_bounded_monitoring_keeps_distributed_sampler_and_single_rank_is_complete():
    from torch.utils.data.distributed import DistributedSampler
    assert isinstance(evaluation_sampler(range(7),world=2,rank=0,max_batches=2),DistributedSampler)
    assert evaluation_sampler(range(7),world=1,rank=0,max_batches=0) is None


def test_all_five_candidates_use_one_percent_and_no_validation_limit():
    for cfg in [_cfg()]+[build_config(x) for x in ['gam','rgbwarp','rgbwarp_no_action','current_bank']]:
        assert cfg.dataset.eval_ratio==.01
        assert cfg.training.eval_max_batches==0
        assert cfg.training.eval_every==5000
        assert cfg.training.eval_every_epoch is True


class _EvalPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(1,1,bias=False)
        self.encoder=nn.Module();self.encoder.qwen=nn.Identity()
        self.register_buffer('broadcast_probe',torch.ones(1))
        self.linear.weight.data.fill_(1)
    def forward(self,x):return self.linear(x)


def _distributed_eval_worker(rank,init_file,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,world_size=3,rank=rank,
        timeout=datetime.timedelta(seconds=20))
    from experiments.uavflow_qwen_pure import train
    policy=torch.nn.parallel.DistributedDataParallel(_EvalPolicy())
    def forward(model,batch,*args):
        pred=model(batch);target=torch.zeros_like(pred);mask=torch.ones_like(pred,dtype=torch.bool)
        return pred.abs().mean(),pred.detach(),target,mask
    train._forward=forward
    ds=[torch.tensor([float(i)/10]) for i in range(7)]
    sampler=evaluation_sampler(ds,world=3,rank=rank,max_batches=0)
    loader=torch.utils.data.DataLoader(ds,batch_size=2,sampler=sampler)
    result=train._eval(policy,loader,None,torch.device('cpu'),torch.float32,0)
    Path(output+str(rank)).write_text(str(result))
    dist.destroy_process_group()


def test_full_eval_all_reduces_uneven_shards_without_forward_broadcast_deadlock(tmp_path):
    context=mp.spawn(_distributed_eval_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')),nprocs=3,join=False)
    try:
        import time
        deadline=time.monotonic()+45
        while not context.join(timeout=1):
            if time.monotonic()>deadline:raise AssertionError('Uneven full validation deadlocked')
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate();process.join()
    for rank in range(3):assert float((tmp_path/f'rank{rank}').read_text())==pytest.approx(.3,abs=1e-6)


def test_shared_trainer_validation_triggers_cover_periodic_and_epoch_end():
    # Exercise the actual loop predicates, including a checkpoint resumed at an epoch boundary.
    import ast
    from types import SimpleNamespace
    tree=ast.parse(Path('experiments/uavflow_predictor_idm/train.py').read_text())
    predicates={node.targets[0].id:compile(ast.Expression(node.value),'<validation trigger>','eval')
                for node in ast.walk(tree) if isinstance(node,ast.Assign)
                and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in {'periodic_eval_due','epoch_eval_due'}}
    assert len(predicates)==2
    cfg=SimpleNamespace(training=OmegaConf.create(dict(eval_every=5000,eval_every_epoch=True)))
    def due(step,batch,epoch_steps):
        scope=dict(cfg=cfg,step=step,logical_batch_idx=batch,steps_per_epoch=epoch_steps)
        return any(eval(code,scope) for code in predicates.values())
    assert due(150000,1919,18510)
    assert due(166590,18509,18510)
    assert due(185100,18509,18510)
    assert not due(148081,0,18510)
    assert not due(152000,3919,18510)  # old 2000-step schedule is gone
    assert due(10000,4999,5000)  # both triggers request one evaluation
