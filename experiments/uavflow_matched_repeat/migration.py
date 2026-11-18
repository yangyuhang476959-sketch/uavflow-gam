"""Explicit 2x8x2 -> 4x8x1 migration, preserving the four original microbatches."""
import math
import torch
from torch.utils.data import Sampler


def validate(saved, current, saved_world, world, step, epoch, batch, size):
    def get(c,k): return c['training'][k]
    assert world == 4 and get(current,'batch_size') == 8 and get(current,'gradient_accumulation_steps') == 1
    assert saved_world in (2,4)
    assert get(saved,'batch_size') == 8
    assert get(saved,'gradient_accumulation_steps') == (2 if saved_world == 2 else 1)
    if saved_world == 4:
        assert saved['training'].get('preserve_two_rank_microgroups') is True
    updates = math.ceil(size / 2) // 8 // 2
    assert updates == math.ceil(size / 4) // 8
    assert step == epoch * updates + batch, (step,epoch,batch,updates)
    if saved_world == 2:
        assert batch == 0, 'Initial migration requires an epoch boundary'
    import copy
    before, after = copy.deepcopy(saved), copy.deepcopy(current)
    for config in (before,after):
        for key in ('gradient_accumulation_steps','epoch_weights_only','preserve_two_rank_microgroups'):
            config['training'].pop(key,None)
    if before != after: raise ValueError('Migration may only change accumulation and storage policy')
    return updates


class PreservedMicrogroupSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, seed, resume_epoch=0, resume_batch=0, **unused):
        assert num_replicas == 4 and 0 <= rank < 4
        self.size=len(dataset); self.rank=rank; self.seed=seed; self.epoch=0
        self.resume_epoch=resume_epoch; self.resume_batch=resume_batch
        self.updates=math.ceil(self.size/2)//16
    def set_epoch(self, epoch): self.epoch=epoch
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        ids=torch.randperm(self.size,generator=g).tolist()
        # Match DistributedSampler's original two-rank padding exactly.
        ids += ids[:2*math.ceil(self.size/2)-self.size]
        offset=self.resume_batch if self.epoch==self.resume_epoch else 0
        for update in range(offset,self.updates):
            begin=update*32+(self.rank//2)*16
            yield from ids[begin+self.rank%2:begin+16:2]
    def __len__(self):
        return (self.updates-(self.resume_batch if self.epoch==self.resume_epoch else 0))*8
    def micro_seed(self, step, micro):
        assert micro == 0
        # Identical seed to the corresponding original rank/accumulation slot.
        return self.seed+((step-1)*2+self.rank//2+1)*2+self.rank%2
