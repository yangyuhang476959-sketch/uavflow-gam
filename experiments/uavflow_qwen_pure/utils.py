"""Exact sampler seeking and checkpoint RNG helpers, independent of DA3 imports."""
import itertools
import os
import random

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler


def masked_l1(pred,target,mask):
    return ((pred.float()-target).abs()*mask).sum()/mask.sum().clamp_min(1)


def merged_rank_l1(pred,target,mask,replicas):
    """Historical DDP mean-of-rank-means, retained for cursor regression tests."""
    return sum(masked_l1(pred[r::replicas],target[r::replicas],mask[r::replicas])
               for r in range(replicas))/replicas


class CursorSampler(DistributedSampler):
    def __init__(self, *args, batch_size, resume_epoch=0, resume_batch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size, self.resume_epoch, self.resume_batch = batch_size, resume_epoch, resume_batch

    def offset(self):
        return min(self.num_samples, self.resume_batch * self.batch_size) if self.epoch == self.resume_epoch else 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.offset(), None)

    def __len__(self):
        return self.num_samples - self.offset()


class MergedCursorSampler:
    """Interleave the original DDP ranks into one unchanged global batch.

    Retains DistributedSampler padding and the optimizer-step cursor, including
    for odd dataset sizes. Does not require a distributed process group.
    """
    def __init__(self, dataset, *, replicas, batch_size, seed, resume_epoch=0, resume_batch=0):
        self.samplers = [CursorSampler(
            dataset, num_replicas=replicas, rank=rank, shuffle=True, seed=seed,
            batch_size=batch_size, resume_epoch=resume_epoch, resume_batch=resume_batch,
        ) for rank in range(replicas)]
        self.num_samples = sum(s.num_samples for s in self.samplers)

    def set_epoch(self, epoch):
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def __iter__(self):
        return itertools.chain.from_iterable(zip(*(iter(s) for s in self.samplers)))

    def __len__(self):
        return sum(len(s) for s in self.samplers)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], device)


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
