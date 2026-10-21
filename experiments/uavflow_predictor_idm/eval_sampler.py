"""Distributed validation without padding or duplicate examples."""
import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class ExactDistributedEvalSampler(Sampler):
    """Each example belongs to exactly one rank; shard lengths may differ."""
    def __init__(self, dataset, num_replicas, rank, shuffle=False, seed=0):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError('Invalid evaluation world size or rank')
        self.dataset=dataset;self.num_replicas=num_replicas;self.rank=rank
        self.shuffle=shuffle;self.seed=seed

    def __iter__(self):
        n=len(self.dataset)
        indices=(torch.randperm(n,generator=torch.Generator().manual_seed(self.seed)).tolist()
                 if self.shuffle else range(n))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        return len(range(self.rank,len(self.dataset),self.num_replicas))


def evaluation_sampler(dataset, *, world, rank, max_batches, shuffle=False, seed=0):
    if world == 1:
        return None
    cls=ExactDistributedEvalSampler if max_batches <= 0 else DistributedSampler
    return cls(dataset,num_replicas=world,rank=rank,shuffle=shuffle,seed=seed)
