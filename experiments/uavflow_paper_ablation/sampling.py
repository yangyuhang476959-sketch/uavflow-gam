"""Preserve Warp24 sample order while repartitioning for supported device counts."""
import math
import torch
from torch.utils.data import Sampler


class Warp24MicrogroupSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, seed, resume_epoch=0,
                 resume_batch=0, batch_size=24, **unused):
        if num_replicas not in (1, 2, 3, 4, 6, 8) or not 0 <= rank < num_replicas:
            raise ValueError('Warp24 sampling supports 1, 2, 3, 4, 6 or 8 devices')
        if batch_size != 24 // num_replicas:
            raise ValueError('Warp24 sampling requires effective batch24')
        self.size = len(dataset)
        self.world, self.rank, self.seed = num_replicas, rank, seed
        self.resume_epoch, self.resume_batch = resume_epoch, resume_batch
        self.epoch = 0
        self.updates = math.ceil(self.size / 3) // 8

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        ids = torch.randperm(self.size, generator=generator).tolist()
        padding = 3 * math.ceil(self.size / 3) - self.size
        if padding:
            ids += (ids * math.ceil(padding / len(ids)))[:padding]
        offset = self.resume_batch if self.epoch == self.resume_epoch else 0
        for update in range(offset, self.updates):
            if self.world not in (1, 3):
                # Same global24 samples and order, divided into equal per-rank batches.
                group = [sample for reference_rank in range(3)
                         for sample in ids[update*24+reference_rank:update*24+reference_rank+24:3]]
                per_rank = 24 // self.world
                yield from group[self.rank*per_rank:(self.rank+1)*per_rank]
                continue
            for reference_rank in range(self.rank, 3, self.world):
                begin = update * 24 + reference_rank
                yield from ids[begin:begin + 24:3]

    def __len__(self):
        offset = self.resume_batch if self.epoch == self.resume_epoch else 0
        return max(0, self.updates - offset) * (24 // self.world)

    def micro_seed(self, step, micro):
        if self.world > 3:
            return self.seed + step * self.world + self.rank + micro * self.world
        return self.seed + step * 3 + self.rank + micro * self.world
