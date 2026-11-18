"""Preserve the baseline 3-rank x 8 sample groups on one or three GPUs; explicitly repartition global groups on two."""
import math
import torch
from torch.utils.data import Sampler


class Warp24MicrogroupSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, seed, resume_epoch=0,
                 resume_batch=0, batch_size=24, **unused):
        if num_replicas not in (1, 2, 3) or not 0 <= rank < num_replicas:
            raise ValueError('Warp24 sampling supports one, two or three GPUs')
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
            if self.world == 2:
                # Same global24 sample set/order as baseline 3x8. The two local
                # batches are 12, so this does not preserve 8-sample loss groups.
                group = [sample for reference_rank in range(3)
                         for sample in ids[update*24+reference_rank:update*24+reference_rank+24:3]]
                yield from group[self.rank*12:(self.rank+1)*12]
                continue
            for reference_rank in range(self.rank, 3, self.world):
                begin = update * 24 + reference_rank
                yield from ids[begin:begin + 24:3]

    def __len__(self):
        offset = self.resume_batch if self.epoch == self.resume_epoch else 0
        return max(0, self.updates - offset) * (24 // self.world)

    def micro_seed(self, step, micro):
        return self.seed + step * 3 + self.rank + micro * self.world
