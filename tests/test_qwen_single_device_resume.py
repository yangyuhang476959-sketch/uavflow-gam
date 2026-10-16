import pytest
import torch
from torch.utils.data import DataLoader

from experiments.uavflow_qwen_pure.utils import CursorSampler, MergedCursorSampler
from experiments.uavflow_qwen_pure.utils import masked_l1, merged_rank_l1


@pytest.mark.parametrize('size', [101, 127, 128, 129])
@pytest.mark.parametrize('cursor', [0, 2])
def test_same_global_batches_padding_and_cursor(size, cursor):
    data = list(range(size))
    ranks = [CursorSampler(data, num_replicas=2, rank=r, seed=42, shuffle=True,
                          batch_size=16, resume_epoch=5, resume_batch=cursor) for r in range(2)]
    merged = MergedCursorSampler(data, replicas=2, seed=42, batch_size=16,
                                resume_epoch=5, resume_batch=cursor)
    for sampler in [*ranks, merged]:
        sampler.set_epoch(5)
    expected = [torch.stack((a, b), dim=1).flatten()
                for a, b in zip(*(DataLoader(data, sampler=s, batch_size=16, drop_last=True) for s in ranks))]
    actual = list(DataLoader(data, sampler=merged, batch_size=32, drop_last=True))
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert merged.num_samples // 32 == ranks[0].num_samples // 16


def test_merged_loss_preserves_ddp_rank_means_and_gradient():
    pred = torch.arange(32., requires_grad=True)[:, None]
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    mask[:20:2] = 0
    loss = merged_rank_l1(pred, target, mask, 2)
    expected = (masked_l1(pred[::2], target[::2], mask[::2]) +
                masked_l1(pred[1::2], target[1::2], mask[1::2])) / 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, pred, retain_graph=True)[0],
                               torch.autograd.grad(expected, pred)[0])
