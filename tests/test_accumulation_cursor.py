"""Portable CPU checks for accumulation grouping and exact sample seeking."""
import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from experiments.uavflow_predictor_idm.runtime import optimizer_batches
from experiments.uavflow_predictor_idm.train import CursorDistributedSampler


def test_grouping_preserves_order_and_discards_only_incomplete_update():
    assert list(optimizer_batches(range(7), 2)) == [[0, 1], [2, 3], [4, 5]]
    assert list(optimizer_batches(range(3), 1)) == [[0], [1], [2]]
    assert list(optimizer_batches(range(3), 2, drop_incomplete=False)) == [[0, 1], [2]]
    with pytest.raises(ValueError):
        list(optimizer_batches(range(3), 0))


@pytest.mark.parametrize('rank', [0, 1])
def test_effective_batch_cursor_matches_remaining_original_sampler(rank):
    data = range(101)
    full = DistributedSampler(data, num_replicas=2, rank=rank, shuffle=True, seed=42)
    full.set_epoch(2)
    expected = list(full)
    # Three completed updates of 4 samples * 2 microbatches per rank.
    for micro, accumulation in [(4, 2), (8, 1)]:
        sampler = CursorDistributedSampler(data, num_replicas=2, rank=rank,
            shuffle=True, seed=42, resume_epoch=2, resume_batch=3,
            batch_size=micro * accumulation)
        sampler.set_epoch(2)
        assert list(sampler) == expected[24:]
        sampler.set_epoch(3)
        full.set_epoch(3)
        assert list(sampler) == list(full)
        full.set_epoch(2)


def test_accumulated_update_matches_effective_full_batch():
    torch.manual_seed(42)
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    reference = torch.nn.Linear(3, 2)
    accumulated = torch.nn.Linear(3, 2)
    accumulated.load_state_dict(reference.state_dict())
    first = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    second = torch.optim.AdamW(accumulated.parameters(), lr=1e-3)
    torch.nn.functional.mse_loss(reference(x), y).backward()
    batches = [(x[:4], y[:4]), (x[4:], y[4:])]
    for group in optimizer_batches(batches, 2):
        for xb, yb in group:
            (torch.nn.functional.mse_loss(accumulated(xb), yb) / len(group)).backward()
    first.step(); second.step()
    for a, b in zip(reference.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b)
