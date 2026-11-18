import pytest
from experiments.uavflow_paper_ablation.sampling import Warp24MicrogroupSampler


@pytest.mark.parametrize('world', [2, 4, 6, 8])
def test_repartition_matches_three_rank_baseline_and_resume(world):
    data = range(137)
    baseline = [Warp24MicrogroupSampler(data, 3, rank, 42, batch_size=8)
                for rank in range(3)]
    rank_ids = [list(sampler) for sampler in baseline]
    expected = [index for step in range(baseline[0].updates)
                for rank in range(3) for index in rank_ids[rank][step*8:(step+1)*8]]
    samplers = [Warp24MicrogroupSampler(data, world, rank, 42, batch_size=24//world)
                for rank in range(world)]
    actual_rank = [list(sampler) for sampler in samplers]
    per_rank = 24 // world
    actual = [index for step in range(samplers[0].updates)
              for rank in range(world)
              for index in actual_rank[rank][step*per_rank:(step+1)*per_rank]]
    assert actual == expected
    for rank, sampler in enumerate(samplers):
        sampler.resume_batch = 1
        assert list(sampler) == actual_rank[rank][per_rank:]
