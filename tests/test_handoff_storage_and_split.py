"""Offline regression coverage for selected storage and stratified smoke splits."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot.data.uavflow_dataset import _stratified_episode_split, _UAVFLOW_TASK_CLASSES
from scripts import download_uavflow_assets as assets


INSTRUCTIONS = [
    'Approach the tree', 'Move away from the tree', 'Pass through the gate',
    'Land at the tree', 'Turn to the right', 'Move to a position 4 meters from the tree',
    'Move 4 meters at 30 degrees', 'Rotate 30 degrees', 'Circle around the tree',
    'Ascend 2 meters',
]


def episodes(sizes):
    ids, logs = [], {}
    for category, size in enumerate(sizes):
        for index in range(size):
            tid = f'{category:02d}_{index:04d}'
            ids.append(tid)
            logs[tid] = ([], INSTRUCTIONS[category])
    return ids, logs


def test_twenty_trajectory_smoke_has_nineteen_train_one_validation():
    ids, logs = episodes([2] * 10)
    train, val, counts = _stratified_episode_split(ids, logs, .05, 42)
    assert len(train) == 19 and len(val) == 1
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(ids)
    assert sum(c['eval'] for c in counts.values()) == 1
    assert (train, val, counts) == _stratified_episode_split(ids, logs, .05, 42)


def test_feasible_multicategory_allocation_preserves_historical_ids():
    sizes = [20] * 10
    ids, logs = episodes(sizes)
    train, val, counts = _stratified_episode_split(ids, logs, .05, 42)
    # Historical feasible quota: exactly one per class; same RNG/category order.
    rng = np.random.default_rng(42)
    expected = []
    for category, size in enumerate(sizes):
        pool = np.asarray([f'{category:02d}_{i:04d}' for i in range(size)], dtype=object)
        expected.append(pool[rng.permutation(size)][0])
    assert set(val) == set(expected)
    assert len(train) == 190 and len(val) == 10
    assert all(counts[c]['eval'] == 1 for c in _UAVFLOW_TASK_CLASSES)
    assert not set(train) & set(val)


def test_proportional_multicategory_allocation_and_disjointness():
    ids, logs = episodes([20, 40, 60] + [0] * 7)
    train, val, counts = _stratified_episode_split(ids, logs, .2, 42)
    assert [counts[c]['eval'] for c in _UAVFLOW_TASK_CLASSES[:3]] == [4, 8, 12]
    assert len(val) == 24 and len(train) == 96
    assert not set(train) & set(val)


@pytest.mark.parametrize('custom', [False, True])
def test_download_storage_paths_without_network(tmp_path, monkeypatch, custom):
    repo = tmp_path / 'repository'
    repo.mkdir()
    data = tmp_path / 'chosen-data' if custom else repo / 'data_remote'
    models = tmp_path / 'chosen-models' if custom else repo / 'checkpoints'
    snapshots, transfers = [], []

    def snapshot(repo_id, **kwargs):
        snapshots.append((repo_id, kwargs['local_dir']))

    def download(repo_id, filename, **kwargs):
        path = Path(kwargs['local_dir']) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fake checkpoint')
        return str(path)

    monkeypatch.setattr(assets, 'ROOT', repo)
    monkeypatch.setattr(assets.subprocess, 'run', lambda command, **kwargs: transfers.append(command))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        hf_hub_download=download, snapshot_download=snapshot))
    arguments = ['download_uavflow_assets.py']
    if custom:
        arguments += ['--data-root', str(data), '--model-root', str(models)]
    monkeypatch.setattr(sys, 'argv', arguments)
    assets.main()
    assert (models / 'track4world_da3.pth').read_bytes() == b'fake checkpoint'
    assert snapshots == [
        ('wangxiangyu0814/UAV-Flow-Sim', data / 'UAV-Flow-Sim'),
        ('Qwen/Qwen3.5-2B', models / 'qwen3.5-2b'),
        ('google-t5/t5-base', models / 't5-base'),
    ]
    assert str(data / 'UAV-Flow-Sim-Depth') in transfers[-1]
    if custom:
        assert not (repo / 'checkpoints/track4world_da3.pth').exists()
