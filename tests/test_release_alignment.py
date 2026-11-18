import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_open_ablation_uses_recorded_raw_instructions_and_split():
    baseline = ROOT / 'docs/experiment_baselines/warp_20261029'
    instruction_path = baseline / 'raw_instructions.json'
    instruction_map = json.loads(instruction_path.read_text())
    split = json.loads((baseline / 'episode_split.json').read_text())
    assert not (set(split['train_episode_ids'] + split['eval_episode_ids']) - set(instruction_map))
    for name in ('vla_future', 'vla_gam'):
        cfg = yaml.safe_load((ROOT / 'experiments/uavflow_paper_ablation/configs' / f'{name}.yaml').read_text())
        assert cfg['instruction_map'] == str(instruction_path.relative_to(ROOT))
        assert not cfg.get('legacy_initialization_reference')
        assert not cfg.get('instruction_map_sha256')
        assert cfg['dataset']['instruction_overrides_path'] == 'data/instruction_overrides.json'
        assert cfg['training']['max_epochs'] == 15
        assert cfg['training']['epoch_weights_only'] is False


def test_indoor_split_snapshot_and_preparation_do_not_use_open_instructions():
    split = json.loads((ROOT / 'docs/experiment_baselines/indooruav/split_ids.json').read_text())
    assert len(split['splits']['train']) == 24288
    assert len(split['splits']['val']) == 1279
    assert not set(split['splits']['train']) & set(split['splits']['val'])
