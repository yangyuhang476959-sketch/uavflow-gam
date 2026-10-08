"""CPU-only audit regression tests, not NPU platform validation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.verify_uavflow_remote import PINNED_RUNTIME, verify_runtime


def test_npu_availability_is_still_required():
    modules = {
        'numpy': SimpleNamespace(), 'scipy': SimpleNamespace(),
        'torch': SimpleNamespace(__version__='2.9.0'),
        'torch_npu': SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)),
    }
    with patch('scripts.verify_uavflow_remote.importlib.metadata.version',
               side_effect=lambda name: PINNED_RUNTIME[name]), \
         patch('scripts.verify_uavflow_remote.importlib.import_module',
               side_effect=lambda name: modules[name]):
        with pytest.raises(RuntimeError, match='no Ascend NPU is available'):
            verify_runtime('npu')


def test_project_pins_are_still_required():
    with patch('scripts.verify_uavflow_remote.importlib.metadata.version', return_value='wrong'):
        with pytest.raises(RuntimeError, match='Version drift'):
            verify_runtime('npu')


@pytest.mark.parametrize('versions', [('2.7.1', '2.7.1.post4'), ('2.9.0', '2.9.0')])
def test_different_versions_reach_the_actual_tensor_check(versions):
    bad_probe = SimpleNamespace(sum=lambda: SimpleNamespace(cpu=lambda: 7.0))
    modules = {
        'numpy': SimpleNamespace(), 'scipy': SimpleNamespace(),
        'torch': SimpleNamespace(__version__=versions[0], device=lambda name: name,
                                 float32='float32', arange=lambda *a, **kw: bad_probe),
        'torch_npu': SimpleNamespace(__version__=versions[1],
                                    npu=SimpleNamespace(is_available=lambda: True)),
    }
    with patch('scripts.verify_uavflow_remote.importlib.metadata.version',
               side_effect=lambda name: PINNED_RUNTIME[name]), \
         patch('scripts.verify_uavflow_remote.importlib.import_module',
               side_effect=lambda name: modules[name]):
        with pytest.raises(RuntimeError, match='Ascend tensor smoke test failed'):
            verify_runtime('npu')


def test_no_runtime_policy_switch_in_handoff_or_launcher():
    root = Path(__file__).resolve().parents[1]
    for path in ('NPU_SETUP.md', 'scripts/verify_uavflow_remote.py',
                 'experiments/uavflow_remote_ablation/run_experiment.py'):
        text = (root / path).read_text()
        assert 'ASCEND_RUNTIME_POLICY' not in text
        assert '--ascend-runtime-policy' not in text
