"""CPU-only policy tests; these are not NPU platform validation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.verify_uavflow_remote import (
    PINNED_RUNTIME, verify_ascend_versions, verify_runtime,
)


@pytest.mark.parametrize('torch_version', ['2.7.1', '2.7.1+cpu'])
def test_reference_accepts_validated_pair(torch_version):
    verify_ascend_versions(torch_version, '2.7.1.post4')


@pytest.mark.parametrize('versions', [('2.9.0', '2.7.1.post4'), ('2.7.1', '2.7.1.post5')])
def test_reference_rejects_alternative_by_default(versions):
    with pytest.raises(RuntimeError, match='Reference Ascend audit requires'):
        verify_ascend_versions(*versions)


def test_alternative_requires_explicit_compatibility_and_warns(capsys):
    verify_ascend_versions('2.9.0', '2.9.0', 'compatibility')
    warning = capsys.readouterr().out
    assert 'WARNING' in warning
    assert 'operator must confirm' in warning
    assert 'does not establish full model/platform validation' in warning


def test_unknown_policy_fails():
    with pytest.raises(ValueError, match='Unknown Ascend runtime policy'):
        verify_ascend_versions('2.7.1', '2.7.1.post4', 'guess')


def test_compatibility_does_not_bypass_npu_availability():
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
            verify_runtime('npu', 'compatibility')


def test_compatibility_does_not_bypass_project_pins():
    with patch('scripts.verify_uavflow_remote.importlib.metadata.version', return_value='wrong'):
        with pytest.raises(RuntimeError, match='Version drift'):
            verify_runtime('npu', 'compatibility')


def test_compatibility_still_runs_npu_tensor_check():
    bad_probe = SimpleNamespace(sum=lambda: SimpleNamespace(cpu=lambda: 7.0))
    modules = {
        'numpy': SimpleNamespace(), 'scipy': SimpleNamespace(),
        'torch': SimpleNamespace(__version__='2.9.0', device=lambda name: name,
                                 float32='float32', arange=lambda *a, **kw: bad_probe),
        'torch_npu': SimpleNamespace(__version__='2.9.0',
                                    npu=SimpleNamespace(is_available=lambda: True)),
    }
    with patch('scripts.verify_uavflow_remote.importlib.metadata.version',
               side_effect=lambda name: PINNED_RUNTIME[name]), \
         patch('scripts.verify_uavflow_remote.importlib.import_module',
               side_effect=lambda name: modules[name]):
        with pytest.raises(RuntimeError, match='Ascend tensor smoke test failed'):
            verify_runtime('npu', 'compatibility')


def test_matrix_passes_explicit_policy_to_audit():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] /
            'experiments/uavflow_remote_ablation/run_experiment.py').read_text()
    assert '"--ascend-runtime-policy", os.environ.get("ASCEND_RUNTIME_POLICY", "reference")' in text
