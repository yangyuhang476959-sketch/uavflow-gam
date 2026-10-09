"""Keep manual DA3 installation from reintroducing isolated build downloads."""
from pathlib import Path


def test_da3_editable_build_tools_precede_nonisolated_install():
    document = (Path(__file__).resolve().parents[1] / 'NPU_SETUP.md').read_text()
    install = 'python -m pip install --no-build-isolation --no-deps -e ./Depth-Anything-3'
    assert install in document
    for tool in ('hatchling>=1.25', 'hatch-vcs>=0.4', 'editables'):
        assert document.index(tool) < document.index(install)
    assert 'python -m pip install --no-deps -e Depth-Anything-3' not in document
    assert '_install_da3_optional_stubs()' in document
    assert '$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}' in document


def test_yaml_precedes_runtime_import_and_pip_check_precedes_da3():
    document = (Path(__file__).resolve().parents[1] / 'NPU_SETUP.md').read_text()
    assert document.index('pip install --no-deps PyYAML==6.0.2') < document.index("python -c 'import torch, torch_npu")
    assert document.index('python -m pip check') < document.index('git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git')
    assert document.count('python -m pip check') == 1


def test_triton_reference_install_is_isolated_and_disables_dependency_resolution():
    document = (Path(__file__).resolve().parents[1] / 'NPU_SETUP.md').read_text()
    start = document.index('python -m pip install --target "$TRITON_ASCEND_TARGET"')
    end = document.index('\npython -m pip install', start + 1)
    command = document[start:end]
    for required in ('--target "$TRITON_ASCEND_TARGET"', '--ignore-installed',
                     '--no-deps', "'triton-ascend==3.2.1'", 'pybind11'):
        assert required in command
    assert 'pip install -c "$PLATFORM_CONSTRAINTS" \'triton-ascend==3.2.1\'' not in document
    assert '--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple' in document
    assert '--only-binary=:all:' in document
    assert 'export PYTHONPATH="$TRITON_ASCEND_TARGET:$FLA_ASCEND_DIR:' in document
    assert 'print("Triton path:", triton.__file__)' in document
    assert 'Path(triton.__file__).resolve().is_relative_to(' in document
    assert 'import fla' in document
