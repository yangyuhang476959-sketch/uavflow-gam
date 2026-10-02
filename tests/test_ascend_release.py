from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.uavflow_predictor_idm.runtime import (
    env_truthy,
    grad_scaler_enabled,
)
from experiments.uavflow_remote_ablation.run_experiment import (
    benchmark_missing_checkpoint_allowed,
    python_interpreter,
)
from robot.modeling.lora import LoRALinear
from scripts.ascend_bootstrap import (
    MINICONDA_BASE_URL,
    PYTORCH_CPU_INDEX,
    REFERENCE_CANN,
    cann_installer_url,
    cann_installer_name,
    cann_ops_installer_url,
    cann_ops_installer_name,
    detect_cann_version,
    detect_ops_version,
    find_cached_cann_installer,
    is_python311,
    miniconda_installer_name,
    miniconda_installer_url,
    normalize_arch,
    online_torch_commands,
    validate_local_wheel,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_env_truthy_true(monkeypatch, value):
    monkeypatch.setenv("UAVFLOW_TEST_FLAG", value)
    assert env_truthy("UAVFLOW_TEST_FLAG")


@pytest.mark.parametrize("value", ["0", "false", "NO", "off", ""])
def test_env_truthy_false(monkeypatch, value):
    monkeypatch.setenv("UAVFLOW_TEST_FLAG", value)
    assert not env_truthy("UAVFLOW_TEST_FLAG")


def test_env_truthy_rejects_typo(monkeypatch):
    monkeypatch.setenv("UAVFLOW_TEST_FLAG", "maybe")
    with pytest.raises(ValueError):
        env_truthy("UAVFLOW_TEST_FLAG")


@pytest.mark.parametrize(
    "policy,dtype,expected",
    [("auto", "bf16", False), ("auto", "fp16", True),
     ("on", "bf16", True), ("off", "fp16", False)],
)
def test_grad_scaler_policy(policy, dtype, expected):
    assert grad_scaler_enabled(
        "npu", amp_enabled=True, dtype_name=dtype, policy=policy
    ) is expected


def test_grad_scaler_disabled_amp_and_invalid_policy():
    assert not grad_scaler_enabled(
        "cuda", amp_enabled=False, dtype_name="fp16", policy="auto"
    )
    with pytest.raises(ValueError, match="requires AMP"):
        grad_scaler_enabled(
            "cuda", amp_enabled=False, dtype_name="bf16", policy="on"
        )
    with pytest.raises(ValueError, match="auto, on, or off"):
        grad_scaler_enabled(
            "cuda", amp_enabled=True, dtype_name="fp16", policy="typo"
        )


def test_python_interpreter_preserves_venv_symlink(tmp_path):
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    selected = python_interpreter(str(venv_python))
    assert selected == venv_python
    assert selected.is_symlink()


def test_checkpoint_skip_is_benchmark_stage1_only():
    assert benchmark_missing_checkpoint_allowed(
        stage="stage1", requested_stage="stage1", skip_final_checkpoint=True
    )
    assert not benchmark_missing_checkpoint_allowed(
        stage="stage1", requested_stage="both", skip_final_checkpoint=True
    )
    assert not benchmark_missing_checkpoint_allowed(
        stage="stage2", requested_stage="stage2", skip_final_checkpoint=True
    )
    assert not benchmark_missing_checkpoint_allowed(
        stage="stage1", requested_stage="stage1", skip_final_checkpoint=False
    )


def test_train_imports_profile_phase_used_by_backward_scopes():
    source = (ROOT / "experiments/uavflow_predictor_idm/train.py").read_text()
    import_block = source.split(")\n", 1)[0]
    assert "profile_phase" in import_block
    assert 'with profile_phase("R1/backward")' in source
    assert 'with profile_phase("R1/optimizer")' in source


def test_qwen_module_import_does_not_require_ascend(monkeypatch):
    monkeypatch.delenv("UAVFLOW_QWEN_FLA_NPU", raising=False)
    before = set(sys.modules)
    module = importlib.import_module(
        "experiments.uavflow_direct_visual_probe.qwen35_semantic"
    )
    assert hasattr(module, "FrozenQwen35SemanticEncoder")
    newly_loaded = set(sys.modules) - before
    assert "torch_npu" not in newly_loaded
    assert "fla" not in newly_loaded


def test_qwen_fla_patch_is_lazy_and_preserves_layout(monkeypatch):
    module = importlib.import_module(
        "experiments.uavflow_direct_visual_probe.qwen35_semantic"
    )
    monkeypatch.setenv("UAVFLOW_ACCELERATOR", "npu")
    monkeypatch.setenv("UAVFLOW_QWEN_FLA_NPU", "1")
    calls = {}

    def fake_conv(x, weight, **kwargs):
        calls["conv_shape"] = tuple(x.shape)
        return x

    def fake_gdr(q, k, v, **kwargs):
        calls["qk_norm"] = kwargs["use_qk_l2norm_in_kernel"]
        return q, None

    fake_fla = types.ModuleType("fla")
    fake_modules = types.ModuleType("fla.modules")
    fake_conv_module = types.ModuleType("fla.modules.convolution")
    fake_conv_module.causal_conv1d = fake_conv
    fake_ops = types.ModuleType("fla.ops")
    fake_gdr_module = types.ModuleType("fla.ops.gated_delta_rule")
    fake_gdr_module.chunk_gated_delta_rule = fake_gdr
    for name, value in {
        "fla": fake_fla,
        "fla.modules": fake_modules,
        "fla.modules.convolution": fake_conv_module,
        "fla.ops": fake_ops,
        "fla.ops.gated_delta_rule": fake_gdr_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    layers = [
        types.SimpleNamespace(linear_attn=types.SimpleNamespace())
        for _ in range(18)
    ]
    qwen = types.SimpleNamespace(
        config=types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                layer_types=["linear_attention"] * 18
            )
        ),
        model=types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=layers)
        )
    )
    assert module._patch_qwen_fla_npu(qwen) == 18
    x = torch.randn(2, 7, 11)
    result = layers[0].linear_attn.causal_conv1d_fn(
        x=x, weight=torch.randn(7, 3), activation="silu"
    )
    assert calls["conv_shape"] == (2, 11, 7)
    assert tuple(result.shape) == tuple(x.shape)
    q = torch.randn(2, 3, 4, 5)
    layers[0].linear_attn.chunk_gated_delta_rule(
        q, q, q, g=q[..., 0], beta=q[..., 0],
        use_qk_l2norm_in_kernel=True,
    )
    assert calls["qk_norm"] is True


def test_qwen_fla_patch_rejects_partial_coverage(monkeypatch):
    module = importlib.import_module(
        "experiments.uavflow_direct_visual_probe.qwen35_semantic"
    )
    monkeypatch.setenv("UAVFLOW_ACCELERATOR", "npu")
    monkeypatch.setenv("UAVFLOW_QWEN_FLA_NPU", "1")
    fake_conv = types.ModuleType("fla.modules.convolution")
    fake_conv.causal_conv1d = lambda x, weight, **kwargs: x
    fake_gdr = types.ModuleType("fla.ops.gated_delta_rule")
    fake_gdr.chunk_gated_delta_rule = lambda q, k, v, **kwargs: (q, None)
    for name, value in {
        "fla": types.ModuleType("fla"),
        "fla.modules": types.ModuleType("fla.modules"),
        "fla.modules.convolution": fake_conv,
        "fla.ops": types.ModuleType("fla.ops"),
        "fla.ops.gated_delta_rule": fake_gdr,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    layers = [types.SimpleNamespace(linear_attn=types.SimpleNamespace()) for _ in range(17)]
    qwen = types.SimpleNamespace(
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(
            layer_types=["linear_attention"] * 18)),
        model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=layers)),
    )
    with pytest.raises(RuntimeError, match="patched 17 of 18"):
        module._patch_qwen_fla_npu(qwen)


def test_cann_discovery_rejects_multiple_installations(tmp_path):
    import subprocess
    for name in ("cann-a", "cann-b"):
        path = tmp_path / name / "bin" / "set_env.sh"
        path.parent.mkdir(parents=True)
        path.write_text("#!/usr/bin/env bash\n")
    command = (
        f"source {ROOT / 'scripts/ascend_cann.sh'}; "
        "uavflow_select_cann_env"
    )
    result = subprocess.run(
        ["bash", "-c", command], text=True, capture_output=True,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "ASCEND_SEARCH_ROOT": str(tmp_path),
            "CANN_ROOT": "",
        },
    )
    assert result.returncode != 0
    assert "Multiple CANN installations" in result.stderr


def test_cann_discovery_accepts_one_installation(tmp_path):
    import subprocess

    selected = tmp_path / "search/cann/set_env.sh"
    selected.parent.mkdir(parents=True)
    selected.write_text("#!/usr/bin/env bash\n")
    result = subprocess.run(
        ["bash", "-c", f"source {ROOT / 'scripts/ascend_cann.sh'}; uavflow_select_cann_env"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "ASCEND_SEARCH_ROOT": str(tmp_path / "search"),
            "CANN_ROOT": "",
        },
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == selected.resolve()


def test_setup_has_no_cann_version_guessing():
    sources = "\n".join(
        (ROOT / path).read_text()
        for path in ("scripts/setup_ascend_cluster.sh", "scripts/setup_ascend_cann.sh")
    )
    assert "grep -R" not in sources
    assert "cann-version" in sources


@pytest.mark.parametrize(
    "value,expected", [("aarch64", "aarch64"), ("arm64", "aarch64"),
                       ("x86_64", "x86_64"), ("amd64", "x86_64")]
)
def test_supported_architecture_detection(value, expected):
    assert normalize_arch(value) == expected


def test_unsupported_architecture_rejected():
    with pytest.raises(ValueError, match="Unsupported CPU architecture"):
        normalize_arch("ppc64le")


@pytest.mark.parametrize("version", ["3.11", "3.11.0", "3.11.14", "3.11.16"])
def test_python311_accepts_any_patch(version):
    assert is_python311(version)


@pytest.mark.parametrize("version", ["3.10.14", "3.12.0", "3.11rc1"])
def test_python311_rejects_other_versions(version):
    assert not is_python311(version)


def test_cann_official_metadata_parser(tmp_path):
    metadata = tmp_path / "ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "package_name=Ascend-cann-toolkit\nversion=9.0.0\n"
        "arch=aarch64\nos=linux\npath=/example\n"
    )
    version, selected = detect_cann_version(tmp_path, arch="aarch64")
    assert version == "9.0.0"
    assert selected == metadata


def test_cann_metadata_parser_rejects_wrong_architecture(tmp_path):
    metadata = tmp_path / "x86_64-linux/ascend_toolkit_install.info"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "package_name=Ascend-cann-toolkit\nversion=9.0.0\narch=x86_64\n"
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        detect_cann_version(tmp_path, arch="aarch64")


def test_cann_installer_cache_and_architecture(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    installer = cache / cann_installer_name("aarch64")
    installer.touch()
    assert find_cached_cann_installer([cache], arch="aarch64") == installer
    assert installer.name == "Ascend-cann-toolkit_9.0.0_linux-aarch64.run"
    assert REFERENCE_CANN == "9.0.0"


def test_official_cann_toolkit_and_910b_ops_names():
    assert cann_installer_name("aarch64") == "Ascend-cann-toolkit_9.0.0_linux-aarch64.run"
    assert cann_ops_installer_name("aarch64") == "Ascend-cann-910b-ops_9.0.0_linux-aarch64.run"
    assert cann_ops_installer_name("x86_64").endswith("linux-x86_64.run")
    assert cann_installer_url("aarch64") == (
        "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/"
        "CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run"
    )
    assert cann_ops_installer_url("x86_64").endswith(
        "/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run"
    )


def test_cann_ops_official_metadata_parser(tmp_path):
    metadata = tmp_path / "cann/aarch64-linux/ascend_ops_install.info"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "package_name=Ascend-cann-910b-ops\nversion=9.0.0\narch=aarch64\n"
    )
    version, selected, package = detect_ops_version(tmp_path, arch="aarch64")
    assert version == "9.0.0"
    assert selected == metadata
    assert package == "ascend-cann-910b-ops"


def test_cann_ops_metadata_rejects_wrong_chip(tmp_path):
    metadata = tmp_path / "cann/aarch64-linux/ascend_ops_install.info"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "package_name=Ascend-cann-310p-ops\nversion=9.0.0\narch=aarch64\n"
    )
    with pytest.raises(RuntimeError, match="910B"):
        detect_ops_version(tmp_path, arch="aarch64")


@pytest.mark.parametrize(
    "arch,suffix",
    [("aarch64", "Miniconda3-latest-Linux-aarch64.sh"),
     ("x86_64", "Miniconda3-latest-Linux-x86_64.sh")],
)
def test_miniconda_selection_uses_official_anaconda_source(arch, suffix):
    assert miniconda_installer_name(arch) == suffix
    assert miniconda_installer_url(arch) == f"{MINICONDA_BASE_URL}/{suffix}"


def test_online_torch_plan_uses_authoritative_indexes():
    commands = online_torch_commands("/env/bin/python")
    assert commands[0][-1] == PYTORCH_CPU_INDEX
    assert "torch==2.7.1" in commands[0]
    assert "torch-npu==2.7.1.post4" in commands[1]
    assert commands[1][-1] == "https://pypi.org/simple"
    assert "torchvision==0.22.1" in commands[2]
    assert commands[2][-1] == PYTORCH_CPU_INDEX


def test_local_wheel_validation_is_arch_and_python_aware(tmp_path):
    wheel = tmp_path / "torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl"
    wheel.touch()
    validate_local_wheel(wheel, arch="aarch64")
    with pytest.raises(ValueError, match="architecture"):
        validate_local_wheel(wheel, arch="x86_64")


def test_ascend_stack_modes_and_benchmark_are_explicit():
    setup = (ROOT / "scripts/setup_ascend_cann.sh").read_text()
    project = (ROOT / "constraints-ascend.txt").read_text()
    reference = (ROOT / "constraints-ascend-reference.txt").read_text()
    bench = (ROOT / "scripts/bench_r1_ascend.sh").read_text()
    assert "ASCEND_STACK_MODE" in setup and "reference" in setup and "vendor" in setup
    assert "torch==" not in project and "torch-npu==" not in project
    assert "torch==2.7.1" in reference and "torch-npu==2.7.1.post4" in reference
    assert '--max-trajectories "${MAX_TRAJECTORIES:-20}"' in bench
    assert "unset UAVFLOW_PROFILE_NPU" in bench
    assert "UAVFLOW_QWEN_BENCHMARK_MODE=normal" in bench


def test_ascend_project_install_resolves_dependencies_but_protects_runtime():
    setup = (ROOT / "scripts/setup_ascend_python.sh").read_text()
    flattened = setup.replace("\\\n", " ")
    requirements_commands = [
        line for line in flattened.splitlines()
        if 'requirements-ascend.txt' in line and 'pip install' in line
    ]
    assert len(requirements_commands) == 2  # dry-run and actual installation
    assert all("--no-deps" not in line for line in requirements_commands)
    assert all("INSTALL_CONSTRAINTS" in line for line in requirements_commands)
    assert '--upgrade-strategy only-if-needed' in flattened

    assert 'INSTALL_CONSTRAINTS=(-c "${PROJECT_CONSTRAINTS}")' in setup
    assert 'INSTALL_CONSTRAINTS+=(-c "${REFERENCE_CONSTRAINTS}")' in setup
    assert 'VENDOR_RUNTIME_CONSTRAINTS="$(mktemp' in setup
    assert 'accelerator_snapshot | sed' in setup
    assert 'INSTALL_CONSTRAINTS+=(-c "${VENDOR_RUNTIME_CONSTRAINTS}")' in setup
    assert 'numpy==1.26.4 scipy==1.15.3 transformers==5.5.4 huggingface-hub==1.10.1' in setup


def test_ascend_no_deps_is_limited_to_validated_special_cases():
    setup = (ROOT / "scripts/setup_ascend_python.sh").read_text()
    flattened = setup.replace("\\\n", " ")
    no_deps_lines = [line.strip() for line in flattened.splitlines() if "--no-deps" in line]
    assert len(no_deps_lines) == 3
    assert any('"${TORCHVISION_WHEEL}"' in line for line in no_deps_lines)
    assert any('triton-ascend==3.2.1' in line for line in no_deps_lines)
    assert any('-e "${DA3_DIR}"' in line for line in no_deps_lines)


def test_bootstrap_control_flow_and_environment_report_are_present():
    setup = (ROOT / "scripts/setup_ascend_cluster.sh").read_text()
    python_setup = (ROOT / "scripts/setup_ascend_python.sh").read_text()
    cann_setup = (ROOT / "scripts/setup_ascend_cann.sh").read_text()
    runtime = (ROOT / "scripts/ascend_env.sh").read_text()
    assert "setup_ascend_cann.sh" in setup and "setup_ascend_python.sh" in setup
    assert "--bootstrap-only" in setup
    assert "ASCEND_BOOTSTRAP_PYTHON" in setup
    assert "ASCEND_ENV_ROOT" in python_setup
    assert "command -v conda" in python_setup
    assert "micromamba" not in python_setup and " mamba" not in python_setup
    assert "repo.anaconda.com/miniconda" in (ROOT / "scripts/ascend_bootstrap.py").read_text()
    assert "CANN_TOOLKIT_INSTALLER" in cann_setup and "CANN_OPS_INSTALLER" in cann_setup
    assert "cann-installer-url" in cann_setup and "cann-ops-installer-url" in cann_setup
    assert "ascend_ops_install.info" in (ROOT / "scripts/ascend_bootstrap.py").read_text()
    assert "--install-path=" in cann_setup
    assert "environment-report.json" in setup
    assert "ASCEND_STACK_MODE" in cann_setup
    assert "export ASCEND_STACK_MODE=vendor" not in cann_setup
    assert '${REPO_ROOT}/.ascend/env' in runtime
    assert 'ASCEND_VENV:-${ASCEND_ENV_ROOT}' in runtime
    assert '.ascend/runtime.env' in runtime
    assert "apt install" not in cann_setup and "yum install" not in cann_setup
    assert "driver.run" not in cann_setup and "firmware.run" not in cann_setup
    for report_key in (
        "driver_version_info", "firmware_version_info", "cann_metadata",
        "triton_ascend_distribution", "fla_commit", "da3_commit",
    ):
        assert report_key in setup


def test_cann_reference_paths_handle_missing_wrong_and_existing_versions():
    source = (ROOT / "scripts/setup_ascend_cann.sh").read_text()
    assert "select_reference_env" in source
    assert "install_reference_cann" in source
    assert "No reusable CANN 9.0.0 installation found" in source
    assert "administrator CANN" in source
    assert "CANN_USER_ROOT" in source
    assert "CANN_TOOLKIT_INSTALLER" in source
    assert "CANN_OPS_INSTALLER" in source
    assert source.count('--install-path="${CANN_USER_ROOT}"') == 2
    assert "CANN_OPS_VERSION_DETECTED" in source
    assert "ops_json_for_env" in source


def test_vendor_mode_requires_existing_explicit_python_environment():
    source = (ROOT / "scripts/setup_ascend_python.sh").read_text()
    assert "ASCEND_ENV_ROOT_EXPLICIT" in source
    assert "Vendor mode does not create a new environment" in source
    assert "Vendor ASCEND_ENV_ROOT has no bin/python" in source
    vendor_branch = source.split('if [[ "${ASCEND_STACK_MODE}" == vendor ]]', 1)[1]
    vendor_branch = vendor_branch.split("else", 1)[0]
    assert "conda create" not in vendor_branch
    assert "import torch, torch_npu, torchvision" in vendor_branch


def test_pythonless_bootstrap_precedes_cann_and_uses_no_python_helper():
    orchestrator = (ROOT / "scripts/setup_ascend_cluster.sh").read_text()
    python_setup = (ROOT / "scripts/setup_ascend_python.sh").read_text()
    assert orchestrator.index("--bootstrap-only") < orchestrator.index("setup_ascend_cann.sh")
    bootstrap_body = python_setup.split("bootstrap_conda() {", 1)[1].split(
        "\n}\n\nif [[", 1
    )[0]
    assert "python3" not in bootstrap_body
    assert "https://repo.anaconda.com/miniconda/${expected}" in bootstrap_body


def test_runtime_selection_is_persisted_and_preferred():
    installer = (ROOT / "scripts/setup_ascend_cann.sh").read_text()
    runtime = (ROOT / "scripts/ascend_env.sh").read_text()
    assert 'RUNTIME_ENV="${ASCEND_RUNTIME_ENV:-${ROOT}/.ascend/runtime.env}"' in installer
    assert "CANN_ENV_FILE" in installer and "CANN_SELECTION_SOURCE" in installer
    assert 'source "${_UAVFLOW_RUNTIME_ENV}"' in runtime
    assert "Persisted CANN_ENV_FILE is stale" in runtime


def test_cann_discovery_canonicalizes_symlink_aliases(tmp_path):
    import subprocess
    physical = tmp_path / "real/cann/set_env.sh"
    physical.parent.mkdir(parents=True)
    physical.write_text("#!/usr/bin/env bash\n")
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    (aliases / "one").symlink_to(physical.parent, target_is_directory=True)
    (aliases / "two").symlink_to(physical.parent, target_is_directory=True)
    result = subprocess.run(
        ["bash", "-c", f"source {ROOT / 'scripts/ascend_cann.sh'}; uavflow_list_cann_envs"],
        text=True, capture_output=True,
        env={**os.environ, "HOME": str(tmp_path / "empty-home"),
             "ASCEND_SEARCH_ROOT": str(tmp_path), "CANN_ROOT": ""},
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [str(physical.resolve())]


def test_setup_smoke_includes_fla_gated_delta_rule_backward():
    setup = (ROOT / "scripts/setup_ascend_cluster.sh").read_text()
    assert "chunk_gated_delta_rule" in setup
    assert "use_qk_l2norm_in_kernel=True" in setup
    assert "gdr_loss.backward()" in setup


def test_lora_forward_and_gradients_match_explicit_reference():
    torch.manual_seed(3)
    wrapped = LoRALinear(nn.Linear(7, 5), rank=3, alpha=6.0)
    with torch.no_grad():
        wrapped.lora_B.normal_()
    x = torch.randn(2, 4, 7, requires_grad=True)
    output = wrapped(x)
    reference = wrapped.base(x) + (
        torch.nn.functional.linear(
            torch.nn.functional.linear(x, wrapped.lora_A), wrapped.lora_B
        )
        * wrapped.scaling
    )
    assert torch.allclose(output, reference)
    grad = torch.randn_like(output)
    dx, d_a, d_b = torch.autograd.grad(
        output, (x, wrapped.lora_A, wrapped.lora_B), grad, retain_graph=True
    )
    rdx, rd_a, rd_b = torch.autograd.grad(
        reference, (x, wrapped.lora_A, wrapped.lora_B), grad
    )
    assert torch.allclose(dx, rdx)
    assert torch.allclose(d_a, rd_a)
    assert torch.allclose(d_b, rd_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_lora_amp_forward_and_gradients_match_reference_cuda():
    torch.manual_seed(4)
    wrapped = LoRALinear(nn.Linear(16, 12), rank=4, alpha=8.0).cuda()
    with torch.no_grad():
        wrapped.lora_B.normal_()
    x = torch.randn(3, 5, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = wrapped(x)
        reference = wrapped.base(x) + (
            torch.nn.functional.linear(
                torch.nn.functional.linear(x, wrapped.lora_A), wrapped.lora_B
            ).to(wrapped.base(x).dtype)
            * wrapped.scaling
        )
    assert torch.allclose(output, reference, atol=2e-2, rtol=2e-2)
    grad = torch.randn_like(output)
    gradients = torch.autograd.grad(
        output, (x, wrapped.lora_A, wrapped.lora_B), grad, retain_graph=True
    )
    references = torch.autograd.grad(
        reference, (x, wrapped.lora_A, wrapped.lora_B), grad
    )
    for actual, expected in zip(gradients, references):
        assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_ascend_requirements_keep_pycolmap_optional():
    core = [
        line.strip()
        for line in (ROOT / "requirements-ascend.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    optional = (ROOT / "requirements-optional-geometry.txt").read_text()
    assert not any(line.startswith("pycolmap") for line in core)
    assert "pycolmap==3.13.0" in optional


def test_r1_keeps_full_method_semantics():
    from experiments.uavflow_remote_ablation.matrix_v2 import overrides

    values = set(overrides("R1"))
    required = {
        "stage1.qwen_lora_enabled=true",
        "model.parallel_vla_gfm_mode=qwen_tokens",
        "model.parallel_current_depth_enabled=true",
        "model.parallel_current_geometry_read_enabled=true",
        "model.current_geometry_bank_mode=output_current",
        "model.parallel_action_decode_mode=geometry_residual",
        "model.train_deep_backbone=true",
        "model.deep_lora.enabled=false",
        "loss.depth_target_mode=both",
    }
    assert required <= values
