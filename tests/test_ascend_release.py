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
        env={**os.environ, "ASCEND_SEARCH_ROOT": str(tmp_path), "CANN_ROOT": ""},
    )
    assert result.returncode != 0
    assert "Multiple CANN installations" in result.stderr


def test_ascend_stack_modes_and_benchmark_are_explicit():
    setup = (ROOT / "scripts/setup_ascend_cluster.sh").read_text()
    project = (ROOT / "constraints-ascend.txt").read_text()
    reference = (ROOT / "constraints-ascend-reference.txt").read_text()
    bench = (ROOT / "scripts/bench_r1_ascend.sh").read_text()
    assert "ASCEND_STACK_MODE" in setup and "reference" in setup and "vendor" in setup
    assert "torch==" not in project and "torch-npu==" not in project
    assert "torch==2.7.1" in reference and "torch-npu==2.7.1.post4" in reference
    assert '--max-trajectories "${MAX_TRAJECTORIES:-20}"' in bench
    assert "unset UAVFLOW_PROFILE_NPU" in bench
    assert "UAVFLOW_QWEN_BENCHMARK_MODE=normal" in bench


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
