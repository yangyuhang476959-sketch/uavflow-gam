"""Small runtime helpers shared by the Stage-2 training entrypoint."""

from __future__ import annotations

import importlib.util
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf


def requested_accelerator() -> str:
    """Return the requested execution backend without importing its plugin."""
    value = os.environ.get("UAVFLOW_ACCELERATOR", "auto").strip().lower()
    aliases = {"gpu": "cuda", "ascend": "npu"}
    value = aliases.get(value, value)
    if value not in {"auto", "cuda", "npu", "cpu"}:
        raise ValueError(
            "UAVFLOW_ACCELERATOR must be auto, cuda, npu/ascend, or cpu; "
            f"got {value!r}."
        )
    return value


def bootstrap_accelerator_plugin() -> Any | None:
    """Import torch_npu before model libraries when an Ascend run is requested.

    TorchNPU registers the ``npu`` device and monkey-patches supported PyTorch
    operations during import.  Doing this before Transformers/DA3 are imported
    avoids backend capability checks being cached as CUDA/CPU-only.
    """
    requested = requested_accelerator()
    if requested == "npu" or (
        requested == "auto"
        and not torch.cuda.is_available()
        and (
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None
            or importlib.util.find_spec("torch_npu") is not None
        )
    ):
        try:
            import torch_npu  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ascend execution requires a CANN-compatible torch + torch_npu "
                "runtime. Start from the vendor Ascend PyTorch image and set "
                "UAVFLOW_ACCELERATOR=npu."
            ) from exc
        return torch_npu
    return None


@dataclass(frozen=True)
class Accelerator:
    kind: str
    device: torch.device
    distributed_backend: str
    module: Any | None = None


def configure_accelerator(*, local_rank: int, distributed: bool) -> Accelerator:
    """Select CUDA, Ascend NPU, or CPU and initialize the process group."""
    requested = requested_accelerator()
    torch_npu = bootstrap_accelerator_plugin()
    npu_available = bool(
        torch_npu is not None
        and hasattr(torch_npu, "npu")
        and torch_npu.npu.is_available()
    )
    if requested == "npu" and not npu_available:
        raise RuntimeError("UAVFLOW_ACCELERATOR=npu but torch_npu reports no available NPU.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("UAVFLOW_ACCELERATOR=cuda but PyTorch reports no available CUDA GPU.")

    if requested == "npu" or (requested == "auto" and npu_available):
        assert torch_npu is not None
        torch_npu.npu.set_device(local_rank)
        accelerator = Accelerator(
            kind="npu",
            device=torch.device(f"npu:{local_rank}"),
            distributed_backend="hccl",
            module=torch_npu.npu,
        )
    elif requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        torch.cuda.set_device(local_rank)
        accelerator = Accelerator(
            kind="cuda",
            device=torch.device(f"cuda:{local_rank}"),
            distributed_backend="nccl",
            module=torch.cuda,
        )
    else:
        accelerator = Accelerator(
            kind="cpu", device=torch.device("cpu"), distributed_backend="gloo"
        )

    if distributed:
        torch.distributed.init_process_group(accelerator.distributed_backend)
    return accelerator


def amp_dtype(name: str, device_type: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "auto":
        # FP16 works across more Ascend generations. CUDA keeps the validated
        # BF16 behavior used by the original experiments.
        value = "fp16" if device_type == "npu" else "bf16"
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"training.amp_dtype must be auto, fp16, or bf16; got {name!r}.")


def autocast_context(
    device: torch.device | str,
    *,
    enabled: bool,
    dtype_name: str = "auto",
):
    device_type = torch.device(device).type
    if not enabled or device_type == "cpu":
        return nullcontext()
    dtype = amp_dtype(dtype_name, device_type)
    if device_type == "npu":
        torch_npu = bootstrap_accelerator_plugin()
        assert torch_npu is not None
        from torch_npu.npu import amp as npu_amp  # type: ignore

        # FP16 is TorchNPU AMP's broadest/default compatibility mode. Avoid a
        # dtype keyword for older, still common vendor images.
        if dtype == torch.float16:
            return npu_amp.autocast(enabled=True)
        return npu_amp.autocast(enabled=True, dtype=dtype)
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=True)


def create_grad_scaler(device_type: str, *, enabled: bool):
    if device_type == "npu" and enabled:
        bootstrap_accelerator_plugin()
        from torch_npu.npu import amp as npu_amp  # type: ignore

        return npu_amp.GradScaler()
    return torch.amp.GradScaler(
        "cuda" if device_type == "cuda" else "cpu",
        enabled=bool(enabled and device_type == "cuda"),
    )


def manual_seed_all(accelerator: Accelerator, seed: int) -> None:
    if accelerator.kind == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif accelerator.kind == "npu":
        accelerator.module.manual_seed_all(seed)


def step_generator(accelerator: Accelerator, seed: int) -> torch.Generator:
    """Create a resume-stable generator supported by the active backend.

    CUDA supports device-local generators.  TorchNPU releases in common CANN
    images do not consistently support ``torch.Generator(device='npu')``;
    use a CPU generator there and let the small random-control tensors be
    copied to the target device by ``rand_on_device``/``randn_on_device``.
    """
    device: torch.device | str = accelerator.device
    if accelerator.kind in {"npu", "cpu"}:
        device = "cpu"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _generator_device(generator: torch.Generator | None) -> torch.device | None:
    if generator is None:
        return None
    return torch.device(generator.device)


def rand_on_device(
    shape: tuple[int, ...] | torch.Size,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    target = torch.device(device)
    source = _generator_device(generator) or target
    value = torch.rand(shape, device=source, dtype=dtype, generator=generator)
    return value.to(target) if source.type != target.type else value


def randn_on_device(
    shape: tuple[int, ...] | torch.Size,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    target = torch.device(device)
    source = _generator_device(generator) or target
    value = torch.randn(shape, device=source, dtype=dtype, generator=generator)
    return value.to(target) if source.type != target.type else value


def max_memory_allocated_gb(accelerator: Accelerator) -> float:
    if accelerator.module is None:
        return 0.0
    function = getattr(accelerator.module, "max_memory_allocated", None)
    if function is None:
        return 0.0
    try:
        value = function(accelerator.device)
    except TypeError:
        # Older torch_npu versions expose the CUDA-compatible function without
        # a device parameter and report the current local NPU.
        value = function()
    return float(value) / (1024 ** 3)


def distributed_info() -> tuple[bool, int, int, int]:
    """Return ``(is_distributed, rank, local_rank, world_size)``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world > 1, rank, local_rank, world


def apply_overrides(cfg: Any, overrides: list[str]) -> Any:
    """Apply repeated ``--set dotted.key=value`` command-line overrides."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def lr_scale(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step <= warmup:
        return float(step) / max(1, int(warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
