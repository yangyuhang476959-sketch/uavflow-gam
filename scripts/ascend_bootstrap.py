#!/usr/bin/env python3
"""Pure bootstrap helpers for the Ascend setup shell script.

This module performs no downloads or installations.  Keeping detection and
command construction pure makes the new-node control flow testable on CUDA
development machines without importing torch_npu.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import sys
from pathlib import Path
from typing import Iterable


REFERENCE_CANN = "9.0.0"
REFERENCE_TORCH = "2.7.1"
REFERENCE_TORCH_NPU = "2.7.1.post4"
REFERENCE_TORCHVISION = "0.22.1"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
MINICONDA_BASE_URL = "https://repo.anaconda.com/miniconda"
CANN_900_BASE_URL = (
    "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/"
    "CANN/CANN%209.0.0"
)


def normalize_arch(value: str | None = None) -> str:
    raw = (value or platform.machine()).strip().lower()
    aliases = {"arm64": "aarch64", "amd64": "x86_64"}
    result = aliases.get(raw, raw)
    if result not in {"aarch64", "x86_64"}:
        raise ValueError(
            f"Unsupported CPU architecture {raw!r}; expected aarch64 or x86_64"
        )
    return result


def is_python311(version: str) -> bool:
    match = re.fullmatch(r"3\.11(?:\.\d+)?", version.strip())
    return match is not None


def parse_install_info(path: Path, *, expected_arch: str) -> str:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        fields[key.strip().lower()] = value.strip().strip('"\'')
    package = fields.get("package_name", "").lower()
    if "toolkit" not in package:
        raise ValueError(f"Not CANN Toolkit metadata: {path}")
    metadata_arch = normalize_arch(fields.get("arch", expected_arch))
    if metadata_arch != normalize_arch(expected_arch):
        raise ValueError(
            f"CANN metadata architecture mismatch: {metadata_arch} != {expected_arch}"
        )
    version = fields.get("version", "")
    if not version:
        raise ValueError(f"CANN Toolkit metadata has no version field: {path}")
    return version


def detect_cann_version(root: Path, *, arch: str) -> tuple[str, Path]:
    """Read only Huawei's official Toolkit install-info metadata."""
    root = root.expanduser()
    candidates = sorted(set(root.rglob("ascend_toolkit_install.info")))
    parsed: list[tuple[str, Path]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            parsed.append((parse_install_info(path, expected_arch=arch), path))
        except ValueError as exc:
            errors.append(str(exc))
    if not parsed:
        detail = "; ".join(errors) if errors else "no ascend_toolkit_install.info"
        raise RuntimeError(f"Cannot detect CANN Toolkit version under {root}: {detail}")
    versions = {version for version, _ in parsed}
    if len(versions) != 1:
        rendered = ", ".join(f"{version} ({path})" for version, path in parsed)
        raise RuntimeError(f"Conflicting CANN Toolkit metadata under {root}: {rendered}")
    return parsed[0]


def parse_ops_install_info(path: Path, *, expected_arch: str) -> tuple[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        fields[key.strip().lower()] = value.strip().strip('"\'')
    metadata_arch = normalize_arch(fields.get("arch", expected_arch))
    if metadata_arch != normalize_arch(expected_arch):
        raise ValueError(
            f"CANN ops metadata architecture mismatch: {metadata_arch} != {expected_arch}"
        )
    version = fields.get("version", "")
    if not version:
        raise ValueError(f"CANN ops metadata has no version field: {path}")
    package = fields.get("package_name", "").lower()
    if package and ("ops" not in package or "910b" not in package):
        raise ValueError(f"Not the Atlas A2/910B ops metadata: {path} ({package})")
    return version, package


def detect_ops_version(root: Path, *, arch: str) -> tuple[str, Path, str]:
    """Read Huawei's official ascend_ops_install.info metadata only."""
    root = root.expanduser()
    candidates = sorted(set(root.rglob("ascend_ops_install.info")))
    parsed: list[tuple[str, Path, str]] = []
    errors: list[str] = []
    for path in candidates:
        try:
            version, package = parse_ops_install_info(path, expected_arch=arch)
            parsed.append((version, path, package))
        except ValueError as exc:
            errors.append(str(exc))
    if not parsed:
        detail = "; ".join(errors) if errors else "no ascend_ops_install.info"
        raise RuntimeError(f"Cannot detect CANN 910B ops version under {root}: {detail}")
    versions = {version for version, _, _ in parsed}
    if len(versions) != 1:
        rendered = ", ".join(f"{version} ({path})" for version, path, _ in parsed)
        raise RuntimeError(f"Conflicting CANN ops metadata under {root}: {rendered}")
    return parsed[0]


def cann_installer_name(arch: str) -> str:
    return f"Ascend-cann-toolkit_{REFERENCE_CANN}_linux-{normalize_arch(arch)}.run"


def cann_ops_installer_name(arch: str, chip: str = "910b") -> str:
    """Official CANN >=8.5 operator-package naming documented by Huawei."""
    chip = chip.strip().lower()
    if chip != "910b":
        raise ValueError(f"Unsupported reference ops target {chip!r}; expected 910b")
    return f"Ascend-cann-{chip}-ops_{REFERENCE_CANN}_linux-{normalize_arch(arch)}.run"


def cann_installer_url(arch: str) -> str:
    return f"{CANN_900_BASE_URL}/{cann_installer_name(arch)}"


def cann_ops_installer_url(arch: str) -> str:
    return f"{CANN_900_BASE_URL}/{cann_ops_installer_name(arch)}"


def miniconda_installer_name(arch: str) -> str:
    suffix = "aarch64" if normalize_arch(arch) == "aarch64" else "x86_64"
    return f"Miniconda3-latest-Linux-{suffix}.sh"


def miniconda_installer_url(arch: str) -> str:
    return f"{MINICONDA_BASE_URL}/{miniconda_installer_name(arch)}"


def validate_cann_installer(path: Path, *, arch: str) -> Path:
    expected = cann_installer_name(arch)
    if path.name != expected:
        raise ValueError(
            f"Expected official {expected}, got {path.name}. "
            "Set CANN_INSTALLER to the exact official Toolkit package."
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_cann_ops_installer(path: Path, *, arch: str) -> Path:
    expected = cann_ops_installer_name(arch)
    if path.name != expected:
        raise ValueError(
            f"Expected official {expected}, got {path.name}. "
            "Set CANN_OPS_INSTALLER to the exact official 910B ops package."
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_cached_cann_installer(paths: Iterable[Path], *, arch: str) -> Path | None:
    expected = cann_installer_name(arch)
    matches = [path / expected for path in paths if (path / expected).is_file()]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple cached CANN installers found; set CANN_INSTALLER explicitly: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def find_cached_package(
    paths: Iterable[Path], *, expected_name: str, variable: str
) -> Path | None:
    matches = [path / expected_name for path in paths if (path / expected_name).is_file()]
    unique = sorted({path.resolve() for path in matches})
    if len(unique) > 1:
        raise RuntimeError(
            f"Multiple cached {expected_name} packages found; set {variable} explicitly: "
            + ", ".join(str(path) for path in unique)
        )
    return unique[0] if unique else None


def online_torch_commands(python: str) -> list[list[str]]:
    constraints = [
        "-c", "constraints-ascend.txt",
        "-c", "constraints-ascend-reference.txt",
    ]
    prefix = [python, "-m", "pip", "install", *constraints]
    return [
        [*prefix, f"torch=={REFERENCE_TORCH}", "--index-url", PYTORCH_CPU_INDEX],
        [*prefix, f"torch-npu=={REFERENCE_TORCH_NPU}", "--index-url", PYPI_INDEX],
        [*prefix, f"torchvision=={REFERENCE_TORCHVISION}", "--index-url", PYTORCH_CPU_INDEX],
    ]


def validate_local_wheel(path: Path, *, arch: str, python_tag: str = "cp311") -> None:
    name = path.name.lower()
    normalized = normalize_arch(arch)
    accepted_arches = ("aarch64", "arm64") if normalized == "aarch64" else ("x86_64",)
    if not path.is_file():
        raise FileNotFoundError(path)
    if python_tag not in name:
        raise ValueError(f"Wheel is not for {python_tag}: {path}")
    if not any(value in name for value in accepted_arches):
        raise ValueError(f"Wheel does not match host architecture {normalized}: {path}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    arch_parser = sub.add_parser("arch")
    arch_parser.add_argument("value", nargs="?")
    python_parser = sub.add_parser("python311")
    python_parser.add_argument("version")
    cann_parser = sub.add_parser("cann-version")
    cann_parser.add_argument("root", type=Path)
    cann_parser.add_argument("--arch", required=True)
    ops_version = sub.add_parser("cann-ops-version")
    ops_version.add_argument("root", type=Path)
    ops_version.add_argument("--arch", required=True)
    name_parser = sub.add_parser("cann-installer-name")
    name_parser.add_argument("--arch", required=True)
    ops_parser = sub.add_parser("cann-ops-installer-name")
    ops_parser.add_argument("--arch", required=True)
    cann_url = sub.add_parser("cann-installer-url")
    cann_url.add_argument("--arch", required=True)
    ops_url = sub.add_parser("cann-ops-installer-url")
    ops_url.add_argument("--arch", required=True)
    miniconda_name = sub.add_parser("miniconda-installer-name")
    miniconda_name.add_argument("--arch", required=True)
    miniconda_url = sub.add_parser("miniconda-installer-url")
    miniconda_url.add_argument("--arch", required=True)
    torch_parser = sub.add_parser("torch-plan")
    torch_parser.add_argument("--python", required=True)
    args = parser.parse_args()

    if args.command == "arch":
        print(normalize_arch(args.value))
    elif args.command == "python311":
        if not is_python311(args.version):
            raise SystemExit(f"Python 3.11.x required, got {args.version}")
    elif args.command == "cann-version":
        version, metadata = detect_cann_version(args.root, arch=args.arch)
        print(json.dumps({"version": version, "metadata": str(metadata)}))
    elif args.command == "cann-ops-version":
        version, metadata, package = detect_ops_version(args.root, arch=args.arch)
        print(json.dumps({"version": version, "metadata": str(metadata), "package": package}))
    elif args.command == "cann-installer-name":
        print(cann_installer_name(args.arch))
    elif args.command == "cann-ops-installer-name":
        print(cann_ops_installer_name(args.arch))
    elif args.command == "cann-installer-url":
        print(cann_installer_url(args.arch))
    elif args.command == "cann-ops-installer-url":
        print(cann_ops_installer_url(args.arch))
    elif args.command == "miniconda-installer-name":
        print(miniconda_installer_name(args.arch))
    elif args.command == "miniconda-installer-url":
        print(miniconda_installer_url(args.arch))
    elif args.command == "torch-plan":
        for command in online_torch_commands(args.python):
            print(shlex.join(command))


if __name__ == "__main__":
    _main()
