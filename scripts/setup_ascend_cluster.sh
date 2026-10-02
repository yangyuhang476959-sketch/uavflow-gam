#!/usr/bin/env bash
# One-command orchestrator. CANN and Python phases are independently runnable.
set -Eeuo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export REPO_ROOT="${ROOT}"

echo '== Phase 1/4: CANN Toolkit + 910B ops =='
bash "${ROOT}/scripts/setup_ascend_cann.sh"

echo '== Phase 2/4: Python 3.11 + accelerator/project dependencies =='
bash "${ROOT}/scripts/setup_ascend_python.sh"

echo '== Phase 3/4: Ascend compatibility smoke =='
# shellcheck disable=SC1091
source "${ROOT}/scripts/ascend_env.sh"
PY="${ASCEND_ENV_ROOT}/bin/python"
"${PY}" - <<'PY'
import torch, torch_npu, torchvision
assert torch.npu.is_available() and torch.npu.device_count() > 0
x = torch.randn(64, 64, device="npu", dtype=torch.bfloat16, requires_grad=True)
loss = (x @ x).float().square().mean()
loss.backward()
torch.npu.synchronize()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()

from transformers import Qwen3_5ForConditionalGeneration
from robot.modeling.da3_giant_encoder import _install_da3_optional_stubs
_install_da3_optional_stubs()
from depth_anything_3.api import DepthAnything3
import triton, fla

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
q = torch.randn(1, 2, 16, 16, device="npu", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
g = torch.randn(1, 2, 16, device="npu", dtype=torch.float32, requires_grad=True)
beta = torch.sigmoid(torch.randn(1, 2, 16, device="npu", dtype=torch.float32, requires_grad=True))
out, _ = chunk_gated_delta_rule(
    q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
)
gdr_loss = out.float().square().mean()
gdr_loss.backward()
torch.npu.synchronize()
assert torch.isfinite(out).all().item()
for tensor in (q, k, v, g):
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
print("NPU BF16 + FLA GatedDeltaRule forward/backward PASS", torchvision.__version__, triton.__version__)
PY

echo '== Phase 4/4: environment report =='
REPORT="${ROOT}/.ascend/environment-report.json"
export UAVFLOW_REPORT="${REPORT}"
"${PY}" - <<'PY'
import datetime, importlib.metadata as md, json, os, platform, shlex, subprocess
from pathlib import Path
import numpy, scipy, torch, torch_npu, torchvision, transformers, huggingface_hub, triton

def cmd(*args):
    try:
        return subprocess.run(args, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False).stdout.strip()
    except OSError as exc:
        return repr(exc)

root = Path(os.environ["REPO_ROOT"])
runtime = {}
for line in (root / ".ascend/runtime.env").read_text().splitlines():
    if line.startswith("export ") and "=" in line:
        key, value = line[7:].split("=", 1)
        runtime[key] = shlex.split(value)[0] if value else ""
admin = cmd("bash", "-lc", f"source {root}/scripts/ascend_cann.sh; uavflow_list_cann_envs").splitlines()
metadata_candidates = list(Path(runtime.get("CANN_ROOT", "/nonexistent")).rglob("ascend_toolkit_install.info"))
report = {
    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "architecture": platform.machine(),
    "npu_model_and_status": cmd("npu-smi", "info"),
    "driver_version_info": Path("/usr/local/Ascend/driver/version.info").read_text(errors="replace") if Path("/usr/local/Ascend/driver/version.info").is_file() else None,
    "firmware_version_info": Path("/usr/local/Ascend/firmware/version.info").read_text(errors="replace") if Path("/usr/local/Ascend/firmware/version.info").is_file() else None,
    "administrator_cann_installations_discovered": admin,
    "selected_cann_installation": runtime.get("CANN_ROOT"),
    "selected_cann_source": runtime.get("CANN_SELECTION_SOURCE"),
    "cann_metadata": str(metadata_candidates[0]) if metadata_candidates else None,
    "toolkit_version": runtime.get("CANN_VERSION"),
    "ops_package_version": runtime.get("CANN_VERSION") if os.environ.get("ASCEND_OPP_PATH") else None,
    "cann_set_env": runtime.get("CANN_ENV_FILE"),
    "python_environment_type": "conda-prefix",
    "python_environment_path": os.environ["ASCEND_ENV_ROOT"],
    "python": platform.python_version(),
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "torchvision": torchvision.__version__, "numpy": numpy.__version__,
    "scipy": scipy.__version__, "transformers": transformers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "triton_ascend_distribution": md.version("triton-ascend"),
    "triton_imported_version": triton.__version__,
    "fla_commit": cmd("git", "-C", os.environ["FLA_ASCEND_DIR"], "rev-parse", "HEAD"),
    "da3_commit": cmd("git", "-C", os.environ["DA3_DIR"], "rev-parse", "HEAD"),
}
Path(os.environ["UAVFLOW_REPORT"]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
echo "Ascend setup complete. Report: ${REPORT}"
