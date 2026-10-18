# NVIDIA Linux simulation handoff

This directory copies the existing local closed-loop runner, **without changing
its control/model logic**. It is independent of Ascend training setup. The UE
binary, task JSONs, weights, annotations and generated results are not uploaded.
The local setup has run on NVIDIA RTX 4090/Linux; this copied handoff has only
been import/static tested, not freshly rendered on another machine.

## 1. Separate simulator Python environment

Keep the simulator client separate from the DA3/Qwen policy environment.
`requirements-sim.txt` records the currently installed client versions, including
NumPy 2; do not apply it to the training environment's NumPy <2 stack.
Use Linux x86_64 with an operational NVIDIA graphics/Vulkan driver. `nvidia-smi`
alone does not establish Vulkan rendering availability. Do not install/replace
system drivers using these instructions.

```bash
cd /path/to/uavflow-gam
export PROJECT_ROOT="$PWD"
git clone https://github.com/buaa-colalab/UAV-Flow.git third_party/UAV-Flow
git -C third_party/UAV-Flow checkout dbcee9e553f7fb5cffae91c941b0d3c27bf190e8
export UAVFLOW_EVAL_ROOT="$PROJECT_ROOT/third_party/UAV-Flow/UAV-Flow-Eval"

# Apply the two small fixes already present in our working local installation:
# forward gpu_id/sleep_time to UE; scalar conversion for newer NumPy.
git -C third_party/UAV-Flow apply --check "$PROJECT_ROOT/simulation/uavflow_eval_linux.patch"
git -C third_party/UAV-Flow apply "$PROJECT_ROOT/simulation/uavflow_eval_linux.patch"

conda create -n uavflow-sim python=3.11 pip -y
conda activate uavflow-sim
python -m pip install -r "$PROJECT_ROOT/simulation/requirements-sim.txt"
python -m pip install --no-deps --no-build-isolation -e "$UAVFLOW_EVAL_ROOT"
export SIM_PYTHON="$(command -v python)"
export PYTHONPATH="$UAVFLOW_EVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python "$PROJECT_ROOT/simulation/batch_run_act_all_debug.py" --help
```

Apply the patch only once to the pinned checkout. If already applied, a second
`git apply` fails intentionally; do not reset a checkout with your own changes.
`third_party/` is ignored by this repository.

## 2. UE binary and task files

Use the same **Collection_v4_LinuxNoEditor** package as the local evaluations.
Transfer the entire extracted package (including Content/Paks), not just its
executable. The upstream README links a Windows archive; it is **not** a Linux
replacement. An exact public download URL for this local Linux build has not
been confirmed, so obtain it from the experiment owner. Different Collection
builds can change assets, blueprints and camera behavior.

Example transfer, with paths chosen by the receiving engineer:

```bash
rsync -avP /source/Collection_v4_LinuxNoEditor/ user@server:/chosen/path/Collection_v4_LinuxNoEditor/
```

On the receiving machine, set the actual executable and update only the Linux
`env_bin` field in your own upstream checkout:

```bash
export UE_BINARY=/chosen/path/Collection_v4_LinuxNoEditor/Collection/Binaries/Linux/Collection
test -f "$UE_BINARY"
chmod u+x "$UE_BINARY"
python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['UAVFLOW_EVAL_ROOT']) / 'gym_unrealcv/envs/setting/Track/DowntownWest.json'
cfg = json.loads(p.read_text())
cfg['env_bin'] = str(Path(os.environ['UE_BINARY']).resolve(strict=True))
p.write_text(json.dumps(cfg, indent=2) + '\n')
print('Configured:', cfg['env_bin'])
PY
export TASK_ROOT="$UAVFLOW_EVAL_ROOT/test_jsons"
find "$TASK_ROOT" -maxdepth 1 -name '*.json' | wc -l
```

The local full evaluation uses 273 JSONs. Verify the received task IDs and scene
package against the experiment being reproduced. Use a normal non-root Linux
user for UE4. Offscreen still needs graphics/Vulkan libraries and may require
the host's working X session (`DISPLAY`/`XAUTHORITY`); use that host's actual
configuration, not a copied local username or display number. Run only one
instance against a given UE installation/UnrealCV port at a time.

## 3. Render smoke and closed-loop run

The runner starts UE through Gym/UnrealCV. Do not also launch the same UE package
manually. `UE_ADAPTER` is the **Vulkan graphics adapter index**, not a guaranteed
CUDA index. Verify the selected GPU in the UE log/process list. A CUDA mask does
not reliably choose the UE rendering adapter.

First capture one initial onboard frame without a policy server:

```bash
export UE_ADAPTER=0
export OUT="$PROJECT_ROOT/results/simulation/initial_smoke"
bash "$PROJECT_ROOT/simulation/run_nvidia.sh" --capture_initial_only --limit_tasks 1
```

For closed loop, separately start a compatible policy service on localhost
port 5007 in its **model environment**. This handoff is the simulator client,
not a new model server; it does not guarantee compatibility with every new
checkpoint/architecture. Use the policy's matching inference implementation.
If inference uses the second CUDA GPU, set `CUDA_VISIBLE_DEVICES=1` in the
**policy server terminal only**. No CUDA/PyTorch installation is needed in the
simulator client environment.

```bash
export PORT=5007
export OUT="$PROJECT_ROOT/results/simulation/my_model_full273"
mkdir -p "$OUT"
bash "$PROJECT_ROOT/simulation/run_nvidia.sh" 2>&1 | tee -a "$OUT/eval.log"
```

Optional detached session (export the variables above before running):

```bash
screen -dmS uavflow_sim bash -c 'bash "$PROJECT_ROOT/simulation/run_nvidia.sh" >> "$OUT/eval.log" 2>&1'
screen -r uavflow_sim
```

Reuse the same `OUT` to skip existing nonempty trajectory JSONs. This is the
original completion rule, not a guarantee that every saved rollout reached a
successful terminal state: inspect logs after interruptions. Use a fresh `OUT`
for another model/protocol; add `--overwrite_existing` only to intentionally
replace selected episodes. Add `--task_ids ID1,ID2` for a subset.

Outputs: `results/<episode>.json`, 2D/3D trajectory PNGs, and `obs/` containing
the exact 224x224 RGB observations sent to the policy. No preview service or
annotation database is bundled, and no annotations are modified.

## Preserved protocol / API

- Initial pose convergence wait, mounted agent camera, per-task Blueprint
  motion reset and target-placement readback; restart particles before capture.
- Maximum 100 executed returned action poses; original small-motion stop enabled:
  10 successive commanded-pose changes with each XYZ axis <3 cm and yaw <1 degree. The server can
  additionally return `done: true`.
- Client POST `/reset` between episodes, POST `/predict` with `image` (base64 PNG),
  `instr`, and `proprio=[x_cm,y_cm,z_cm,yaw_deg]` relative to the episode's first
  heading. Response `action` contains **absolute episode-relative poses**
  `[x_cm,y_cm,z_cm,yaw_rad]`, not raw local incremental `[dx,dy,dz,dyaw]`.
  The client executes every returned pose; chunk length/executed-step selection
  is the policy adapter's responsibility. Never return raw training actions
  without the matching decoder/coordinate conversion.
- Normal completion calls `env.close()`. There is no name-based `pkill`, no
  global GPU cleanup, and no automatic policy-service shutdown. If interrupted,
  inspect the exact UE PID before stopping it; do not kill unrelated sessions.

## Provenance

`batch_run_act_all_debug.py` is an unchanged copy of the working local
Geometric-Action-Model runner. Upstream dependency: buaa-colalab/UAV-Flow commit
`dbcee9e553f7fb5cffae91c941b0d3c27bf190e8` (Apache-2.0; license copied as
`UPSTREAM_LICENSE`). The patch contains the existing local Linux launch and
NumPy scalar fixes only. The cloned upstream sources retain their licenses.
