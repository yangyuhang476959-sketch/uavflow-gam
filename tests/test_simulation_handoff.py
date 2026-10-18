"""Static handoff checks; do not launch Unreal or allocate GPU memory."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'simulation'


def test_simulator_source_compiles_and_preserves_protocol():
    source = (SIM / 'batch_run_act_all_debug.py').read_text()
    ast.parse(source)
    for setting in ('ACTION_SMALL_DELTA_POS: float = 3.0',
                    'ACTION_SMALL_DELTA_YAW: float = 1.0',
                    'ACTION_SMALL_STEPS: int = 10',
                    '--reset_drone_motion_per_task', '--capture_initial_only'):
        assert setting in source
    assert '/home/vis/' not in source


def test_launcher_shell_and_safe_process_scope():
    launcher = SIM / 'run_nvidia.sh'
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    source = launcher.read_text()
    assert 'pkill' not in source
    assert 'killall' not in source
    assert 'export CUDA_VISIBLE_DEVICES=' not in source
    assert '--reset_drone_motion_per_task' in source
    assert '--disable_small_motion_stop' not in source
    assert '--debug_obs_every 1' in source
    assert '"$@"' in source


def test_linux_patch_is_limited_to_existing_compatibility_fixes():
    patch = (SIM / 'uavflow_eval_linux.patch').read_text()
    assert patch.count('diff --git ') == 2
    assert 'gpu_id=self.gpu_id' in patch
    assert 'sleep_time=self.sleep_time' in patch
    assert 'float(np.asarray(direction)' in patch
