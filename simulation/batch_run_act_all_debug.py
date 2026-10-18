import base64
import requests
import sys
sys.path.append('./')
from io import BytesIO
import json
import numpy as np
from PIL import Image
import gym
import gym_unrealcv
import time
import argparse
import os
import shutil
import cv2
from relative import calculate_new_pose
glob = __import__('glob')
from gym_unrealcv.envs.wrappers import time_dilation, configUE, augmentation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.ticker import MultipleLocator
logger = logging.getLogger(__name__)


class InitialPoseConvergenceError(RuntimeError):
    """Raised when one task cannot reach a safe initial camera pose."""

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
INITIAL_POSE_TIMEOUT_S: float = 15.0
INITIAL_POSE_POLL_S: float = 0.05
INITIAL_LOCATION_TOLERANCE: float = 0.05
INITIAL_YAW_STABILITY_TOLERANCE_DEG: float = 0.02
INITIAL_YAW_TARGET_TOLERANCE_DEG: float = 0.10
INITIAL_POSE_MIN_SETTLE_S: float = 0.5
INITIAL_POSE_STABLE_READS: int = 5
INITIAL_POSE_MAX_YAW_CORRECTIONS: int = 5
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10
DEBUG_IMAGE_PATH: str = './debug.jpg'
TRAJ_IMG_SIZE_2D: Tuple[int, int] = (10, 10)
TRAJ_IMG_SIZE_3D: Tuple[int, int] = (12, 10)
PLOT_YAW_ARROW_MIN_LEN_2D: int = 10
PLOT_YAW_ARROW_MIN_LEN_3D: int = 12


def _trajectory_result_is_complete(path: str) -> bool:
    """Return true only when a prior result contains at least one state.

    Older runs could leave an empty ``[]`` together with successfully rendered
    (but blank) 2D/3D PNGs.  Treating the PNGs alone as a completion marker
    caused resume runs to skip those failed tasks forever.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if isinstance(payload, list):
        return len(payload) > 0
    if isinstance(payload, dict):
        states = payload.get("state")
        return isinstance(states, list) and len(states) > 0
    return False


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _reset_particles(env: Any) -> None:
    """Reset particles to a deterministic phase.

    The DowntownWest package contains ``FountainFX_2``.  If one UE process is
    reused for a batch, its emitters otherwise keep advancing between tasks;
    model latency and preceding trajectory length then decide whether a frame
    contains normal spray, bright green particles, or a broken black billboard.
    ``fx.FreezeParticleSimulation`` does not freeze all GPU particles in this
    cooked build.  Therefore every policy capture restarts the effect instead
    of relying on elapsed UE time.  Do not use ``showflag.Particles 0``: that
    command can deadlock this cooked UE4/Vulkan build.
    """
    client = env.unwrapped.unrealcv.client
    client.request("vrun fx.RestartAll")


def _capture_policy_image(env: Any, use_agent_camera: bool) -> Any:
    """Capture at a fixed particle phase, independent of policy latency."""
    _reset_particles(env)
    return env.unwrapped.unrealcv.get_image(
        _policy_camera_id(env, use_agent_camera), 'lit'
    )


def _set_and_wait_initial_pose(
    env: Any,
    initial_pos: List[float],
    *,
    use_agent_camera: bool,
    reset_drone_motion: bool = False,
    timeout_s: float = INITIAL_POSE_TIMEOUT_S,
    location_tolerance: float = INITIAL_LOCATION_TOLERANCE,
    yaw_stability_tolerance_deg: float = INITIAL_YAW_STABILITY_TOLERANCE_DEG,
    yaw_target_tolerance_deg: float = INITIAL_YAW_TARGET_TOLERANCE_DEG,
) -> Dict[str, Any]:
    """Teleport the UAV and wait for its Blueprint yaw interpolation to finish.

    ``vbp ... set_rotation`` sets a target yaw rather than an instantaneous
    transform. A fixed wall-clock sleep made the first observation depend on
    UE frame rate and therefore on which policy occupied the GPU.

    The Blueprint command is interpolated and can retain a session-dependent
    offset.  We first wait for stable readback, then compensate that offset and
    repeat until the generic actor yaw reaches the requested world yaw.  Merely
    accepting a stable value caused several-pixel initial-camera shifts.
    """
    obj = env.unwrapped.player_list[0]
    if reset_drone_motion:
        # The lightweight evaluator reuses one drone Blueprint for the whole
        # batch.  Teleporting its root transform does not necessarily clear
        # the Blueprint's previous movement/mesh-tilt state, which can leave a
        # rotor arm in front of the mounted camera at the next episode start.
        # Keep this opt-in so an A/B run can compare against the legacy path.
        _sync_unrealcv(env, f"vbp {obj} set_move 0 0 0 0")
        _sync_unrealcv(env, f"vbp {obj} set_phy 0")
        time.sleep(0.25)
    expected_xyz = np.asarray(initial_pos[:3], dtype=np.float64)
    expected_yaw = float(initial_pos[4])
    env.unwrapped.unrealcv.set_obj_location(obj, expected_xyz.tolist())
    blueprint_yaw = expected_yaw - 180.0
    env.unwrapped.unrealcv.set_rotation(obj, blueprint_yaw)

    deadline = time.monotonic() + max(float(timeout_s), INITIAL_POSE_POLL_S)
    stable_reads = 0
    last_xyz: Optional[np.ndarray] = None
    last_yaw: Optional[float] = None
    previous_yaw: Optional[float] = None
    yaw_step: Optional[float] = None
    started = time.monotonic()
    correction_count = 0
    while time.monotonic() < deadline:
        last_xyz = np.asarray(
            env.unwrapped.unrealcv.get_obj_location(obj), dtype=np.float64
        )
        rotation = env.unwrapped.unrealcv.get_obj_rotation(obj)
        last_yaw = float(rotation[1])
        location_error = float(np.linalg.norm(last_xyz[:3] - expected_xyz))
        yaw_error = abs(_wrap_degrees(last_yaw - expected_yaw))
        yaw_step = (
            None
            if previous_yaw is None
            else abs(_wrap_degrees(last_yaw - previous_yaw))
        )
        settled_long_enough = time.monotonic() - started >= INITIAL_POSE_MIN_SETTLE_S
        if (
            settled_long_enough
            and location_error <= float(location_tolerance)
            and yaw_step is not None
            and yaw_step <= float(yaw_stability_tolerance_deg)
        ):
            stable_reads += 1
            if stable_reads >= INITIAL_POSE_STABLE_READS:
                signed_yaw_error = _wrap_degrees(last_yaw - expected_yaw)
                if abs(signed_yaw_error) > float(yaw_target_tolerance_deg):
                    correction_count += 1
                    if correction_count > INITIAL_POSE_MAX_YAW_CORRECTIONS:
                        break
                    blueprint_yaw -= signed_yaw_error
                    env.unwrapped.unrealcv.set_rotation(obj, blueprint_yaw)
                    stable_reads = 0
                    previous_yaw = None
                    started = time.monotonic()
                    time.sleep(INITIAL_POSE_POLL_S)
                    continue
                set_cam(env, use_agent_camera=use_agent_camera)
                # Force two render/readback ticks so the attached camera sees
                # the converged actor transform before the policy image is used.
                camera_id = _policy_camera_id(env, use_agent_camera)
                env.unwrapped.unrealcv.get_image(camera_id, 'lit')
                time.sleep(INITIAL_POSE_POLL_S)
                env.unwrapped.unrealcv.get_image(camera_id, 'lit')
                return {
                    "location": last_xyz[:3].tolist(),
                    "yaw": last_yaw,
                    "location_error": location_error,
                    "yaw_error": yaw_error,
                    "yaw_step": yaw_step,
                    "yaw_corrections": correction_count,
                    "accepted_with_warning": False,
                }
        else:
            stable_reads = 0
        previous_yaw = last_yaw
        time.sleep(INITIAL_POSE_POLL_S)

    raise InitialPoseConvergenceError(
        "Initial UAV pose did not converge: "
        f"expected_xyz={expected_xyz.tolist()} actual_xyz="
        f"{None if last_xyz is None else last_xyz[:3].tolist()} "
        f"expected_yaw={expected_yaw:.6f} actual_yaw={last_yaw} "
        f"last_yaw_step={yaw_step} tolerances=(xyz={location_tolerance}, "
        f"yaw_stability={yaw_stability_tolerance_deg}, "
        f"yaw_target={yaw_target_tolerance_deg}, corrections={correction_count})"
    )


def _processed_gt_pose(state: List[float]) -> np.ndarray:
    """Return Eval-relative [x_cm, y_cm, z_cm, yaw_deg] from a GT state."""
    if not isinstance(state, (list, tuple)) or len(state) < 5:
        raise ValueError(f"Invalid reference_path_preprocessed state: {state!r}")
    return np.asarray(
        [float(state[0]), float(state[1]), float(state[2]), _wrap_degrees(state[4])],
        dtype=np.float64,
    )


def _yaw_local_action(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Use the policy server's convention: local xyz in metres, dyaw in radians."""
    yaw = np.radians(float(previous[3]))
    world_m = (np.asarray(current[:3]) - np.asarray(previous[:3])) * 0.01
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            c * world_m[0] + s * world_m[1],
            -s * world_m[0] + c * world_m[1],
            world_m[2],
            np.radians(_wrap_degrees(float(current[3] - previous[3]))),
        ],
        dtype=np.float64,
    )


def _integrate_yaw_local_action(pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Mirror Stage2Policy._integrate and return an Eval-relative pose."""
    yaw = np.radians(float(pose[3]))
    dx, dy, dz, dyaw = map(float, action[:4])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            float(pose[0]) + 100.0 * (c * dx - s * dy),
            float(pose[1]) + 100.0 * (s * dx + c * dy),
            float(pose[2]) + 100.0 * dz,
            _wrap_degrees(float(pose[3]) + np.degrees(dyaw)),
        ],
        dtype=np.float64,
    )


def _execute_relative_pose(
    env: Any,
    pose: np.ndarray,
    initial_pos: List[float],
    use_agent_camera: bool,
) -> None:
    """Execute the same absolute Eval waypoint representation returned by the server."""
    initial_x, initial_y, initial_z = map(float, initial_pos[:3])
    initial_yaw = float(initial_pos[4])
    theta = np.radians(initial_yaw)
    c, s = np.cos(theta), np.sin(theta)
    global_x = float(pose[0]) * c - float(pose[1]) * s
    global_y = float(pose[0]) * s + float(pose[1]) * c
    absolute_yaw = _wrap_degrees(float(pose[3]) + initial_yaw)
    obj = env.unwrapped.player_list[0]
    env.unwrapped.unrealcv.set_obj_location(
        obj, [global_x + initial_x, global_y + initial_y, float(pose[2]) + initial_z]
    )
    env.unwrapped.unrealcv.set_rotation(obj, absolute_yaw - 180.0)
    set_cam(env, use_agent_camera=use_agent_camera)


def _read_relative_position(
    env: Any, initial_pos: List[float], commanded_yaw: float
) -> Tuple[np.ndarray, List[float]]:
    """Read back position while retaining the Eval waypoint's commanded yaw.

    The drone blueprint's ``vbp set_rotation`` and generic UnrealCV
    ``/object/.../rotation`` endpoint do not expose the same yaw representation.
    The official evaluator therefore carries forward the commanded waypoint
    yaw instead of feeding the generic actor rotation back into the controller.
    Return that faithful pose plus the raw actor rotation for diagnostics.
    """
    obj = env.unwrapped.player_list[0]
    location = np.asarray(env.unwrapped.unrealcv.get_obj_location(obj), dtype=np.float64)
    rotation = np.asarray(env.unwrapped.unrealcv.get_obj_rotation(obj), dtype=np.float64)
    initial_xyz = np.asarray(initial_pos[:3], dtype=np.float64)
    initial_yaw = float(initial_pos[4])
    delta = location[:3] - initial_xyz
    theta = np.radians(initial_yaw)
    c, s = np.cos(theta), np.sin(theta)
    # Inverse of _execute_relative_pose's initial-frame rotation.
    local_x = c * delta[0] + s * delta[1]
    local_y = -s * delta[0] + c * delta[1]
    pose = np.asarray(
        [local_x, local_y, delta[2], _wrap_degrees(commanded_yaw)], dtype=np.float64
    )
    return pose, rotation.tolist()



def send_prediction_request(
    image: Image,
    proprio: np.ndarray,
    instr: str,
    server_url: str,
    debug_save_path: Optional[str] = None,
    request_timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response.

    Args:
        image: PIL image object, resized to 224x224 and sent as PNG.
        proprio: Vehicle state vector (np.ndarray), converted to list.
        instr: Text instruction.
        server_url: Inference service /predict endpoint URL.
    Returns:
        dict or None: Parsed JSON if successful, otherwise None on error.
    """
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        image.save(debug_save_path)
    image.save(img_io, format='PNG')
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    payload: Dict[str, Any] = {
        'image': img_base64,
        'proprio': proprio_list,
        'instr': instr
    }
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=float(request_timeout),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def draw_2d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], reference_path: Optional[List[List[float]]] = None, tick_x: Optional[float] = None, tick_y: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None) -> None:
    """Plot 2D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y to specify grid spacing
    - Minimum span: use min_span_x/min_span_y to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (no data)')
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        ax = plt.gca()
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + '\n' + log_path + ' (invalid data)')
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return
    # The control log starts after the first executed action, whereas the GT
    # reference starts at the episode origin.  Include that common origin in
    # the plotted prediction so the two curves are directly comparable.
    trajectory = np.concatenate(
        [np.zeros((1, trajectory.shape[1]), dtype=trajectory.dtype), trajectory], axis=0
    )
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_2D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    plt.figure(figsize=TRAJ_IMG_SIZE_2D)
    if target is not None:
        plt.scatter(target[1], target[0], color='red', label='target')
    plt.plot(y, x, color='tab:blue', linewidth=2.0, label='Pred (closed loop)')
    if reference_path is not None:
        gt = np.asarray(reference_path, dtype=np.float64)
        if gt.ndim == 2 and gt.shape[0] > 0 and gt.shape[1] >= 3:
            plt.plot(
                gt[:, 1], gt[:, 0], color='black', linestyle='--', linewidth=2.0,
                label='GT reference',
            )
            plt.scatter(gt[-1, 1], gt[-1, 0], color='black', marker='x', s=60, label='GT end')
    plt.quiver(y, x, dy, dx, angles='xy', scale_units='xy', scale=1, color='green', width=0.003, label='yaw')

    ax = plt.gca()
    # Set equal aspect ratio first, then expand to minimum span based on this
    ax.set_aspect('equal', adjustable='box')
    
    # Expand to minimum span without shrinking: X-axis shows Y, Y-axis shows X
    cur_xmin, cur_xmax = ax.get_xlim()
    cur_ymin, cur_ymax = ax.get_ylim()
    span_y_axis = cur_xmax - cur_xmin
    span_x_axis = cur_ymax - cur_ymin
    if min_span_y is not None and min_span_y > 0 and span_y_axis < min_span_y:
        cx = (cur_xmin + cur_xmax) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(cx - half, cx + half)
    if min_span_x is not None and min_span_x > 0 and span_x_axis < min_span_x:
        cy = (cur_ymin + cur_ymax) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(cy - half, cy + half)

    # Fixed tick spacing (grid size)
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))

    plt.legend()
    plt.title(instruction + '\n' + log_path)
    plt.xlabel("Y (right)")
    plt.ylabel("X (forward)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_3d_trajectory_from_log(log_path: str, save_path: str, instruction: str, target: Optional[List[float]], reference_path: Optional[List[List[float]]] = None, tick_x: Optional[float] = None, tick_y: Optional[float] = None, tick_z: Optional[float] = None, min_span_x: Optional[float] = None, min_span_y: Optional[float] = None, min_span_z: Optional[float] = None) -> None:
    """Plot 3D trajectory from log and save the image.

    - Fixed tick step: use tick_x/tick_y/tick_z to specify grid spacing
    - Minimum span: use min_span_x/min_span_y/min_span_z to expand view when data span is smaller
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    if not isinstance(log, list) or len(log) == 0:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (no data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        if tick_z is not None and tick_z > 0:
            ax.zaxis.set_major_locator(MultipleLocator(tick_z))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.array([item['state'][0] + item['state'][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (invalid data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    trajectory = np.concatenate(
        [np.zeros((1, trajectory.shape[1]), dtype=trajectory.dtype), trajectory], axis=0
    )
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length =  max(PLOT_YAW_ARROW_MIN_LEN_3D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length
    dz = np.zeros_like(dx)

    fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(y, x, z, color='tab:blue', linewidth=2.0, label='Pred (closed loop)')
    if reference_path is not None:
        gt = np.asarray(reference_path, dtype=np.float64)
        if gt.ndim == 2 and gt.shape[0] > 0 and gt.shape[1] >= 3:
            ax.plot(
                gt[:, 1], gt[:, 0], gt[:, 2], color='black', linestyle='--',
                linewidth=2.0, label='GT reference',
            )
            ax.scatter(gt[-1, 1], gt[-1, 0], gt[-1, 2], color='black', marker='x', s=60)
    ax.quiver(y, x, z, dy, dx, dz, color='green', length=1.0, normalize=False, linewidth=0.5, label='Yaw direction')
    ax.scatter(y[0], x[0], z[0], color='blue', s=50)
    ax.text(y[0], x[0], z[0], 'Start', color='blue', fontsize=12)
    if target is not None:
        ax.scatter(target[1], target[0], target[2], color='red', s=50)
        ax.text(target[1], target[0], target[2], 'Target', color='red', fontsize=12)

    # Expand to minimum span without shrinking. Axis mapping: x-axis shows Y, y-axis shows X
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_zlim = ax.get_zlim()
    span_y = cur_xlim[1] - cur_xlim[0]
    span_x = cur_ylim[1] - cur_ylim[0]
    span_z = cur_zlim[1] - cur_zlim[0]
    if min_span_y is not None and min_span_y > 0 and span_y < min_span_y:
        c = (cur_xlim[0] + cur_xlim[1]) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(c - half, c + half)
    if min_span_x is not None and min_span_x > 0 and span_x < min_span_x:
        c = (cur_ylim[0] + cur_ylim[1]) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(c - half, c + half)
    if min_span_z is not None and min_span_z > 0 and span_z < min_span_z:
        c = (cur_zlim[0] + cur_zlim[1]) / 2.0
        half = min_span_z / 2.0
        ax.set_zlim(c - half, c + half)

    # Fixed tick spacing
    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))
    if tick_z is not None and tick_z > 0:
        ax.zaxis.set_major_locator(MultipleLocator(tick_z))

    # Equal aspect box: use current axis limits to avoid flattening when Z is zero
    _xl = ax.get_xlim(); _yl = ax.get_ylim(); _zl = ax.get_zlim()
    ax.set_box_aspect(((max(_xl[1]-_xl[0], 1e-6)), (max(_yl[1]-_yl[0], 1e-6)), (max(_zl[1]-_zl[0], 1e-6))))
    ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction}")
    ax.set_xlabel("Y (right)")
    ax.set_ylabel("X (forward)")
    ax.set_zlabel("Z (up)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=-100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _policy_pil_image(image: np.ndarray) -> Image.Image:
    """Convert this UnrealCV install's OpenCV-decoded BGR frame to RGB.

    ``unrealcv.api.Decoder.decode_bmp`` calls ``cv2.imdecode`` and returns the
    resulting BGR array unchanged.  The training dataset is decoded by PIL in
    RGB order, so wrapping that array directly with ``Image.fromarray`` swaps
    red and blue at policy inference time.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 UnrealCV image, got {array.shape}")
    rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
    return Image.fromarray(rgb, mode="RGB")


def _save_policy_image(path: str, image: np.ndarray) -> None:
    """Save the RGB image exactly as it will be sent to the policy."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil = _policy_pil_image(image)
    if pil.size != IMG_INPUT_SIZE:
        pil = pil.resize(IMG_INPUT_SIZE)
    pil.save(path)


def _camera_probe_ids(env: Any, spec: str) -> List[int]:
    ids: List[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "active":
            try:
                ids.append(int(env.unwrapped.cam_list[0]))
            except Exception:
                pass
        else:
            try:
                ids.append(int(token))
            except ValueError:
                logger.warning("Ignoring invalid camera probe id: %s", token)
    out: List[int] = []
    for cam_id in ids:
        if cam_id not in out:
            out.append(cam_id)
    return out


def _camera_pose_debug(env: Any, cam_ids: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "player_list": list(getattr(env.unwrapped, "player_list", [])),
        "cam_list": list(getattr(env.unwrapped, "cam_list", [])),
    }
    try:
        obj = env.unwrapped.player_list[0]
        out["active_object"] = obj
        out["active_agent_cfg"] = dict(env.unwrapped.agents.get(obj, {}))
        out["obj_location"] = list(env.unwrapped.unrealcv.get_obj_location(obj))
        out["obj_rotation"] = list(env.unwrapped.unrealcv.get_obj_rotation(obj))
    except Exception as e:
        out["obj_error"] = str(e)
    cams: Dict[str, Any] = {}
    for cam_id in cam_ids:
        item: Dict[str, Any] = {}
        try:
            item["location"] = list(env.unwrapped.unrealcv.get_cam_location(cam_id))
        except Exception as e:
            item["location_error"] = str(e)
        try:
            item["rotation"] = list(env.unwrapped.unrealcv.get_cam_rotation(cam_id))
        except Exception as e:
            item["rotation_error"] = str(e)
        cams[str(cam_id)] = item
    out["cameras"] = cams
    return out


def _policy_camera_id(env: Any, use_agent_camera: bool) -> int:
    if use_agent_camera:
        try:
            return int(env.unwrapped.cam_list[0])
        except Exception:
            obj = env.unwrapped.player_list[0]
            return int(env.unwrapped.agents[obj]["cam_id"])
    return 0


def control_loop(
    initial_pos: List[float],
    env: Any,
    instruction: str,
    max_steps: Optional[int],
    trajectory_log: List[Dict[str, Any]],
    server_url: str,
    request_timeout: float = 30.0,
    debug_obs_dir: Optional[str] = None,
    debug_obs_every: int = 1,
    debug_camera_probe: str = "",
    use_agent_camera: bool = False,
    freeze_policy_image: bool = False,
    horizontal_flip_policy_image: bool = False,
    task_name: str = "task",
    gt_reference_path: Optional[List[List[float]]] = None,
    gt_warmup_steps: int = 0,
    gt_action_only: bool = False,
    disable_small_motion_stop: bool = False,
    reset_drone_motion_per_task: bool = False,
) -> None:
    """Main control loop: capture image/state, call inference, act in env, log trajectory."""
    initial_x, initial_y, initial_z = initial_pos[0:3]
    initial_yaw = initial_pos[4]
    _reset_particles(env)
    initial_pose_debug = _set_and_wait_initial_pose(
        env,
        initial_pos,
        use_agent_camera=use_agent_camera,
        reset_drone_motion=reset_drone_motion_per_task,
    )
    logger.info(
        "Initial UAV pose converged: xyz_error=%.6f yaw_error=%.6f deg",
        float(initial_pose_debug["location_error"]),
        float(initial_pose_debug["yaw_error"]),
    )
    image = _capture_policy_image(env, use_agent_camera)
    # Diagnostic ablation: keep the environment and mounted camera running as
    # usual, but always send the task's initial onboard observation to the
    # policy.  Pose and language remain unchanged, isolating reliance on Ft.
    frozen_policy_image = image.copy() if freeze_policy_image else None

    def policy_image_for_request(current_image: Any) -> Image.Image:
        """Return the exact policy input, optionally mirrored for diagnostics."""
        policy_image = _policy_pil_image(
            frozen_policy_image if frozen_policy_image is not None else current_image
        )
        if horizontal_flip_policy_image:
            policy_image = policy_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return policy_image

    current_pose: List[float] = [0, 0, 0, 0]
    logger.info('Start control loop!')
    last_pose: Optional[List[float]] = None
    small_count = 0
    step_count = 0
    predict_count = 0
    debug_jsonl = os.path.join(debug_obs_dir, "obs_actions.jsonl") if debug_obs_dir else None
    if debug_jsonl and os.path.exists(debug_jsonl):
        os.remove(debug_jsonl)

    def transform_to_global(x: float, y: float, initial_yaw: float) -> Tuple[float, float]:
        theta = np.radians(initial_yaw)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        global_x = x * cos_theta - y * sin_theta
        global_y = x * sin_theta + y * cos_theta
        return global_x, global_y

    def normalize_angle(angle: float) -> float:
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    try:
        if gt_action_only:
            if not gt_reference_path or len(gt_reference_path) < 2:
                raise ValueError("--gt_action_only requires reference_path_preprocessed")
            logger.info(
                "GT-action chain: executing %d reference transitions through the policy action convention",
                len(gt_reference_path) - 1,
            )
            for gt_index in range(len(gt_reference_path) - 1):
                expected_before = _processed_gt_pose(gt_reference_path[gt_index])
                expected_after = _processed_gt_pose(gt_reference_path[gt_index + 1])
                gt_action = _yaw_local_action(expected_before, expected_after)
                commanded_pose = _integrate_yaw_local_action(
                    np.asarray(current_pose, dtype=np.float64), gt_action
                )
                pre_path = post_path = None
                if debug_obs_dir and (debug_obs_every <= 1 or gt_index % debug_obs_every == 0):
                    pre_path = os.path.join(debug_obs_dir, f"{task_name}_gt_{gt_index:04d}_pre.png")
                    _save_policy_image(pre_path, image)
                _execute_relative_pose(env, commanded_pose, initial_pos, use_agent_camera)
                achieved_pose, raw_object_rotation = _read_relative_position(
                    env, initial_pos, commanded_pose[3]
                )
                image = _capture_policy_image(env, use_agent_camera)
                if debug_obs_dir and (debug_obs_every <= 1 or gt_index % debug_obs_every == 0):
                    post_path = os.path.join(debug_obs_dir, f"{task_name}_gt_{gt_index:04d}_post.png")
                    _save_policy_image(post_path, image)
                pose_error = achieved_pose - expected_after
                pose_error[3] = _wrap_degrees(pose_error[3])
                if debug_jsonl:
                    _append_jsonl(debug_jsonl, {
                        "event": "gt_action_executed",
                        "task": task_name,
                        "step": gt_index,
                        "expected_before": expected_before.tolist(),
                        "gt_action_local_m_rad": gt_action.tolist(),
                        "commanded_pose": commanded_pose.tolist(),
                        "expected_after": expected_after.tolist(),
                        "achieved_pose": achieved_pose.tolist(),
                        "raw_object_rotation": raw_object_rotation,
                        "achieved_minus_gt": pose_error.tolist(),
                        "pre_image": pre_path,
                        "post_image": post_path,
                    })
                current_pose = commanded_pose.tolist()
                trajectory_log.append({
                    'state': [current_pose[:3], [0, current_pose[3], 0]],
                    'source': 'gt_action_chain',
                    'gt_action': gt_action.tolist(),
                    'expected_state': expected_after.tolist(),
                })
                step_count += 1
                time.sleep(0.1)
            return

        # Diagnostic-only oracle prefix.  Feed F0..F{K-1} to the policy while
        # executing K ground-truth transitions, then begin genuine closed-loop
        # control from FK.  With K=3 the next request has a complete four-frame
        # history.  These transitions must not be counted as model success.
        warmup_steps = max(0, int(gt_warmup_steps))
        if warmup_steps:
            if not gt_reference_path or len(gt_reference_path) < 2:
                raise ValueError("--gt_warmup_steps requires reference_path_preprocessed in the task JSON")
            warmup_steps = min(warmup_steps, len(gt_reference_path) - 1)
            logger.info("Oracle GT warmup: executing %d transitions before closed-loop control", warmup_steps)
            for warmup_index in range(warmup_steps):
                debug_save_path = None
                if debug_obs_dir and (debug_obs_every <= 1 or predict_count % debug_obs_every == 0):
                    debug_save_path = os.path.join(
                        debug_obs_dir, f"{task_name}_warmup_{warmup_index:04d}_sent.png"
                    )
                response = send_prediction_request(
                    image=policy_image_for_request(image),
                    proprio=np.asarray(current_pose),
                    instr=instruction,
                    server_url=server_url,
                    debug_save_path=debug_save_path,
                    request_timeout=request_timeout,
                )
                if response is None:
                    raise RuntimeError(f"Policy request failed during GT warmup step {warmup_index}")
                if debug_jsonl:
                    _append_jsonl(debug_jsonl, {
                        "event": "gt_warmup_prediction_ignored",
                        "task": task_name,
                        "predict_index": predict_count,
                        "warmup_index": warmup_index,
                        "current_pose_before": current_pose,
                        "sent_image": debug_save_path,
                        "server_debug": response.get("debug", {}),
                        "instruction": instruction,
                    })
                predict_count += 1

                expected_before = _processed_gt_pose(gt_reference_path[warmup_index])
                expected_after = _processed_gt_pose(gt_reference_path[warmup_index + 1])
                gt_action = _yaw_local_action(expected_before, expected_after)
                commanded_pose = _integrate_yaw_local_action(
                    np.asarray(current_pose, dtype=np.float64), gt_action
                )
                _execute_relative_pose(env, commanded_pose, initial_pos, use_agent_camera)
                achieved_pose, raw_object_rotation = _read_relative_position(
                    env, initial_pos, commanded_pose[3]
                )
                pose_error = achieved_pose - expected_after
                pose_error[3] = _wrap_degrees(pose_error[3])
                current_pose = commanded_pose.tolist()
                image = _capture_policy_image(env, use_agent_camera)
                trajectory_log.append({
                    'state': [current_pose[:3], [0, current_pose[3], 0]],
                    'source': 'gt_warmup',
                    'gt_action': gt_action.tolist(),
                    'expected_state': expected_after.tolist(),
                    'achieved_minus_gt': pose_error.tolist(),
                })
                if debug_jsonl:
                    _append_jsonl(debug_jsonl, {
                        "event": "gt_warmup_action_executed",
                        "task": task_name,
                        "warmup_index": warmup_index,
                        "gt_action_local_m_rad": gt_action.tolist(),
                        "commanded_pose": commanded_pose.tolist(),
                        "expected_after": expected_after.tolist(),
                        "achieved_pose": achieved_pose.tolist(),
                        "raw_object_rotation": raw_object_rotation,
                        "achieved_minus_gt": pose_error.tolist(),
                    })
                step_count += 1
                last_pose = list(current_pose)
                time.sleep(0.1)

        while True:
            set_cam(env, use_agent_camera=use_agent_camera)
            logger.debug(f"current_pose: {current_pose}")
            debug_save_path = None
            probe_paths: Dict[str, str] = {}
            cam_probe_ids = _camera_probe_ids(env, debug_camera_probe) if debug_obs_dir and debug_camera_probe else []
            if debug_obs_dir and (debug_obs_every <= 1 or predict_count % debug_obs_every == 0):
                debug_save_path = os.path.join(debug_obs_dir, f"{task_name}_predict_{predict_count:04d}_sent.png")
                for cam_id in cam_probe_ids:
                    probe_path = os.path.join(debug_obs_dir, f"{task_name}_predict_{predict_count:04d}_cam{cam_id}.png")
                    try:
                        _save_policy_image(probe_path, env.unwrapped.unrealcv.get_image(cam_id, 'lit'))
                        probe_paths[str(cam_id)] = probe_path
                    except Exception as e:
                        logger.warning("Failed to save camera probe cam%s: %s", cam_id, e)
            t1 = time.time()
            response = send_prediction_request(
                image=policy_image_for_request(image),
                proprio=np.array(current_pose),
                instr=instruction,
                server_url=server_url,
                debug_save_path=debug_save_path,
                request_timeout=request_timeout,
            )
            t2 = time.time()
            # logger.info(f"Prediction time: {t2 - t1} seconds")
            if response is None:
                logger.warning("No valid response, ending control")
                break
            if debug_jsonl:
                _append_jsonl(debug_jsonl, {
                    "event": "predict_response",
                    "task": task_name,
                    "predict_index": predict_count,
                    "step_count_before": step_count,
                    "current_pose_before": current_pose,
                    "sent_image": debug_save_path,
                    "camera_probe_images": probe_paths,
                    "camera_debug": _camera_pose_debug(env, cam_probe_ids),
                    "num_actions": len(response.get("action", [])) if isinstance(response.get("action"), list) else None,
                    "server_debug": response.get("debug", {}),
                    "instruction": instruction,
                })
            predict_count += 1
            
            try:
                action_poses = response.get('action')
                if not isinstance(action_poses, list) or len(action_poses) == 0:
                    logger.warning("Response 'action' is empty or invalid, stopping.")
                    break
                for i, action_pose in enumerate(action_poses):
                    if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
                        logger.warning(f"Invalid action element at {i}: {action_pose}")
                        continue
                    relative_x, relative_y = float(action_pose[0]), float(action_pose[1])
                    relative_z = float(action_pose[2])
                    relative_yaw = float(np.degrees(action_pose[3]))
                    relative_yaw = (relative_yaw + 180) % 360 - 180
                    global_x, global_y = transform_to_global(relative_x, relative_y, initial_yaw)
                    absolute_yaw = normalize_angle(relative_yaw + initial_yaw)
                    absolute_pos = [
                        global_x + initial_x,
                        global_y + initial_y,
                        relative_z + initial_z,
                        absolute_yaw
                    ]
                    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], absolute_pos[:3])
                    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], absolute_pos[3] - 180)
                    set_cam(env, use_agent_camera=use_agent_camera)
                    
                    if i == len(action_poses) - 1:
                        current_pose = [relative_x, relative_y, relative_z, relative_yaw]
                        image = _capture_policy_image(env, use_agent_camera)
                        try:
                            cv2.imwrite(DEBUG_IMAGE_PATH, image)
                        except Exception as e:
                            logger.debug(f"Failed to write debug image: {e}")
                    step_count += 1
                    trajectory_log.append({'state': [[relative_x, relative_y, relative_z], [0, relative_yaw, 0]]})
                    pose_now = [relative_x, relative_y, relative_z, relative_yaw]
                    if last_pose is not None and not disable_small_motion_stop:
                        diffs = [abs(a - b) for a, b in zip(pose_now, last_pose)]
                        if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                            small_count += 1
                        else:
                            small_count = 0
                        if small_count >= ACTION_SMALL_STEPS:
                            logger.info(f"Detected x,y,z,yaw continuous {ACTION_SMALL_STEPS} steps change is very small, automatically end task.")
                            return
                    last_pose = pose_now
                    time.sleep(0.1)
                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"Already inferred {max_steps} steps, automatically switch to next task.")
                    break
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                break
            try:
                if response.get('done') is True:
                    logger.info("Server returned done=True. Ending control loop.")
                    return
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Control loop error: {e}")


def set_cam(env: Any, use_agent_camera: bool = False) -> None:
    """Compute and set camera pose based on current object pose."""
    if use_agent_camera:
        obj = env.unwrapped.player_list[0]
        cfg = env.unwrapped.agents.get(obj, {})
        env.unwrapped.unrealcv.set_cam(
            obj,
            cfg.get("relative_location", [0, 0, 0]),
            cfg.get("relative_rotation", [0, 0, 0]),
        )
        env.unwrapped.unrealcv.cam = env.unwrapped.unrealcv.get_camera_config()
        env.unwrapped.update_camera_assignments()
        return

    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]
    env.unwrapped.unrealcv.set_cam(0, cam_loc, cam_rot)


def _sync_unrealcv(env: Any, command: str) -> Any:
    """Execute one UnrealCV command synchronously.

    The convenience setters in unrealcv 1.1.7 use async requests.  Reusing one
    target actor for hundreds of tasks can therefore race appearance/location
    updates.  Target setup uses this helper when verification is requested.
    """
    return env.unwrapped.unrealcv.client.request(command)


def _reset_target_actors(env: Any) -> None:
    """Synchronously hide and disable physics on the persistent target actors.

    Do not destroy and immediately respawn an actor with the same name.  This
    packaged UE4 build processes destruction on a later game tick and fatally
    rejects the new actor because the old UObject name still exists.
    """
    commands = (
        "vbp BP_Character_21 set_phy 0",
        "vset /object/BP_Character_21/location 0 0 -1000",
        "vbp BP_Character_22 set_phy 0",
        "vset /object/BP_Character_22/location 0 0 -1000",
    )
    for command in commands:
        _sync_unrealcv(env, command)


def _verified_target_location(
    env: Any,
    name: str,
    location: List[float],
    *,
    retries: int,
    tolerance: float,
    hard_tolerance: float,
) -> None:
    """Place an actor and verify its horizontal location.

    Character and vehicle blueprints snap their root component to the terrain,
    so their reported Z can legitimately differ from the task JSON altitude.
    X/Y must still match: those coordinates determine target bearing/distance.
    """
    expected = np.asarray(location, dtype=np.float64)
    last_actual: Optional[np.ndarray] = None
    for attempt in range(max(1, int(retries))):
        _sync_unrealcv(
            env,
            f"vset /object/{name}/location "
            f"{float(expected[0])} {float(expected[1])} {float(expected[2])}",
        )
        time.sleep(0.15)
        try:
            last_actual = np.asarray(
                env.unwrapped.unrealcv.get_obj_location(name), dtype=np.float64
            )
        except Exception as exc:
            logger.warning(
                "Target %s location readback failed on attempt %d/%d: %s",
                name,
                attempt + 1,
                max(1, int(retries)),
                exc,
            )
            continue
        if last_actual.shape[0] >= 3 and float(
            np.linalg.norm(last_actual[:2] - expected[:2])
        ) <= float(tolerance):
            z_delta = float(last_actual[2] - expected[2])
            if abs(z_delta) > float(tolerance):
                logger.info(
                    "Target %s XY verified; UE terrain snap changed Z "
                    "from %.3f to %.3f (delta=%+.3f).",
                    name,
                    float(expected[2]),
                    float(last_actual[2]),
                    z_delta,
                )
            return
    if last_actual is not None and last_actual.shape[0] >= 3:
        xy_error = float(np.linalg.norm(last_actual[:2] - expected[:2]))
        if xy_error <= float(hard_tolerance):
            logger.warning(
                "Target %s was collision-adjusted in XY by %.3f Unreal units "
                "(soft tolerance %.3f, hard tolerance %.3f); accepting actual=%s.",
                name,
                xy_error,
                float(tolerance),
                float(hard_tolerance),
                last_actual[:3].tolist(),
            )
            return
    raise RuntimeError(
        f"Failed to place target {name} in XY: expected={expected[:3].tolist()} "
        f"actual={None if last_actual is None else last_actual[:3].tolist()} "
        f"after {max(1, int(retries))} attempts."
    )


def create_obj_if_needed(
    env: Any,
    obj_info: Optional[Dict[str, Any]],
    *,
    verify_per_task: bool = False,
    placement_retries: int = 3,
    placement_tolerance: float = 2.0,
    placement_hard_tolerance: float = 100.0,
) -> None:
    """Create/place task objects, optionally resetting and verifying actors."""
    if verify_per_task:
        _reset_target_actors(env)
    if obj_info is None:
        return
    use_obj = obj_info.get('use_obj', None)
    obj_id = obj_info.get('obj_id', None)
    obj_pos = obj_info.get('obj_pos', None)
    obj_rot = obj_info.get('obj_rot', None)
    if use_obj == 1:
        if verify_per_task:
            _sync_unrealcv(env, f"vbp BP_Character_21 set_app {int(obj_id)}")
            _sync_unrealcv(
                env,
                f"vset /object/BP_Character_21/rotation "
                f"{float(obj_rot[2])} {float(obj_rot[1])} {float(obj_rot[0])}",
            )
            _verified_target_location(
                env,
                "BP_Character_21",
                obj_pos,
                retries=placement_retries,
                tolerance=placement_tolerance,
                hard_tolerance=placement_hard_tolerance,
            )
        else:
            env.unwrapped.unrealcv.set_appearance("BP_Character_21", obj_id)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
    elif use_obj == 2:
        car_pos = [obj_pos[0], obj_pos[1], 0]
        if verify_per_task:
            _sync_unrealcv(env, "vbp BP_Character_22 set_app 2")
            _sync_unrealcv(env, "vbp BP_Character_22 set_phy 0")
            _sync_unrealcv(
                env,
                f"vset /object/BP_Character_22/rotation "
                f"{float(obj_rot[2])} {float(obj_rot[1])} {float(obj_rot[0])}",
            )
            _verified_target_location(
                env,
                "BP_Character_22",
                car_pos,
                retries=placement_retries,
                tolerance=placement_tolerance,
                hard_tolerance=placement_hard_tolerance,
            )
        else:
            env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", car_pos)
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot)
            env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", car_pos)

    if use_obj in [1, 2]:
        logger.debug("Object placement finished for use_obj=%s", use_obj)
        time.sleep(SLEEP_SHORT_S)


def force_camera_zero(env: Any) -> None:
    """Use camera 0 for all agents and skip automatic camera matching."""
    unwrapped = env.unwrapped

    def _use_camera_zero() -> None:
        for obj in unwrapped.player_list:
            if obj in unwrapped.agents:
                unwrapped.agents[obj]['cam_id'] = 0
        unwrapped.cam_list = [0 for _ in unwrapped.player_list]

    unwrapped.update_camera_assignments = _use_camera_zero
    _use_camera_zero()


def patch_remove_agent_camera_wait() -> None:
    """Avoid a UAV-Flow-Eval busy-wait when pruning non-drone agents.

    The stock env removes all non-selected agents in ``init_agents()``.  After
    destroying an actor it waits for Unreal's camera count to change:

        while self.unrealcv.get_camera_num() > len(last_cam_list) + 1: pass

    On this UE package/headless setup that condition can stay true forever,
    so startup hangs at "waiting for remove agent BP_Character_2...".  This
    patch keeps the official agent/camera assignment logic, but removes only
    that fragile wait loop.
    """
    from gym_unrealcv.envs.base_env import UnrealCv_base

    def _remove_agent_no_camera_wait(self: Any, name: str) -> None:
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.unrealcv.destroy_obj(name)
        self.agents.pop(name)
        logger.info("Removed non-active agent %s without camera-count wait.", name)

    UnrealCv_base.remove_agent = _remove_agent_no_camera_wait


def patch_unrealcv_fixed_camera() -> None:
    """Avoid UnrealCV /cameras discovery on Linux headless runs."""
    from unrealcv.api import UnrealCv_API

    def _fixed_get_camera_num(self: Any) -> int:
        return 1

    def _fixed_get_camera_config(self: Any) -> Dict[int, Dict[str, Any]]:
        return {
            0: {
                'location': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0],
                'fov': 90.0,
            }
        }

    def _fixed_init_map(self: Any) -> None:
        self.cam = _fixed_get_camera_config(self)
        self.obj_dict = {}

    def _safe_destroy_obj(self: Any, obj: str) -> None:
        self.client.request(f'vset /object/{obj}/destroy', -1)
        self.obj_dict.pop(obj, None)

    UnrealCv_API.get_camera_num = _fixed_get_camera_num
    UnrealCv_API.get_camera_config = _fixed_get_camera_config
    UnrealCv_API.init_map = _fixed_init_map
    UnrealCv_API.destroy_obj = _safe_destroy_obj


def lightweight_reset_for_closed_loop(
    env: Any,
    use_random_population: bool = False,
    population_num: int = 2,
    force_cam0: bool = True,
) -> None:
    """Launch Unreal and initialize actors without the full Gym reset path."""
    unwrapped = env.unwrapped
    if not unwrapped.launched:
        unwrapped.launched = unwrapped.launch_ue_env()
        unwrapped.init_agents()
        unwrapped.init_objects()

    if use_random_population:
        unwrapped.num_agents = population_num
        unwrapped.set_population(unwrapped.num_agents)

    if force_cam0:
        force_camera_zero(env)
    unwrapped.count_close = 0
    unwrapped.count_steps = 0
    unwrapped.count_eps += 1
    unwrapped.cam_flag = unwrapped.get_cam_flag(unwrapped.observation_type)


def reset_model(server_url: str) -> None:
    """Call server /reset to reset the model."""
    try:
        resp = requests.post(server_url.replace('/predict', '/reset'), timeout=10)
        logger.info(f"Model reset response: {resp.status_code}")
    except Exception as e:
        logger.error(f"Model reset failed: {e}")


def world_to_local(target: List[float], init_pos: List[float]) -> List[float]:
    """Convert world-frame target coordinates to the first-frame local frame."""
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    z_rel = dz
    return [x_rel, y_rel, z_rel]

if __name__ == '__main__':
    # ====== Default parameters ======
    DEFAULT_ENV_ID = 'UnrealTrack-DowntownWest-ContinuousColor-v0'
    DEFAULT_TIME_DILATION = 10
    DEFAULT_SEED = 0
    DEFAULT_JSON_FOLDER = r'./test_jsons'
    DEFAULT_IMAGES_DIR = f'./results/{DEFAULT_ENV_ID}/openvla'
    DEFAULT_SERVER_PORT = 5007
    DEFAULT_MAX_STEPS = 100
    DEFAULT_INSTRUCTION_TYPE = "instruction"
    DEFAULT_RESOLUTION = 256
    DEFAULT_SLEEP_TIME = 90
    # Fixed tick spacing (grid size)
    DEFAULT_TICK_X: Optional[float] = 100
    DEFAULT_TICK_Y: Optional[float] = 100
    DEFAULT_TICK_Z: Optional[float] = 100
    # Minimum display span
    DEFAULT_MIN_SPAN_X: Optional[float] = 400
    DEFAULT_MIN_SPAN_Y: Optional[float] = 400
    DEFAULT_MIN_SPAN_Z: Optional[float] = 400

    import argparse
    parser = argparse.ArgumentParser(description='Batch run UAV control with instruction-conditioned policy')
    parser.add_argument("-e", "--env_id", default=DEFAULT_ENV_ID, help='Environment ID to run')
    parser.add_argument("-t", '--time_dilation', default=DEFAULT_TIME_DILATION, type=int, help='Time dilation parameter to keep FPS in simulator')
    parser.add_argument("-s", '--seed', default=DEFAULT_SEED, type=int, help='Random seed')
    parser.add_argument("-f", '--json_folder', default=DEFAULT_JSON_FOLDER, help='Folder path containing batch task json files')
    parser.add_argument("-o", '--images_dir', default=DEFAULT_IMAGES_DIR, help='Directory to save images and trajectory logs')
    parser.add_argument("-p", '--server_port', default=DEFAULT_SERVER_PORT, type=int, help='Inference server port')
    parser.add_argument('--request_timeout', default=30.0, type=float, help='Seconds to wait for each policy response; use a larger value for CPU inference.')
    parser.add_argument("-m", '--max_steps', default=DEFAULT_MAX_STEPS, type=int, help='Maximum inference steps')
    parser.add_argument("-i", "--instruction_type", default=DEFAULT_INSTRUCTION_TYPE, choices=["instruction", "instruction_unified"], help='Choose which field to use: instruction or instruction_unified')
    parser.add_argument(
        '--instruction_override',
        default='',
        help=(
            'Optional instruction used for every selected task instead of the JSON instruction. '
            'Useful for closed-loop language/image counterfactual tests.'
        ),
    )
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    parser.add_argument('--offscreen', action='store_true', help='Run Unreal with offscreen rendering')
    parser.add_argument('--gpu_id', default=None, help='GPU id passed to Unreal -graphicsadapter')
    parser.add_argument('--sleep_time', default=DEFAULT_SLEEP_TIME, type=int, help='Seconds to wait for Unreal to launch')
    parser.add_argument('--resolution', default=DEFAULT_RESOLUTION, type=int, help='Square UnrealCV capture resolution')
    parser.add_argument('--limit_tasks', default=None, type=int, help='Maximum number of task JSON files to run')
    parser.add_argument(
        '--start_task_index',
        default=1,
        type=int,
        help='One-based start index after task-id/object-task filtering.',
    )
    parser.add_argument(
        '--overwrite_existing',
        action='store_true',
        help=(
            'Overwrite outputs for selected tasks. Existing trajectory JSON, '
            '2D/3D plots, and that task debug-observation directory are removed first.'
        ),
    )
    parser.add_argument(
        '--object_tasks_only',
        action='store_true',
        help='Run only JSON tasks whose use_obj is 1 (person/animal) or 2 (vehicle).',
    )
    parser.add_argument(
        '--capture_initial_only',
        action='store_true',
        help=(
            'Place the task target and UAV, save the initial policy-camera image, '
            'then continue to the next task without calling the policy server.'
        ),
    )
    parser.add_argument(
        '--task_ids',
        default='',
        help=(
            'Comma-separated task/episode ids to run, with or without the .json suffix. '
            'When provided, only these files are used and their command-line order is preserved.'
        ),
    )
    parser.add_argument('--debug_obs_dir', default=None, help='Optional directory to save the exact resized images sent to the policy server.')
    parser.add_argument('--debug_obs_every', default=1, type=int, help='Save every N policy requests when --debug_obs_dir is set; <=1 saves all.')
    parser.add_argument('--debug_camera_probe', default='', help='Comma-separated camera ids to save alongside the sent image, e.g. "0,1,active,9". Does not change policy input.')
    parser.add_argument('--use_agent_camera', action='store_true', help='Use env.cam_list[0] / active agent mounted camera as policy input instead of camera 0.')
    parser.add_argument(
        '--verify_target_per_task',
        '--recreate_target_per_task',
        dest='verify_target_per_task',
        action='store_true',
        default=True,
        help=(
            'Synchronously reset and place the persistent person/dog/car actor '
            'for every task, then verify its location by readback (default: enabled). The old '
            '--recreate_target_per_task spelling is retained as an alias.'
        ),
    )
    parser.add_argument(
        '--no_verify_target_per_task',
        dest='verify_target_per_task',
        action='store_false',
        help='Disable deterministic target reset/readback verification.',
    )
    parser.add_argument(
        '--target_placement_retries',
        default=3,
        type=int,
        help='Placement attempts used with --verify_target_per_task.',
    )
    parser.add_argument(
        '--target_placement_tolerance',
        default=2.0,
        type=float,
        help='Maximum Unreal-unit target XY location readback error.',
    )
    parser.add_argument(
        '--target_placement_hard_tolerance',
        default=100.0,
        type=float,
        help=(
            'Abort only when target XY error remains above this value after '
            'retries; smaller collision adjustment is logged and accepted.'
        ),
    )
    parser.add_argument(
        '--freeze_policy_image',
        action='store_true',
        help='Diagnostic ablation: repeatedly send the task initial onboard image while the environment and pose continue updating.',
    )
    parser.add_argument(
        '--horizontal_flip_policy_image',
        action='store_true',
        help=(
            'Diagnostic ablation: horizontally mirror only the image sent to the policy. '
            'The simulator observation, task instruction, pose, and control convention are unchanged.'
        ),
    )
    parser.add_argument('--gt_warmup_steps', default=0, type=int, help='Diagnostic only: execute this many GT reference transitions before starting closed-loop policy control.')
    parser.add_argument('--gt_action_only', action='store_true', help='Diagnostic only: execute the full GT action chain without calling the policy server.')
    parser.add_argument(
        '--reset_drone_motion_per_task',
        action='store_true',
        help=(
            'Before each task initial-pose teleport, synchronously clear the '
            'drone Blueprint movement state and disable physics. Intended for '
            'A/B testing mounted-camera occlusion from residual mesh tilt.'
        ),
    )
    parser.add_argument(
        '--disable_small_motion_stop',
        action='store_true',
        help='Diagnostic only: do not terminate after 10 nearly stationary actions.',
    )
    parser.add_argument('--use_camera_discovery', action='store_true', help='Use UnrealCV /cameras discovery instead of fixed camera 0')
    parser.add_argument('--full_reset', action='store_true', help='Use the original Gym reset path')
    parser.add_argument('--use_random_population', action='store_true', help='Enable RandomPopulationWrapper/set_population')
    # Fixed tick spacing (grid size)
    parser.add_argument('--tick_x', type=float, default=None, help='Tick step for X (forward) axis')
    parser.add_argument('--tick_y', type=float, default=None, help='Tick step for Y (right) axis')
    parser.add_argument('--tick_z', type=float, default=None, help='Tick step for Z (up) axis')
    # Minimum display span
    parser.add_argument('--min_span_x', type=float, default=None, help='Minimum span for X (forward) axis')
    parser.add_argument('--min_span_y', type=float, default=None, help='Minimum span for Y (right) axis')
    parser.add_argument('--min_span_z', type=float, default=None, help='Minimum span for Z (up) axis')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')

    tick_x = args.tick_x if args.tick_x is not None else DEFAULT_TICK_X
    tick_y = args.tick_y if args.tick_y is not None else DEFAULT_TICK_Y
    tick_z = args.tick_z if args.tick_z is not None else DEFAULT_TICK_Z

    min_span_x = args.min_span_x if args.min_span_x is not None else DEFAULT_MIN_SPAN_X
    min_span_y = args.min_span_y if args.min_span_y is not None else DEFAULT_MIN_SPAN_Y
    min_span_z = args.min_span_z if args.min_span_z is not None else DEFAULT_MIN_SPAN_Z

    server_url = f"http://127.0.0.1:{args.server_port}/predict"

    patch_remove_agent_camera_wait()

    if args.use_agent_camera:
        args.use_camera_discovery = True

    if not args.use_camera_discovery:
        patch_unrealcv_fixed_camera()
        logger.info("Using fixed UnrealCV camera 0; /cameras discovery is disabled.")

    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    env.unwrapped.agents_category = ['drone']
    if not args.use_agent_camera:
        force_camera_zero(env)
    env = configUE.ConfigUEWrapper(
        env,
        resolution=(args.resolution, args.resolution),
        offscreen=args.offscreen,
        gpu_id=args.gpu_id,
        sleep_time=args.sleep_time,
    )
    if args.use_random_population:
        env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env.seed(int(args.seed))
    if args.full_reset:
        env.reset()
    else:
        lightweight_reset_for_closed_loop(
            env,
            use_random_population=args.use_random_population,
            population_num=2,
            force_cam0=not args.use_agent_camera,
        )
    if not args.use_agent_camera:
        force_camera_zero(env)
    logger.info(
        "Policy camera mode: use_agent_camera=%s cam_list=%s player_list=%s",
        args.use_agent_camera,
        getattr(env.unwrapped, "cam_list", None),
        getattr(env.unwrapped, "player_list", None),
    )
    env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
    env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)

    json_folder = args.json_folder
    requested_task_ids = [
        item.strip() for item in str(args.task_ids).split(',') if item.strip()
    ]
    if requested_task_ids:
        json_files = []
        missing_task_ids = []
        for task_id in requested_task_ids:
            filename = task_id if task_id.endswith('.json') else f'{task_id}.json'
            candidate = os.path.join(json_folder, filename)
            if os.path.isfile(candidate):
                json_files.append(candidate)
            else:
                missing_task_ids.append(task_id)
        if missing_task_ids:
            parser.error(
                f"Task ids not found under {json_folder}: {missing_task_ids}"
            )
    else:
        json_files = glob.glob(os.path.join(json_folder, '*.json'))
    if args.object_tasks_only:
        object_json_files = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    task_payload = json.load(f)
                if isinstance(task_payload, dict) and task_payload.get('use_obj') in (1, 2):
                    object_json_files.append(json_file)
            except Exception as exc:
                logger.warning("Unable to inspect %s for use_obj: %s", json_file, exc)
        json_files = object_json_files
    start_task_index = max(1, int(args.start_task_index))
    if start_task_index > 1:
        json_files = json_files[start_task_index - 1:]
    if args.limit_tasks is not None:
        json_files = json_files[:args.limit_tasks]
    logger.info(f"Detected {len(json_files)} json task files")
    images_dir = args.images_dir
    os.makedirs(images_dir, exist_ok=True)

    # init object
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0,0,-1000])
    env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21",  [0,0,0])
    time.sleep(SLEEP_SHORT_S)
    env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [0,0,-1000])
    env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
    env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0,0,0])
    env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
    time.sleep(SLEEP_SHORT_S)


    for idx, json_file in enumerate(json_files):

        base_name = os.path.splitext(os.path.basename(json_file))[0]
        img2d_path = os.path.join(images_dir, base_name + '_2d.png')
        img3d_path = os.path.join(images_dir, base_name + '_3d.png')
        traj_json_path = os.path.join(images_dir, base_name + '.json')
        if args.overwrite_existing:
            for stale_path in (
                img2d_path,
                img3d_path,
                traj_json_path,
            ):
                if os.path.isfile(stale_path) or os.path.islink(stale_path):
                    os.remove(stale_path)
            if args.debug_obs_dir:
                stale_obs_dir = os.path.join(args.debug_obs_dir, base_name)
                if os.path.isdir(stale_obs_dir):
                    shutil.rmtree(stale_obs_dir)
        elif os.path.exists(img2d_path) and os.path.exists(img3d_path):
            if _trajectory_result_is_complete(traj_json_path):
                logger.info(f"{base_name} valid trajectory and images already exist, skipping.")
                continue
            logger.warning(
                "%s has blank/stale plots without a valid trajectory; rerunning.",
                base_name,
            )
            for stale_path in (img2d_path, img3d_path, traj_json_path):
                if os.path.isfile(stale_path) or os.path.islink(stale_path):
                    os.remove(stale_path)
            if args.debug_obs_dir:
                stale_obs_dir = os.path.join(args.debug_obs_dir, base_name)
                if os.path.isdir(stale_obs_dir):
                    shutil.rmtree(stale_obs_dir)
        logger.info(f"\n===== Start task {idx+1}/{len(json_files)} file: {json_file} =====")
        if not args.gt_action_only and not args.capture_initial_only:
            reset_model(server_url)
        with open(json_file, 'r') as f:
            manual_data = json.load(f)
        if not isinstance(manual_data, dict):
            logger.warning(f"Unsupported json format, skipping: {json_file}")
            continue
        # Choose instruction field based on argument (compatible with legacy/new format)
        if args.instruction_type == 'instruction_unified':
            instruction = manual_data.get('instruction_unified', manual_data.get('instruction', ''))
        else:
            instruction = manual_data.get('instruction', manual_data.get('instruction_unified', ''))
        if str(args.instruction_override).strip():
            instruction = str(args.instruction_override).strip()
        initial_pos = manual_data.get('initial_pos', None)
        target_pos = manual_data.get('target_pos', None)
        # Only create objects when both obj_id and use_obj are provided
        obj_info = None
        if 'obj_id' in manual_data and 'use_obj' in manual_data:
            if 'target_pos' in manual_data and isinstance(manual_data['target_pos'], list) and len(manual_data['target_pos']) == 6:
                obj_pos = manual_data['target_pos'][:3]
                obj_rot = manual_data['target_pos'][3:]
            else:
                obj_pos = manual_data.get('obj_pos', None)
                obj_rot = manual_data.get('obj_rot', [0, 0, 0])
            if obj_pos is not None:
                obj_info = {
                    'use_obj': manual_data['use_obj'],
                    'obj_id': manual_data['obj_id'],
                    'obj_pos': obj_pos,
                    'obj_rot': obj_rot
                }
        create_obj_if_needed(
            env,
            obj_info,
            verify_per_task=bool(args.verify_target_per_task),
            placement_retries=int(args.target_placement_retries),
            placement_tolerance=float(args.target_placement_tolerance),
            placement_hard_tolerance=float(args.target_placement_hard_tolerance),
        )
        set_cam(env, use_agent_camera=bool(args.use_agent_camera))
        time.sleep(SLEEP_SHORT_S)
        logger.info(f"instruction: {instruction}")
        trajectory_log: List[Dict[str, Any]] = []
        if not initial_pos or len(initial_pos) < 5:
            logger.error("Invalid or missing 'initial_pos' in task json; skipping.")
            continue
        if args.capture_initial_only:
            if not args.debug_obs_dir:
                raise ValueError("--capture_initial_only requires --debug_obs_dir")
            _reset_particles(env)
            try:
                initial_pose_debug = _set_and_wait_initial_pose(
                    env,
                    initial_pos,
                    use_agent_camera=bool(args.use_agent_camera),
                    reset_drone_motion=bool(args.reset_drone_motion_per_task),
                )
            except InitialPoseConvergenceError as exc:
                logger.error("Skipping task %s: %s", base_name, exc)
                continue
            image = _capture_policy_image(env, bool(args.use_agent_camera))
            capture_dir = os.path.join(args.debug_obs_dir, base_name)
            capture_path = os.path.join(
                capture_dir, f"{base_name}_predict_0000_sent.png"
            )
            _save_policy_image(capture_path, image)
            logger.info(
                "Initial-only capture saved: %s (xyz_error=%.6f yaw_error=%.6f deg)",
                capture_path,
                float(initial_pose_debug["location_error"]),
                float(initial_pose_debug["yaw_error"]),
            )
            continue
        try:
            control_loop(
                initial_pos,
                env=env,
                instruction=instruction,
                max_steps=args.max_steps,
                trajectory_log=trajectory_log,
                server_url=server_url,
                request_timeout=float(args.request_timeout),
                debug_obs_dir=os.path.join(args.debug_obs_dir, base_name) if args.debug_obs_dir else None,
                debug_obs_every=int(args.debug_obs_every),
                debug_camera_probe=str(args.debug_camera_probe),
                use_agent_camera=bool(args.use_agent_camera),
                freeze_policy_image=bool(args.freeze_policy_image),
                horizontal_flip_policy_image=bool(args.horizontal_flip_policy_image),
                task_name=base_name,
                gt_reference_path=manual_data.get('reference_path_preprocessed'),
                gt_warmup_steps=int(args.gt_warmup_steps),
                gt_action_only=bool(args.gt_action_only),
                disable_small_motion_stop=bool(args.disable_small_motion_stop),
                reset_drone_motion_per_task=bool(args.reset_drone_motion_per_task),
            )
        except InitialPoseConvergenceError as exc:
            logger.error("Skipping task %s: %s", base_name, exc)
            continue
        with open(traj_json_path, 'w') as f:
            json.dump(trajectory_log, f, indent=2)
        # Transform target_pos to the coordinate system of the first frame
        target_local = None
        if target_pos is not None and initial_pos is not None and len(initial_pos) > 0:
            target_local = world_to_local(target_pos, initial_pos)
        try:
            # The official reference belongs to the JSON instruction.  It is
            # not a valid GT after a counterfactual language override, so do
            # not draw a misleading Pred-vs-GT overlay in that case.
            reference_path = (
                None
                if str(args.instruction_override).strip()
                else manual_data.get('reference_path_preprocessed')
            )
            draw_2d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_2d.png'), instruction, target_local, reference_path=reference_path, tick_x=tick_x, tick_y=tick_y, min_span_x=min_span_x, min_span_y=min_span_y)
            draw_3d_trajectory_from_log(traj_json_path, os.path.join(images_dir, base_name + '_3d.png'), instruction, target_local, reference_path=reference_path, tick_x=tick_x, tick_y=tick_y, tick_z=tick_z, min_span_x=min_span_x, min_span_y=min_span_y, min_span_z=min_span_z)
            logger.info(f"Trajectory and images saved: {base_name}")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")
        logger.info(f"===== Task {idx+1} finished =====\n")
    env.close() 
