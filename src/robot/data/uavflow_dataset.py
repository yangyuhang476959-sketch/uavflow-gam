"""UAV-Flow parquet adapter for GAM training."""

from __future__ import annotations
from collections import OrderedDict
import io, json, math, multiprocessing as mp
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_UAVFLOW_TASK_CLASSES = (
    "Approach", "Retreat", "Pass", "Land", "Turn", "Move", "Shift",
    "Rotate", "Surround", "Ascend/Descend",
)


def _instruction_task_class(instruction):
    """Map released UAV-Flow unified instructions to official task classes."""
    text = " ".join(str(instruction).lower().strip().rstrip(".").split())
    if text.startswith("rotate "):
        return "Rotate"
    if text.startswith("turn to"):
        return "Turn"
    if text.startswith("circle around") or "surround" in text:
        return "Surround"
    if (
        text.startswith(("ascend ", "descend ", "take off"))
        or (text.startswith("land ") and len(text.split()) > 1 and text.split()[1][0].isdigit())
    ):
        return "Ascend/Descend"
    if "land" in text:
        return "Land"
    if any(token in text for token in (
        "away", "back off", "back away", "move off", "withdraw", "retreat",
        "stepping back",
    )):
        return "Retreat"
    if "through" in text or "past" in text or "cross" in text:
        return "Pass"
    if text.startswith("move ") and " meters at " in text:
        return "Shift"
    if text.startswith("move to a position ") and " meters from " in text:
        return "Move"
    # All remaining released templates are goal-directed Approach paraphrases.
    return "Approach"


def _stratified_episode_split(ids, logs, eval_ratio, split_seed):
    """Return a deterministic class-proportional episode split."""
    ratio = float(eval_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"Stratified eval_ratio must be in (0,1), got {ratio}.")
    groups = {name: [] for name in _UAVFLOW_TASK_CLASSES}
    for tid in ids:
        groups[_instruction_task_class(logs[tid][1])].append(tid)

    target_eval = max(1, min(len(ids) - 1, int(round(len(ids) * ratio))))
    exact = {name: len(values) * ratio for name, values in groups.items()}
    quota = {
        name: (max(1, int(math.floor(exact[name]))) if len(values) > 1 else 0)
        for name, values in groups.items()
    }
    while sum(quota.values()) < target_eval:
        candidates = [name for name, values in groups.items() if quota[name] < len(values) - 1]
        name = max(
            candidates,
            key=lambda key: (exact[key] - quota[key], -_UAVFLOW_TASK_CLASSES.index(key)),
        )
        quota[name] += 1
    while sum(quota.values()) > target_eval:
        candidates = [name for name in groups if quota[name] > 1]
        name = min(
            candidates,
            key=lambda key: (exact[key] - quota[key], _UAVFLOW_TASK_CLASSES.index(key)),
        )
        quota[name] -= 1

    rng = np.random.default_rng(int(split_seed))
    eval_ids = set()
    counts = {}
    for name in _UAVFLOW_TASK_CLASSES:
        values = np.asarray(sorted(groups[name]), dtype=object)
        if len(values):
            values = values[rng.permutation(len(values))]
        eval_ids.update(str(value) for value in values[:quota[name]].tolist())
        counts[name] = {
            "all": len(groups[name]),
            "train": len(groups[name]) - quota[name],
            "eval": quota[name],
        }
    return (
        [tid for tid in ids if tid not in eval_ids],
        [tid for tid in ids if tid in eval_ids],
        counts,
    )


def _rotation(roll, yaw, pitch):
    r,y,p = map(math.radians,(roll,yaw,pitch))
    cx,sx,cy,sy,cz,sz = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]],np.float64)
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]],np.float64)
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]],np.float64)
    return Rz@Ry@Rx


def _rpy(R):
    pitch=math.asin(float(np.clip(-R[2,0],-1,1)))
    roll=math.atan2(float(R[2,1]),float(R[2,2])); yaw=math.atan2(float(R[1,0]),float(R[0,0]))
    return roll,yaw,pitch


def _delta(a, b, convention, action_delta_mode="full_se3_legacy"):
    pa,pb=np.asarray(a[:3],np.float64),np.asarray(b[:3],np.float64)
    if convention == "yaw4d" and action_delta_mode == "openvla_yaw_only":
        # OpenVLA-UAV defines translation in the gravity-aligned heading frame:
        # rotate horizontal world displacement by current yaw only, while z
        # remains world vertical.  Relative yaw is wrapped to [-pi, pi).
        yaw = math.radians(float(a[4]))
        c, s = math.cos(yaw), math.sin(yaw)
        world = pb - pa
        trans = np.array([
            c * world[0] + s * world[1],
            -s * world[0] + c * world[1],
            world[2],
        ], dtype=np.float64)
        dyaw = math.radians(_wrapped_yaw_delta_deg(a, b))
        return np.array([*trans, dyaw], np.float32)
    Ra,Rb=_rotation(a[3],a[4],a[5]),_rotation(b[3],b[4],b[5])
    trans=Ra.T@(pb-pa); roll,yaw,pitch=_rpy(Ra.T@Rb)
    if convention=="yaw4d": return np.array([*trans,yaw],np.float32)
    return np.array([yaw,pitch,roll,*trans],np.float32)


def _wrapped_yaw_delta_deg(a, b):
    """Adjacent raw-yaw delta in degrees, wrapped to [-180, 180)."""
    return (float(b[4]) - float(a[4]) + 180.0) % 360.0 - 180.0


def _wrap_deg(x):
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _so3_angle_rad(R):
    """Geodesic angle of a rotation matrix, in radians."""
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cos_theta)


def _matrix_to_quaternion(R):
    """Convert a rotation matrix to a normalized [w, x, y, z] quaternion."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(R)))
        if axis == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif axis == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                          0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _quaternion_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _so3_slerp(R0, R1, fraction):
    """Interpolate two rotation matrices along the shortest SO(3) geodesic."""
    q0 = _matrix_to_quaternion(R0)
    q1 = _matrix_to_quaternion(R1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    t = float(fraction)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        q = (math.sin((1.0 - t) * theta) / sin_theta) * q0
        q += (math.sin(t * theta) / sin_theta) * q1
    return _quaternion_to_matrix(q)


def _repair_rotation_interpolation_outliers(raw, threshold_deg):
    """Repair UAV-Flow rotation interpolation glitches on SO(3).

    UAV-Flow occasionally interpolates Euler yaw across the +/-180-degree cut,
    producing one or several physically wrong intermediate orientations.  A
    scalar unwrap cannot repair those values: it only changes an angle's
    equivalent representation.  For each interior pose, compare its rotation
    with the SO(3) midpoint of its neighbours.  Consecutive outliers are then
    replaced by shortest-path SO(3) interpolation between the surrounding good
    poses.  This catches both ``-170 -> 19 -> 177`` and same-sign double jumps
    such as ``177 -> 2 -> -175``.

    Translation is left untouched.  Roll/yaw/pitch are replaced together so the
    repaired orientation remains a valid, internally consistent rotation.
    """
    if raw is None or len(raw) < 3:
        return raw, 0
    threshold = float(threshold_deg)
    if threshold <= 0:
        return raw, 0
    rotations = np.stack([_rotation(row[3], row[4], row[5]) for row in raw])
    bad = np.zeros(len(rotations), dtype=bool)
    for i in range(1, len(rotations) - 1):
        expected = _so3_slerp(rotations[i - 1], rotations[i + 1], 0.5)
        midpoint_error_deg = math.degrees(_so3_angle_rad(expected.T @ rotations[i]))
        if midpoint_error_deg > threshold:
            bad[i] = True
    if not bad.any():
        return raw, 0

    repaired_rotations = rotations.copy()
    bad_idx = np.flatnonzero(bad)
    run_start = 0
    while run_start < len(bad_idx):
        run_end = run_start
        while run_end + 1 < len(bad_idx) and bad_idx[run_end + 1] == bad_idx[run_end] + 1:
            run_end += 1
        first_bad = int(bad_idx[run_start])
        last_bad = int(bad_idx[run_end])
        left = first_bad - 1
        right = last_bad + 1
        span = right - left
        for idx in range(first_bad, last_bad + 1):
            repaired_rotations[idx] = _so3_slerp(
                rotations[left], rotations[right], (idx - left) / span
            )
        run_start = run_end + 1

    repaired = [list(row) for row in raw]
    for idx in bad_idx:
        roll, yaw, pitch = _rpy(repaired_rotations[int(idx)])
        repaired[int(idx)][3] = math.degrees(roll)
        repaired[int(idx)][4] = float(_wrap_deg(math.degrees(yaw)))
        repaired[int(idx)][5] = math.degrees(pitch)
    return repaired, int(bad.sum())


class UAVFlowParquetDataset(Dataset):
    """Reads windows without retaining JPEG payloads in the parent process."""
    def __init__(self, parquet_root, image_size=(224,224), future_steps=3, chunk_size=4,
                 visual_anchor_stride=None,
                 action_convention="se3_6d", is_eval=False, eval_ratio=.05,
                 dataset_name="uavflow", max_trajectories=None, source_fps=1.0,
                 expected_shards=None, action_stats_key=None, translation_scale=1.0,
                 openvla_uav_compatible=False, terminal_zero_action=False,
                 endpoint_repeat_count=None, endpoint_self_pair_count=0,
                 endpoint_self_pair_start_count=None, endpoint_self_pair_end_count=None,
                 endpoint_terminal_window_repeat_count=0,
                 stop_soft_target_fraction=0.0,
                 stop_soft_target_floor=0.05,
                 endpoint_absorbing_window_count=0,
                 endpoint_absorbing_max_starts=None,
                 endpoint_absorbing_train_fraction=0.0,
                 stop_absorbing_positive=False,
                 stop_target_mode="chunk_end",
                 episode_start_repeat_count=0,
                 episode_start_train_fraction=0.0,
                 yaw_filter_deg=None, yaw_filter_pad=0,
                 yaw_repair_enabled=False, yaw_repair_deg=None,
                 action_delta_mode="auto", include_episodes=None,
                 episode_split_mode="sorted", episode_split_seed=42,
                 gt_depth_root=None, gt_depth_required=False,
                 gt_depth_fallback_roots=None, instruction_overrides_path=None,
                 gt_depth_min_meters=1e-3, gt_depth_max_meters=None):
        try: import pyarrow.parquet as pq
        except ImportError as exc: raise RuntimeError("UAV-Flow requires pyarrow.") from exc
        self.pq=pq; self.root=Path(parquet_root); self.image_size=tuple(image_size)
        self._depth_cache=OrderedDict()
        self.gt_depth_root=Path(gt_depth_root).expanduser() if gt_depth_root else None
        fallback_roots = gt_depth_fallback_roots or []
        if isinstance(fallback_roots, (str, Path)):
            fallback_roots = [fallback_roots]
        self.gt_depth_roots = ([self.gt_depth_root] if self.gt_depth_root else []) + [
            Path(root).expanduser() for root in fallback_roots if root
        ]
        self._hybrid_depth_paths = {}
        for root in self.gt_depth_roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*/*.npz")):
                episode_id = path.stem
                # Earlier roots have higher priority.  This lets a curated
                # hybrid sidecar override the ordinary replay depth.
                self._hybrid_depth_paths.setdefault(episode_id, path)
        if instruction_overrides_path is None and self.gt_depth_root is not None:
            candidate = self.gt_depth_root / "instruction_overrides.json"
            instruction_overrides_path = candidate if candidate.exists() else None
        self.instruction_overrides = {}
        if instruction_overrides_path:
            override_path = Path(instruction_overrides_path).expanduser()
            with override_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError(f"Instruction overrides must be a JSON object: {override_path}")
            self.instruction_overrides = {str(key): str(value) for key, value in payload.items()}
        self.gt_depth_required=bool(gt_depth_required)
        self.gt_depth_min_meters=float(gt_depth_min_meters)
        self.gt_depth_max_meters=(
            None if gt_depth_max_meters is None else float(gt_depth_max_meters)
        )
        if self.gt_depth_required and self.gt_depth_root is None:
            raise ValueError("gt_depth_required=true requires dataset.gt_depth_root.")
        self.future_steps=int(future_steps); self.chunk_size=int(chunk_size)
        # Historically UAV-Flow followed released GAM exactly: one ``chunk_size``
        # controlled both the number of policy actions emitted per anchor and
        # the distance between visual anchors.  Stage-2 long-horizon ablations
        # need to vary those independently, e.g. adjacent observed frames with
        # a five-action plan and an F(t+5) feature target.  Keep the legacy
        # coupled behavior unless an explicit visual stride is provided.
        self.visual_anchor_stride = (
            self.chunk_size
            if visual_anchor_stride is None
            else int(visual_anchor_stride)
        )
        if self.chunk_size <= 0 or self.visual_anchor_stride <= 0:
            raise ValueError(
                "chunk_size and visual_anchor_stride must be positive, got "
                f"{self.chunk_size} and {self.visual_anchor_stride}."
            )
        aliases={"yaw_4d":"yaw4d","translation_yaw":"yaw4d","xyz_yaw":"yaw4d",
                 "se3":"se3_6d","full_se3":"se3_6d"}
        self.action_convention=aliases.get(str(action_convention).lower(),str(action_convention).lower())
        if self.action_convention not in {"yaw4d","se3_6d"}: raise ValueError(self.action_convention)
        self.action_dim=4 if self.action_convention=="yaw4d" else 6
        self.dataset_name=dataset_name
        self.action_stats_key=str(action_stats_key or dataset_name)
        self.translation_scale=float(translation_scale)
        self.fps=float(source_fps)
        self.openvla_uav_compatible=bool(openvla_uav_compatible)
        delta_mode_aliases = {
            "openvla": "openvla_yaw_only",
            "yaw_only": "openvla_yaw_only",
            "full_se3": "full_se3_legacy",
            "legacy": "full_se3_legacy",
        }
        requested_delta_mode = str(action_delta_mode or "auto").lower()
        requested_delta_mode = delta_mode_aliases.get(requested_delta_mode, requested_delta_mode)
        if requested_delta_mode == "auto":
            requested_delta_mode = (
                "openvla_yaw_only"
                if self.action_convention == "yaw4d" and self.openvla_uav_compatible
                else "full_se3_legacy"
            )
        if requested_delta_mode not in {"openvla_yaw_only", "full_se3_legacy"}:
            raise ValueError(f"Unknown action_delta_mode={action_delta_mode!r}")
        if requested_delta_mode == "openvla_yaw_only" and self.action_convention != "yaw4d":
            raise ValueError("action_delta_mode='openvla_yaw_only' requires action_convention='yaw4d'.")
        self.action_delta_mode = requested_delta_mode
        self.terminal_zero_action=bool(terminal_zero_action) or self.openvla_uav_compatible
        self.endpoint_repeat_count=(
            5 if (endpoint_repeat_count is None and self.openvla_uav_compatible)
            else int(endpoint_repeat_count or 0)
        )
        self.endpoint_self_pair_count=max(0, int(endpoint_self_pair_count))
        self.endpoint_self_pair_start_count = max(
            0,
            int(
                self.endpoint_self_pair_count
                if endpoint_self_pair_start_count is None
                else endpoint_self_pair_start_count
            ),
        )
        self.endpoint_self_pair_end_count = max(
            0,
            int(
                self.endpoint_self_pair_count
                if endpoint_self_pair_end_count is None
                else endpoint_self_pair_end_count
            ),
        )
        # Repeat only the final ordinary full window. With chunk=future=5 and
        # visual stride 1 this is [FN-5,...,FN], supervised by the five real
        # actions that arrive at FN. No synthetic terminal frames are added.
        self.endpoint_terminal_window_repeat_count = max(
            0, int(endpoint_terminal_window_repeat_count)
        )
        # Optional soft approach-to-stop targets.  The exact K-action anchor
        # FN-K remains 1.0; anchors immediately before it receive targets that
        # decay linearly toward ``floor`` over the final trajectory fraction.
        # A zero fraction preserves the original exact binary target.
        self.stop_soft_target_fraction = float(stop_soft_target_fraction)
        self.stop_soft_target_floor = float(stop_soft_target_floor)
        if not 0.0 <= self.stop_soft_target_fraction < 1.0:
            raise ValueError(
                "stop_soft_target_fraction must be in [0,1), got "
                f"{self.stop_soft_target_fraction}"
            )
        if not 0.0 <= self.stop_soft_target_floor <= 1.0:
            raise ValueError(
                "stop_soft_target_floor must be in [0,1], got "
                f"{self.stop_soft_target_floor}"
            )
        # Extend each episode with repeated copies of its final frame and then
        # take the ordinary sliding windows over that extension.  For a
        # four-frame IDM this adds, once per count:
        #   [FN-2,FN-1,FN,FN], [FN-1,FN,FN,FN], [FN,FN,FN,FN].
        # These are training/evaluation samples only.  They intentionally do
        # not enter stat_samples, so changing this option cannot move the
        # action normalization statistics.
        self.endpoint_absorbing_window_count = max(0, int(endpoint_absorbing_window_count))
        self.endpoint_absorbing_max_starts = (
            None
            if endpoint_absorbing_max_starts is None
            else max(0, int(endpoint_absorbing_max_starts))
        )
        self.endpoint_absorbing_train_fraction = float(endpoint_absorbing_train_fraction)
        self.stop_absorbing_positive = bool(stop_absorbing_positive)
        self.stop_target_mode = str(stop_target_mode)
        if self.stop_target_mode not in {"chunk_end", "current"}:
            raise ValueError(f"Unknown stop_target_mode: {self.stop_target_mode}")
        if not 0.0 <= self.endpoint_absorbing_train_fraction < 1.0:
            raise ValueError(
                "endpoint_absorbing_train_fraction must be in [0,1), got "
                f"{self.endpoint_absorbing_train_fraction}"
            )
        self.episode_start_repeat_count = max(0, int(episode_start_repeat_count))
        self.episode_start_train_fraction = float(episode_start_train_fraction)
        if not 0.0 <= self.episode_start_train_fraction < 1.0:
            raise ValueError(
                "episode_start_train_fraction must be in [0,1), got "
                f"{self.episode_start_train_fraction}"
            )
        self.yaw_filter_deg = None if yaw_filter_deg is None or float(yaw_filter_deg) <= 0 else float(yaw_filter_deg)
        self.yaw_filter_pad = max(0, int(yaw_filter_pad))
        self.yaw_repair_enabled = bool(yaw_repair_enabled)
        repair_default = self.yaw_filter_deg if self.yaw_filter_deg is not None else 45.0
        self.yaw_repair_deg = repair_default if yaw_repair_deg is None else float(yaw_repair_deg)
        if self.openvla_uav_compatible:
            # OpenVLA-UAV's simple dataset trains [local dx,dy,dz,dyaw].
            # Keep this as a hard fail instead of silently changing dimensions:
            # the action head is built before samples are read.
            if self.action_convention != "yaw4d":
                raise ValueError(
                    "openvla_uav_compatible=true requires dataset.action_convention='yaw4d' "
                    f"(got {self.action_convention!r})."
                )
        self._files=sorted(self.root.glob("*.parquet")); self._cache=OrderedDict()
        if not self._files: raise FileNotFoundError(f"No parquet files under {self.root}")
        if expected_shards is not None and len(self._files) != int(expected_shards):
            present = {int(p.name.split("-")[1]) for p in self._files if p.name.startswith("train-")}
            missing = sorted(set(range(int(expected_shards))) - present)
            raise RuntimeError(
                f"Incomplete {dataset_name} download under {self.root}: "
                f"found {len(self._files)}/{int(expected_shards)} parquet shards; "
                f"missing indices={missing}."
            )
        trajectories={}; logs={}; openvla_prompt_poses={}
        for fi,path in enumerate(self._files):
            pf=pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                table=pf.read_row_group(rg,columns=["id","frame_idx","log"])
                for ri,row in enumerate(table.to_pylist()):
                    tid=str(row["id"]); trajectories.setdefault(tid,[]).append((int(row["frame_idx"]),fi,rg,ri))
                    if tid not in logs:
                        meta=json.loads(row["log"])
                        text=meta.get("instruction_unified") or meta.get("instruction") or ""
                        logs[tid]=(meta.get("raw_logs",[]),self.instruction_overrides.get(tid,text))
                        preprocessed=np.asarray(meta.get("preprocessed_logs",[]),dtype=np.float32)
                        if preprocessed.ndim == 2 and preprocessed.shape[1] >= 5:
                            # Exact OpenVLA-UAV Current State: first-frame
                            # camera coordinates in centimetres, yaw in degrees.
                            openvla_prompt_poses[tid]=preprocessed[:,[0,1,2,4]]
        ids=sorted(trajectories)
        self.episode_split_mode = str(episode_split_mode).strip().lower()
        self.episode_split_seed = int(episode_split_seed)
        self.split_category_counts = None
        if include_episodes is not None:
            requested = {str(x) for x in include_episodes}
            missing = sorted(requested.difference(trajectories))
            if missing:
                raise KeyError(
                    f"{dataset_name} explicit episode split has {len(missing)} missing IDs; "
                    f"first={missing[:5]}"
                )
            ids=[tid for tid in ids if tid in requested]
        else:
            if max_trajectories: ids=ids[:int(max_trajectories)]
            if self.episode_split_mode == "stratified_instruction" and len(ids) > 1:
                train_ids, eval_ids, self.split_category_counts = _stratified_episode_split(
                    ids, logs, eval_ratio, self.episode_split_seed
                )
                ids = eval_ids if is_eval else train_ids
            elif self.episode_split_mode == "sorted":
                split=max(1,int(len(ids)*(1-float(eval_ratio)))) if len(ids)>1 else len(ids)
                ids=ids[split:] if is_eval else ids[:split]
            else:
                raise ValueError(
                    f"Unsupported episode_split_mode={self.episode_split_mode!r}; "
                    "expected 'sorted' or 'stratified_instruction'."
                )
        self.traj={tid:sorted(trajectories[tid]) for tid in ids}
        repaired_yaw_points = 0
        self.logs={}
        self.openvla_prompt_poses={}
        for tid in ids:
            raw, text = logs[tid]
            if self.yaw_repair_enabled and self.action_convention == "yaw4d":
                raw, repaired_count = _repair_rotation_interpolation_outliers(raw, self.yaw_repair_deg)
                repaired_yaw_points += repaired_count
            self.logs[tid] = (raw, text)
            if tid in openvla_prompt_poses:
                self.openvla_prompt_poses[tid]=openvla_prompt_poses[tid]
        self.samples=[]; self.stat_samples=[]
        last_visual_offset = self.future_steps * self.visual_anchor_stride
        last_action_offset = last_visual_offset + self.chunk_size - 1
        filtered_windows = 0
        endpoint_repeated_windows = 0
        endpoint_self_pair_windows = 0
        endpoint_terminal_window_repeats = 0
        endpoint_absorbing_windows = 0
        endpoint_absorbing_balanced_windows = 0
        endpoint_absorbing_downsampled_windows = 0
        episode_start_repeated_windows = 0
        episode_start_balanced_windows = 0
        absorbing_samples_by_type = {}
        episode_start_samples = []
        for tid in ids:
            n=min(len(self.traj[tid]),len(self.logs[tid][0]))
            # Legacy GAM uses visual_anchor_stride == chunk_size.  The explicit
            # decoupled mode instead supports adjacent visual anchors with an
            # overlapping K-action target starting at every anchor.
            if self.terminal_zero_action:
                # OpenVLA-UAV defines the final frame's action as zero.  For
                # chunked GAM windows, visual anchors only need to exist; action
                # slots that would step beyond the final image are zero-padded.
                max_start = n - 1 - last_visual_offset
                base_samples = [(tid, s) for s in range(max(0, max_start + 1))]
            else:
                # Strict mode requires the last action in the last chunk to
                # have a physical next pose.
                max_start = n - 2 - last_action_offset
                base_samples = [(tid, s) for s in range(max(0, max_start + 1))]
            _sample_is_clean = None
            if self.yaw_filter_deg is not None and self.action_convention == "yaw4d":
                raw = self.logs[tid][0]
                bad = set()
                for idx in range(max(0, n - 1)):
                    if abs(_wrapped_yaw_delta_deg(raw[idx], raw[idx + 1])) > self.yaw_filter_deg:
                        lo = max(0, idx - self.yaw_filter_pad)
                        hi = min(n - 2, idx + self.yaw_filter_pad)
                        bad.update(range(lo, hi + 1))

                def _sample_is_clean(sample_start):
                    # A window supervises actions at every anchor plus every
                    # chunk slot.  Drop it if any supervised transition is bad.
                    for t in range(self.future_steps + 1):
                        anchor = sample_start + t * self.visual_anchor_stride
                        for j in range(self.chunk_size):
                            action_idx = anchor + j
                            if action_idx >= n - 1:
                                # terminal_zero_action pads this slot; it is
                                # not a raw yaw transition and should not count
                                # as bad.
                                continue
                            if action_idx in bad:
                                return False
                    return True

                if base_samples:
                    before = len(base_samples)
                    base_samples = [(tid, s) for _, s in base_samples if _sample_is_clean(s)]
                    filtered_windows += before - len(base_samples)
            self.stat_samples.extend(base_samples)
            self.samples.extend(base_samples)
            if base_samples and int(base_samples[0][1]) == 0:
                episode_start_samples.append(base_samples[0])
            if self.endpoint_absorbing_window_count > 0 and self.terminal_zero_action and n > 0:
                # max_start is the last window whose visual anchors are all
                # physical frames.  Starts after it are the windows obtained
                # by appending repeated FN frames.  __getitem__ already clamps
                # their visual indices and _scaled_delta_at emits terminal
                # zero actions, so retain the normal sample representation.
                first_absorbing_start = max(0, max_start + 1)
                absorbing_samples = [(tid, s) for s in range(first_absorbing_start, n)]
                if self.endpoint_absorbing_max_starts is not None:
                    # Keep the starts immediately following the final fully
                    # physical window.  These are exactly the windows created
                    # by appending K terminal frames.  Taking the final K
                    # starts would instead collapse most/all history anchors
                    # to FN when future_steps > 1 (the M1 ablation).
                    absorbing_samples = (
                        absorbing_samples[:self.endpoint_absorbing_max_starts]
                        if self.endpoint_absorbing_max_starts > 0 else []
                    )
                if _sample_is_clean is not None:
                    absorbing_samples = [
                        sample for sample in absorbing_samples if _sample_is_clean(sample[1])
                    ]
                for _ in range(self.endpoint_absorbing_window_count):
                    self.samples.extend(absorbing_samples)
                    endpoint_absorbing_windows += len(absorbing_samples)
                for sample in absorbing_samples:
                    absorbing_type = int(sample[1]) - int(first_absorbing_start)
                    absorbing_samples_by_type.setdefault(absorbing_type, []).append(sample)
            if (not is_eval) and self.endpoint_repeat_count > 0 and base_samples:
                # OpenVLA-UAV repeats the first and last samples.  Do not add
                # these duplicates to stat_samples: its norm stats are computed
                # from episodes before sample repetition.
                first = base_samples[0]
                last = base_samples[-1]
                for _ in range(self.endpoint_repeat_count):
                    self.samples.append(first)
                    self.samples.append(last)
                    endpoint_repeated_windows += 2
            if (not is_eval) and self.episode_start_repeat_count > 0 and base_samples:
                # Start-only counterpart to OpenVLA-UAV endpoint repetition.
                # It repeats the genuine first control sample (never a zero
                # self-pair) while leaving terminal weighting to absorbing
                # windows when those are enabled.
                first = base_samples[0]
                for _ in range(self.episode_start_repeat_count):
                    self.samples.append(first)
                    episode_start_repeated_windows += 1
            if (
                (not is_eval)
                and (self.endpoint_self_pair_start_count > 0 or self.endpoint_self_pair_end_count > 0)
                and n > 0
            ):
                # OpenVLA-style stationary endpoint supervision for inverse
                # dynamics: duplicate the first/last image as a two-frame pair
                # and supervise zero action.  These synthetic samples are not
                # added to stat_samples; the trajectory-level stats already
                # include one terminal zero action when terminal_zero_action is
                # enabled.
                first_self = (tid, 0, "self_pair")
                last_self = (tid, n - 1, "self_pair")
                for _ in range(self.endpoint_self_pair_start_count):
                    self.samples.append(first_self)
                    endpoint_self_pair_windows += 1
                for _ in range(self.endpoint_self_pair_end_count):
                    self.samples.append(last_self)
                    endpoint_self_pair_windows += 1
            if (
                (not is_eval)
                and self.endpoint_terminal_window_repeat_count > 0
                and base_samples
            ):
                # base_samples[-1] is already present once. Add exactly the
                # requested number of extra copies without touching F0.
                terminal_window = base_samples[-1]
                for _ in range(self.endpoint_terminal_window_repeat_count):
                    self.samples.append(terminal_window)
                    endpoint_terminal_window_repeats += 1
        if (
            (not is_eval)
            and self.endpoint_absorbing_train_fraction > 0.0
            and absorbing_samples_by_type
        ):
            # Make all terminal-window types equally likely, with their union
            # occupying the requested fraction of the training samples.  This
            # is a true target rather than a lower bound: short-horizon runs
            # are upsampled and long-horizon runs whose natural terminal share
            # is already too large are deterministically downsampled.
            normal_windows = len(self.samples) - endpoint_absorbing_windows
            desired_special = int(round(
                normal_windows
                * self.endpoint_absorbing_train_fraction
                / (1.0 - self.endpoint_absorbing_train_fraction)
            ))
            type_ids = sorted(absorbing_samples_by_type)
            base_target, remainder = divmod(desired_special, len(type_ids))
            absorbing_set = {
                sample
                for pool in absorbing_samples_by_type.values()
                for sample in pool
            }
            self.samples = [
                sample for sample in self.samples if sample not in absorbing_set
            ]
            balanced_samples = []
            for position, absorbing_type in enumerate(type_ids):
                pool = absorbing_samples_by_type[absorbing_type]
                desired_for_type = base_target + int(position < remainder)
                current_for_type = len(pool) * self.endpoint_absorbing_window_count
                if desired_for_type <= len(pool):
                    if desired_for_type > 0:
                        keep = np.linspace(
                            0, len(pool) - 1, desired_for_type, dtype=int
                        )
                        balanced_samples.extend(pool[int(index)] for index in keep)
                else:
                    repeats, tail = divmod(desired_for_type, len(pool))
                    balanced_samples.extend(pool * repeats)
                    balanced_samples.extend(pool[:tail])
                endpoint_absorbing_balanced_windows += max(
                    0, desired_for_type - current_for_type
                )
                endpoint_absorbing_downsampled_windows += max(
                    0, current_for_type - desired_for_type
                )
            self.samples.extend(balanced_samples)
            endpoint_absorbing_windows = len(balanced_samples)
        if (not is_eval) and self.episode_start_train_fraction > 0.0 and episode_start_samples:
            # Solve (current_start + extra)/(total + extra) = requested.
            # These are genuine start=0 windows with their real F0->F1 target,
            # never zero-action self-pairs.
            start_set = set(episode_start_samples)
            current_start = sum(1 for sample in self.samples if sample in start_set)
            requested = self.episode_start_train_fraction
            extra = max(0, int(np.ceil(
                (requested * len(self.samples) - current_start) / (1.0 - requested)
            )))
            repeats, tail = divmod(extra, len(episode_start_samples))
            if repeats:
                self.samples.extend(episode_start_samples * repeats)
            if tail:
                self.samples.extend(episode_start_samples[:tail])
            episode_start_balanced_windows = extra
        total_windows = len(self.samples)
        achieved_absorbing_fraction = (
            float(endpoint_absorbing_windows) / float(total_windows)
            if total_windows > 0 else 0.0
        )
        mode="eval" if is_eval else "train"
        print(
            f"UAVFlowParquetDataset[{mode}] {self.dataset_name}: "
            f"trajectories={len(ids)} samples={len(self.samples)} "
            f"stat_samples={len(self.stat_samples)} eval_ratio={float(eval_ratio):.3f} "
            f"chunk={self.chunk_size} visual_anchor_stride={self.visual_anchor_stride} "
            f"future_steps={self.future_steps} "
            f"openvla_uav_compatible={self.openvla_uav_compatible} "
            f"action_delta_mode={self.action_delta_mode} "
            f"terminal_zero_action={self.terminal_zero_action} "
            f"yaw_filter_deg={self.yaw_filter_deg} yaw_filter_pad={self.yaw_filter_pad} "
            f"yaw_repair_enabled={self.yaw_repair_enabled} yaw_repair_deg={self.yaw_repair_deg} "
            f"repaired_yaw_points={repaired_yaw_points} "
            f"filtered_windows={filtered_windows} "
            f"endpoint_repeat_count={self.endpoint_repeat_count if not is_eval else 0} "
            f"endpoint_repeated_windows={endpoint_repeated_windows} "
            f"endpoint_self_pair_count={self.endpoint_self_pair_count if not is_eval else 0} "
            f"endpoint_self_pair_start_count={self.endpoint_self_pair_start_count if not is_eval else 0} "
            f"endpoint_self_pair_end_count={self.endpoint_self_pair_end_count if not is_eval else 0} "
            f"endpoint_self_pair_windows={endpoint_self_pair_windows} "
            f"endpoint_terminal_window_repeat_count={self.endpoint_terminal_window_repeat_count if not is_eval else 0} "
            f"endpoint_terminal_window_repeats={endpoint_terminal_window_repeats} "
            f"stop_soft_target_fraction={self.stop_soft_target_fraction} "
            f"stop_soft_target_floor={self.stop_soft_target_floor} "
            f"endpoint_absorbing_window_count={self.endpoint_absorbing_window_count} "
            f"endpoint_absorbing_max_starts={self.endpoint_absorbing_max_starts} "
            f"endpoint_absorbing_train_fraction={self.endpoint_absorbing_train_fraction if not is_eval else 0.0} "
            f"stop_absorbing_positive={self.stop_absorbing_positive} "
            f"endpoint_absorbing_windows={endpoint_absorbing_windows} "
            f"endpoint_absorbing_balanced_windows={endpoint_absorbing_balanced_windows} "
            f"endpoint_absorbing_downsampled_windows={endpoint_absorbing_downsampled_windows} "
            f"endpoint_absorbing_achieved_fraction={achieved_absorbing_fraction:.4f} "
            f"episode_start_repeat_count={self.episode_start_repeat_count if not is_eval else 0} "
            f"episode_start_repeated_windows={episode_start_repeated_windows} "
            f"episode_start_train_fraction={self.episode_start_train_fraction if not is_eval else 0.0} "
            f"episode_start_balanced_windows={episode_start_balanced_windows} "
            f"endpoint_augmented_windows={endpoint_repeated_windows + endpoint_self_pair_windows + endpoint_terminal_window_repeats + endpoint_absorbing_windows + episode_start_repeated_windows}",
            flush=True,
        )
        if self.split_category_counts is not None:
            split_key = "eval" if is_eval else "train"
            print(
                f"UAVFlow episode split [{split_key}] mode={self.episode_split_mode} "
                f"seed={self.episode_split_seed} counts="
                + ",".join(
                    f"{name}:{values[split_key]}"
                    for name, values in self.split_category_counts.items()
                ),
                flush=True,
            )

    def __len__(self): return len(self.samples)

    def _row(self, fi,rg,ri):
        key=(fi,rg)
        if key not in self._cache:
            self._cache[key]=self.pq.ParquetFile(self._files[fi]).read_row_group(rg,columns=["image"]).to_pylist()
            while len(self._cache)>4: self._cache.popitem(last=False)
        return self._cache[key][ri]

    def _image(self, locator):
        _,fi,rg,ri=locator; obj=self._row(fi,rg,ri)["image"]
        raw=obj["bytes"] if isinstance(obj,dict) else obj
        im=Image.open(io.BytesIO(raw)).convert("RGB").resize((self.image_size[1],self.image_size[0]),Image.BICUBIC)
        return torch.from_numpy(np.asarray(im,dtype=np.uint8).copy()).permute(2,0,1).float()/255.

    def _depth_source(self, episode_id, frame_idx):
        episode_id = str(episode_id)
        hybrid_path = self._hybrid_depth_paths.get(episode_id)
        if hybrid_path is not None:
            return "hybrid", hybrid_path
        first_missing = None
        for root in self.gt_depth_roots:
            episode = root / episode_id
            episode_array = episode / "depth.npy"
            candidates = (
                episode_array,
                episode / f"{int(frame_idx):06d}.npy",
                episode / f"frame_{int(frame_idx):06d}.npy",
                root / f"{episode_id}_{int(frame_idx):06d}.npy",
            )
            found = next((path for path in candidates if path.exists()), None)
            if found is not None:
                return "replay", found
            if first_missing is None:
                first_missing = candidates[0]
        return "missing", first_missing

    def _cached_numpy(self, path, loader):
        key=str(path)
        if key not in self._depth_cache:
            self._depth_cache[key]=loader()
            while len(self._depth_cache)>8:
                self._depth_cache.popitem(last=False)
        else:
            self._depth_cache.move_to_end(key)
        return self._depth_cache[key]

    def _depth(self, episode_id, frame_idx):
        source,path=self._depth_source(episode_id,frame_idx)
        if path is None:
            return None,None,None
        saved_mask = None
        semantic_mask = None
        if source == "hybrid":
            def _load_hybrid():
                with np.load(path) as payload:
                    required = {"hybrid_depth_m", "valid_mask", "frame_indices"}
                    missing = required.difference(payload.files)
                    if missing:
                        raise KeyError(f"Hybrid depth {path} is missing keys {sorted(missing)}")
                    return {
                        "depth": np.asarray(payload["hybrid_depth_m"], dtype=np.float32),
                        "mask": np.asarray(payload["valid_mask"], dtype=np.bool_),
                        "semantic_mask": (
                            np.asarray(payload["semantic_mask"], dtype=np.bool_)
                            if "semantic_mask" in payload.files else None
                        ),
                        "frame_indices": np.asarray(payload["frame_indices"], dtype=np.int64),
                    }
            payload = self._cached_numpy(path, _load_hybrid)
            matches = np.flatnonzero(payload["frame_indices"] == int(frame_idx))
            if matches.size == 0:
                raise IndexError(f"Hybrid depth {path} has no frame_idx={int(frame_idx)}")
            offset = int(matches[0])
            depth=torch.from_numpy(payload["depth"][offset].copy())
            saved_mask=torch.from_numpy(payload["mask"][offset].copy())
            if payload["semantic_mask"] is not None:
                semantic_mask=torch.from_numpy(payload["semantic_mask"][offset].copy())
        elif source == "replay" and path.name == "depth.npy":
            array=self._cached_numpy(path, lambda: np.load(path,mmap_mode="r"))
            if int(frame_idx) >= int(array.shape[0]):
                raise IndexError(
                    f"Depth sidecar {path} has {array.shape[0]} frames, "
                    f"requested {int(frame_idx)}."
                )
            depth=torch.from_numpy(np.asarray(array[int(frame_idx)],dtype=np.float32).copy())
        elif source == "replay" and path.exists():
            depth=torch.from_numpy(np.asarray(np.load(path),dtype=np.float32).squeeze().copy())
        else:
            depth=None
        if not path.exists():
            if self.gt_depth_required:
                raise FileNotFoundError(
                    f"Missing UAV-Flow-Sim GT depth for episode={episode_id} "
                    f"frame={int(frame_idx)}; expected {path}"
                )
            depth=torch.zeros(self.image_size,dtype=torch.float32)
            empty=torch.zeros_like(depth,dtype=torch.bool)
            return depth,empty,empty
        assert depth is not None
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth map in {path}, got {tuple(depth.shape)}")
        if tuple(depth.shape) != self.image_size:
            depth=torch.nn.functional.interpolate(
                depth[None,None],size=self.image_size,mode="nearest"
            )[0,0]
        mask=torch.isfinite(depth) & (depth > self.gt_depth_min_meters)
        if self.gt_depth_max_meters is not None:
            mask=mask & (depth < self.gt_depth_max_meters)
        if saved_mask is not None:
            if tuple(saved_mask.shape) != self.image_size:
                saved_mask=torch.nn.functional.interpolate(
                    saved_mask[None,None].float(),size=self.image_size,mode="nearest"
                )[0,0].bool()
            mask=mask & saved_mask
        if semantic_mask is None:
            semantic_mask=torch.zeros_like(mask,dtype=torch.bool)
        elif tuple(semantic_mask.shape) != self.image_size:
            semantic_mask=torch.nn.functional.interpolate(
                semantic_mask[None,None].float(),size=self.image_size,mode="nearest"
            )[0,0].bool()
        semantic_mask=semantic_mask & mask
        depth=torch.where(mask,depth,torch.zeros_like(depth))
        return depth,mask,semantic_mask

    def _zero_delta(self):
        return np.zeros(self.action_dim, dtype=np.float32)

    def _scaled_delta_at(self, raw, idx):
        if idx >= len(raw) - 1:
            if self.terminal_zero_action:
                return self._zero_delta()
            raise IndexError(f"Action index {idx} requires next pose {idx + 1}, but len(raw)={len(raw)}")
        delta=_delta(
            raw[idx], raw[idx+1], self.action_convention,
            action_delta_mode=self.action_delta_mode,
        )
        if self.action_convention=="yaw4d": delta[:3]*=self.translation_scale
        else: delta[3:]*=self.translation_scale
        return delta

    def _actions(self, raw, start):
        rows=[]
        for t in range(self.future_steps+1):
            anchor=start+t*self.visual_anchor_stride
            chunks=[]
            for j in range(self.chunk_size):
                chunks.append(self._scaled_delta_at(raw, anchor+j))
            rows.append(np.stack(chunks))
        return torch.from_numpy(np.stack(rows)).float()

    def _zero_actions(self):
        return torch.zeros(
            self.future_steps + 1,
            self.chunk_size,
            self.action_dim,
            dtype=torch.float32,
        )

    def _past_actions(self, raw, start, actions):
        past=torch.zeros_like(actions)
        mask=torch.zeros_like(actions,dtype=torch.bool)
        for t in range(self.future_steps+1):
            # The previous-action token contains the K immediately preceding
            # executed actions, independent of the visual anchor stride.
            anchor=start+t*self.visual_anchor_stride-self.chunk_size
            if anchor < 0:
                continue
            chunks=[]
            for j in range(self.chunk_size):
                chunks.append(self._scaled_delta_at(raw, anchor+j))
            past[t]=torch.from_numpy(np.stack(chunks)).to(dtype=actions.dtype)
            mask[t]=True
        return past,mask

    def _episode_pose_at(self, raw, idx):
        """Pose relative to the physical episode start for Stage-2 control.

        Translation follows the same yaw-only local convention and metre
        conversion as the 4-DoF policy action.  Relative yaw is represented by
        sin/cos so the numeric state token has no +/-pi discontinuity.
        """
        delta = _delta(
            raw[0], raw[idx], "yaw4d",
            action_delta_mode="openvla_yaw_only",
        )
        delta[:3] *= self.translation_scale
        yaw = float(delta[3])
        return np.asarray(
            [delta[0], delta[1], delta[2], np.sin(yaw), np.cos(yaw)],
            dtype=np.float32,
        )

    def __getitem__(self,index):
        sample=self.samples[index]
        if len(sample) == 3:
            tid,start,mode=sample
        else:
            tid,start=sample
            mode="normal"
        loc=self.traj[tid]; raw,text=self.logs[tid]
        T=self.future_steps+1
        if mode == "self_pair":
            anchor_indices=[int(start) for _ in range(T)]
        else:
            # Clamp terminal anchors only when terminal_zero_action is enabled:
            # actions beyond len(raw)-1 become zero, and images repeat the last
            # available frame.
            last_idx=len(loc)-1
            anchor_indices=[
                min(start+t*self.visual_anchor_stride, last_idx)
                if self.terminal_zero_action
                else start+t*self.visual_anchor_stride
                for t in range(T)
            ]
        images=torch.stack([self._image(loc[idx]) for idx in anchor_indices])[:,None]
        episode_first_image=(images[0] if int(anchor_indices[0]) == 0 else self._image(loc[0])[None])
        actions=self._zero_actions() if mode == "self_pair" else self._actions(raw,start)
        # Relative pose history in [roll,yaw,pitch,x,y,z], radians/metres.
        base=raw[start]; props=[]
        for idx in anchor_indices:
            d6=_delta(base,raw[idx],"se3_6d"); d6[3:]*=self.translation_scale
            props.append([d6[2],d6[0],d6[1],d6[3],d6[4],d6[5]])
        props=torch.tensor(props,dtype=torch.float32)
        episode_pose=torch.from_numpy(np.stack([
            self._episode_pose_at(raw, idx) for idx in anchor_indices
        ])).float()
        if tid in self.openvla_prompt_poses:
            openvla_prompt_pose=torch.from_numpy(
                self.openvla_prompt_poses[tid][anchor_indices].copy()
            ).float()
        else:
            # Compatibility fallback for data without preprocessed_logs.
            prompt_rows=[]
            for idx in anchor_indices:
                delta=_delta(raw[0],raw[idx],"yaw4d",action_delta_mode="full_se3_legacy")
                prompt_rows.append([delta[0],delta[1],delta[2],math.degrees(float(delta[3]))])
            openvla_prompt_pose=torch.tensor(prompt_rows,dtype=torch.float32)
        # Legacy chunk_end: a positive Stop means the K-action chunk beginning at
        # this anchor arrives exactly at FN. For K=5 this labels FN-5, not the
        # synthetic FN-1/FN/FN/... window and not a FN/FN self-pair.
        # current: hard terminal at FN, optional soft ramp BEFORE FN.
        terminal_transition = len(loc) - 1
        exact_stop_anchor = (
            terminal_transition if self.stop_target_mode == "current"
            else terminal_transition - self.chunk_size
        )
        ramp_span = max(
            1, int(math.ceil(terminal_transition * self.stop_soft_target_fraction))
        )

        def _stop_target(index):
            distance = exact_stop_anchor - int(index)
            if distance == 0 or (
                self.stop_absorbing_positive and int(index) >= exact_stop_anchor
            ):
                return 1.0
            if (
                self.stop_soft_target_fraction > 0.0
                and 0 < distance <= ramp_span
            ):
                progress = 1.0 - float(distance) / float(ramp_span)
                return self.stop_soft_target_floor + (
                    1.0 - self.stop_soft_target_floor
                ) * progress
            return 0.0

        stop_target=torch.tensor(
            [_stop_target(idx) for idx in anchor_indices], dtype=torch.float32
        )
        # Match GAM chunk conditioning: each visual anchor receives the action
        # chunk that was executed immediately before it.  For sample starts in
        # the middle of an episode, the first anchor also has a real previous
        # chunk from before this sampled window; only true episode-start slots
        # stay zero and are masked after action normalization.
        past,past_mask=(torch.zeros_like(actions), torch.zeros_like(actions,dtype=torch.bool)) if mode == "self_pair" else self._past_actions(raw,start,actions)
        result={"current_images":images[0],"future_images":images[1:],"all_view_images":images,
                "episode_first_image":episode_first_image,
                "actions":actions,"proprioception":props,"task_description":text,
                "episode_pose":episode_pose,
                "openvla_prompt_pose":openvla_prompt_pose,
                "start_t":torch.tensor(start),"frame_indices":torch.tensor(anchor_indices),
                "camera_keys":["frontcamera"],"view_valid_mask":torch.ones(T,1,dtype=torch.bool),
                "context_valid_mask":torch.ones(T,dtype=torch.bool),"transition_loss_mask":torch.ones(T-1,dtype=torch.bool),
                "action_loss_mask":torch.ones_like(actions,dtype=torch.bool),"past_action_history":past,
                "past_action_history_mask":past_mask,"dataset_name":self.dataset_name,
                "action_stats_key":self.action_stats_key,"episode_id":tid,"has_action":True,
                "is_self_pair": mode == "self_pair", "stop_target":stop_target}
        if self.gt_depth_roots:
            depth_and_masks=[self._depth(tid,idx) for idx in anchor_indices]
            result["gt_depth_meters"]=torch.stack([item[0] for item in depth_and_masks])[:,None]
            result["gt_depth_mask"]=torch.stack([item[1] for item in depth_and_masks])[:,None]
            result["gt_depth_semantic_mask"]=torch.stack(
                [item[2] for item in depth_and_masks]
            )[:,None]
        return result

    def _action_statistics_values(self,max_samples:Optional[int]=None):
        # OpenVLA-UAV computes action stats at the trajectory level, before
        # first/last-frame sample repetition.  Match that: each adjacent-frame
        # action contributes once, and terminal_zero_action contributes exactly
        # one final zero action per trajectory.
        #
        # max_samples remains a deterministic subsample over flattened
        # trajectory-level actions, not over overlapping training windows.
        rows=[]
        for tid in self.traj:
            raw=self.logs[tid][0]
            n=min(len(self.traj[tid]),len(raw))
            if n <= 0:
                continue
            traj_rows=[]
            for idx in range(max(0,n-1)):
                if (
                    self.yaw_filter_deg is not None
                    and self.action_convention == "yaw4d"
                    and abs(_wrapped_yaw_delta_deg(raw[idx], raw[idx + 1])) > self.yaw_filter_deg
                ):
                    continue
                traj_rows.append(self._scaled_delta_at(raw, idx))
            if self.terminal_zero_action:
                traj_rows.append(self._zero_delta())
            if traj_rows:
                rows.append(np.stack(traj_rows))
        if not rows:
            return np.zeros((0,self.action_dim),dtype=np.float32)
        x=np.concatenate(rows,axis=0)
        if max_samples is not None and int(max_samples) > 0 and x.shape[0] > int(max_samples):
            keep=np.linspace(0,x.shape[0]-1,int(max_samples),dtype=int)
            x=x[keep]
        return x

    @staticmethod
    def _summarize(x, dim):
        return {"mean":x.mean(0),"std":x.std(0),"min":x.min(0),"max":x.max(0),
                "q01":np.percentile(x,1,0),"q99":np.percentile(x,99,0),"mask":np.ones(dim,bool)}

    def compute_action_statistics(self,max_samples:Optional[int]=None):
        x=self._action_statistics_values(max_samples)
        return {self.action_stats_key:self._summarize(x,self.action_dim)}

    def _proprio_statistics_values(self,max_samples:Optional[int]=None):
        count = (
            len(self.samples)
            if max_samples is None or int(max_samples) <= 0
            else min(len(self.samples), int(max_samples))
        ); rows=[]
        # Compute the exact window-relative proprio values from raw poses,
        # without loading image payloads.
        T=self.future_steps+1
        stat_samples = self.stat_samples if getattr(self, "stat_samples", None) else self.samples
        count = (
            len(stat_samples)
            if max_samples is None or int(max_samples) <= 0
            else min(len(stat_samples), int(max_samples))
        )
        for i in np.linspace(0,max(len(stat_samples)-1,0),count,dtype=int):
            tid,start=stat_samples[int(i)]; raw=self.logs[tid][0]; base=raw[start]
            props=[]
            for t in range(T):
                d6=_delta(base,raw[start+t*self.visual_anchor_stride],"se3_6d"); d6[3:]*=self.translation_scale
                props.append([d6[2],d6[0],d6[1],d6[3],d6[4],d6[5]])
            rows.append(np.asarray(props,dtype=np.float32))
        return np.concatenate(rows,axis=0)

    def compute_proprio_statistics(self,max_samples:Optional[int]=None):
        x=self._proprio_statistics_values(max_samples)
        return {self.action_stats_key:self._summarize(x,6)}

    def _episode_pose_statistics_values(self, max_samples:Optional[int]=None):
        """Unique physical-frame states; endpoint/window repeats count once."""
        rows=[]
        for tid,locators in self.traj.items():
            raw=self.logs[tid][0]
            n=min(len(locators),len(raw))
            if n > 0:
                rows.append(np.stack([self._episode_pose_at(raw,idx) for idx in range(n)]))
        if not rows:
            return np.zeros((0,5),dtype=np.float32)
        x=np.concatenate(rows,axis=0)
        if max_samples is not None and int(max_samples)>0 and x.shape[0]>int(max_samples):
            keep=np.linspace(0,x.shape[0]-1,int(max_samples),dtype=int)
            x=x[keep]
        return x

    def compute_episode_pose_statistics(self,max_samples:Optional[int]=None):
        x=self._episode_pose_statistics_values(max_samples)
        stats=self._summarize(x,5)
        # xyz is q01/q99 normalized. sin/cos is already bounded in [-1,1].
        stats["mask"]=np.asarray([True,True,True,False,False],dtype=bool)
        return stats


class UAVFlowMixtureDataset(Dataset):
    """Deterministic weighted mixture with independent per-domain statistics."""

    def __init__(self, datasets, weights, virtual_length=None, schedule_size=1000):
        self.datasets = dict(datasets)
        if not self.datasets:
            raise ValueError("UAVFlowMixtureDataset requires at least one source.")
        if set(self.datasets) != set(weights):
            raise ValueError("Mixture weights must match dataset source names.")
        if any(len(ds) <= 0 for ds in self.datasets.values()):
            raise ValueError("Every UAV mixture source must contain at least one sample.")
        total = float(sum(float(weights[k]) for k in self.datasets))
        if total <= 0:
            raise ValueError("Mixture weights must sum to a positive value.")
        normalized = {k: float(weights[k]) / total for k in self.datasets}
        self.weights = normalized
        slots = max(int(schedule_size), len(self.datasets))
        counts = {k: max(1, int(round(normalized[k] * slots))) for k in self.datasets}
        self.schedule = []
        # Evenly interleave a fixed number of tickets rather than grouping all
        # real then all sim samples. DistributedSampler may shuffle on top.
        remaining = dict(counts)
        keys = list(self.datasets)
        while any(remaining.values()):
            for key in keys:
                if remaining[key] > 0:
                    self.schedule.append(key)
                    remaining[key] -= 1
        self.occurrence = []
        seen = {k: 0 for k in keys}
        for key in self.schedule:
            self.occurrence.append(seen[key])
            seen[key] += 1
        self.counts = seen
        self.physical_length = sum(len(ds) for ds in self.datasets.values())
        self.virtual_length = int(virtual_length or self.physical_length)
        # Number of tickets each source contributes during one complete
        # virtual epoch.  This is deliberately different from ``counts``
        # above: counts is only the source allocation inside one short
        # schedule cycle (for example 600/400), whereas tickets_per_epoch is
        # the actual number consumed before DistributedSampler advances its
        # epoch.  Advancing by only one schedule cycle made consecutive epochs
        # revisit almost the same physical windows.
        full_cycles, remainder = divmod(self.virtual_length, len(self.schedule))
        self.tickets_per_epoch = {
            key: full_cycles * self.counts[key]
            + sum(1 for source in self.schedule[:remainder] if source == key)
            for key in keys
        }
        # Persistent DataLoader workers keep Dataset copies, so share the
        # epoch counter.  This rotates the finite per-source ticket mapping
        # while preserving the configured real/sim ratio and epoch length.
        self._shared_epoch = mp.Value("q", 0, lock=False)

    def set_epoch(self, epoch):
        """Rotate which physical source samples are omitted/repeated."""
        self._shared_epoch.value = int(epoch)

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, index):
        index = int(index)
        slot = index % len(self.schedule)
        cycle = index // len(self.schedule)
        key = self.schedule[slot]
        local_index = cycle * self.counts[key] + self.occurrence[slot]
        # DistributedSampler.set_epoch only shuffles virtual tickets.  This
        # source-specific offset additionally rotates their physical mapping.
        local_index += int(self._shared_epoch.value) * self.tickets_per_epoch[key]
        return self.datasets[key][local_index % len(self.datasets[key])]

    def compute_action_statistics(self, max_samples=None):
        keys={ds.action_stats_key for ds in self.datasets.values()}
        if len(keys)!=1:
            raise ValueError(f"Shared UAV mixture normalization requires one stats key, got {keys}")
        arrays=[]
        for name,ds in self.datasets.items():
            n=max_samples
            if max_samples is not None and int(max_samples)>0:
                n=max(1,int(round(int(max_samples)*self.weights[name])))
            arrays.append(ds._action_statistics_values(n))
        x=np.concatenate(arrays,axis=0)
        return {next(iter(keys)):UAVFlowParquetDataset._summarize(x,next(iter(self.datasets.values())).action_dim)}

    def compute_proprio_statistics(self, max_samples=None):
        keys={ds.action_stats_key for ds in self.datasets.values()}
        if len(keys)!=1:
            raise ValueError(f"Shared UAV mixture normalization requires one stats key, got {keys}")
        arrays=[]
        for name,ds in self.datasets.items():
            n=max_samples
            if max_samples is not None and int(max_samples)>0:
                n=max(1,int(round(int(max_samples)*self.weights[name])))
            arrays.append(ds._proprio_statistics_values(n))
        x=np.concatenate(arrays,axis=0)
        return {next(iter(keys)):UAVFlowParquetDataset._summarize(x,6)}

    def compute_episode_pose_statistics(self, max_samples=None):
        arrays=[]
        for name,ds in self.datasets.items():
            n=max_samples
            if max_samples is not None and int(max_samples)>0:
                n=max(1,int(round(int(max_samples)*self.weights[name])))
            arrays.append(ds._episode_pose_statistics_values(n))
        x=np.concatenate(arrays,axis=0)
        stats=UAVFlowParquetDataset._summarize(x,5)
        stats["mask"]=np.asarray([True,True,True,False,False],dtype=bool)
        return stats
