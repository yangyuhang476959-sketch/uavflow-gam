"""Presentation-only training metrics; no loss or gradient changes."""
import re


def action_statistics(moments):
    count, pred_sum, pred_sq, gt_sum, gt_sq, error = moments.double()
    denom = count.clamp_min(1)
    pred_mean, gt_mean = pred_sum / denom, gt_sum / denom
    pred_var = (pred_sq / denom - pred_mean.square()).clamp_min(0)
    gt_var = (gt_sq / denom - gt_mean.square()).clamp_min(0)
    return dict(pred_mean=pred_mean, gt_mean=gt_mean,
                pred_var=pred_var, gt_var=gt_var,
                pred_std=pred_var.sqrt(), gt_std=gt_var.sqrt(),
                mae=error / denom)


def enabled_metric(key, cfg):
    key = re.sub(r"^H\d+_", "", key)
    loss = cfg.loss
    if key.startswith("step"):
        if key.endswith("_cls"):
            return float(loss.get("feature_cls_weight", 0)) > 0
        if key.endswith("_reg"):
            return float(loss.get("feature_register_weight", 0)) > 0
    if key.startswith("stop"):
        return float(loss.get("stop_weight", 0)) > 0
    for prefix, weight in (("action_pose", "action_rollout_pose_weight"),
                           ("pose_cons", "pose_consistency_weight"),
                           ("pose", "relative_pose_weight"),
                           ("deep", "deep_feature_weight")):
        if key.startswith(prefix):
            return float(loss.get(weight, 0)) > 0
    if key.startswith("depth") or key.startswith("weighted_depth") or key == "scene_scale":
        if max(float(loss.get(name, 0)) for name in
               ("depth_weight", "depth_global_weight", "depth_dynamic_weight")) <= 0:
            return False
        if "scale" in key:
            return loss.get("depth_scale_mode") == "scale_separated"
        if key == "weighted_depth_dynamic":
            return float(loss.get("depth_dynamic_weight", 0)) > 0
    if key in {"direct", "direct_action", "direct_raw"}:
        return float(loss.get("action_direct_weight", 0)) > 0
    if key in {"cls", "feat_cls", "reg", "feat_reg"}:
        weight = "feature_cls_weight" if "cls" in key else "feature_register_weight"
        return float(loss.get(weight, 0)) > 0
    if key in {"gate", "refine", "patch", "feat_steps", "cos_steps"}:
        return False
    return True


def compact_train_line(line, values, cfg):
    # Existing legacy line has list-valued pose fields; split only at key=.
    fields = re.split(r" (?=[A-Za-z_][A-Za-z_0-9/+\-]*=)", line)
    fields = [field for field in fields if enabled_metric(field.split("=", 1)[0], cfg)]
    for key, value in values.items():
        if key.startswith(("depth_current_", "depth_future_")):
            fields.append(f"{key}={float(value):.5f}")
    for key, value in action_statistics(values["action_raw_moments"]).items():
        vector = ",".join(f"{x:.6f}" for x in value.tolist())
        fields.append(f"action_raw_{key}_xyzyaw=[{vector}]")
    fields.append("action_stats_scope=rank0_batch units=m,m,m,rad")
    return " ".join(fields)
