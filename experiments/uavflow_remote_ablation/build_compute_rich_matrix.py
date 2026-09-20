#!/usr/bin/env python3
"""Build a curated interaction-aware UAV-Flow ablation matrix."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


FIELDS = ("pose", "language", "scale", "dynamic", "depth_target", "K", "context", "schedule")
BASE = {
    "pose": "off", "language": "qwen", "scale": "relative", "dynamic": "1",
    "depth_target": "future", "K": "5", "context": "h1", "schedule": "constant",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).expanduser()
    overrides_root = root / "cell_overrides"
    overrides_root.mkdir(parents=True, exist_ok=True)

    rows: OrderedDict[tuple[str, ...], dict] = OrderedDict()

    def add(block: str, **updates: object) -> None:
        row = BASE | {key: str(value) for key, value in updates.items()}
        key = tuple(row[field] for field in FIELDS)
        if key not in rows:
            rows[key] = row | {"blocks": [block]}
        elif block not in rows[key]["blocks"]:
            rows[key]["blocks"].append(block)

    # Reference plus the clean Stage-1 scheduler comparison.
    add("baseline")
    add("scheduler_baseline", schedule="gam_cosine")

    # Conditioning interaction: language semantics and direct state may trade off.
    for pose in ("off", "on"):
        for language in ("qwen", "t5"):
            add("pose_x_language", pose=pose, language=language)

    # Geometry definition and temporal target are tightly coupled.
    for scale in ("relative", "fixed", "separated"):
        for target in ("future", "current", "both"):
            add("scale_x_depth_target", scale=scale, depth_target=target)

    # Dynamic emphasis may help future targets but be unnecessary for current depth.
    for dynamic in ("1", "3", "5", "10"):
        for target in ("future", "current", "both"):
            add("dynamic_x_depth_target", dynamic=dynamic, depth_target=target)

    # K changes both action chunk and future geometry difficulty.
    for k in ("3", "5", "7", "10"):
        for target in ("future", "current", "both"):
            add("K_x_depth_target", K=k, depth_target=target)

    # Test whether longer history specifically compensates for larger K.
    for k in ("3", "5", "7", "10"):
        for context in ("h1", "multi1234"):
            add("K_x_context", K=k, context=context)

    # Scale choice can change how strongly sparse dynamic pixels contribute.
    for scale in ("relative", "fixed", "separated"):
        for dynamic in ("1", "3", "5", "10"):
            add("scale_x_dynamic", scale=scale, dynamic=dynamic)

    # Metric/shape scale choices may behave differently as prediction distance grows.
    for scale in ("relative", "fixed", "separated"):
        for k in ("3", "5", "7", "10"):
            add("scale_x_K", scale=scale, K=k)

    # Smaller targeted interactions omitted by the blocks above.
    for target in ("future", "current", "both"):
        for context in ("h1", "multi1234"):
            add("context_x_depth_target", context=context, depth_target=target)
        for language in ("qwen", "t5"):
            add("language_x_depth_target", language=language, depth_target=target)
    for k in ("3", "5", "7", "10"):
        for pose in ("off", "on"):
            add("pose_x_K", pose=pose, K=k)

    # One deliberately hard setting tests whether conditioning interactions only
    # become visible at long horizon and stronger dynamic supervision.
    for pose in ("off", "on"):
        for language in ("qwen", "t5"):
            add(
                "conditioning_hard_K10", pose=pose, language=language,
                dynamic="5", depth_target="both", K="10",
            )
    add(
        "conditioning_hard_K10", dynamic="5", depth_target="both", K="10",
        context="multi1234",
    )

    # Scheduler robustness on representative non-baseline configurations, not
    # on every row: pose, language, dynamic+both, long-K, and multi-context.
    add("scheduler_representatives", pose="on", schedule="gam_cosine")
    add("scheduler_representatives", language="t5", schedule="gam_cosine")
    add(
        "scheduler_representatives", dynamic="5", depth_target="both",
        schedule="gam_cosine",
    )
    add("scheduler_representatives", K="10", schedule="gam_cosine")
    add("scheduler_representatives", context="multi1234", schedule="gam_cosine")

    manifest = root / "matrix_manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("index", "label", "blocks", *FIELDS, "stage2_schedule"))
        for index, row in enumerate(rows.values()):
            label = (
                f"p-{row['pose']}_l-{row['language']}_d-{row['scale']}_"
                f"w-{row['dynamic']}_t-{row['depth_target']}_k-{row['K']}_"
                f"h-{row['context']}_s-{row['schedule']}"
            )
            writer.writerow((
                index, label, ",".join(row["blocks"]),
                *(row[field] for field in FIELDS), "cosine",
            ))
            values = [
                f"model.use_pose_history={'true' if row['pose'] == 'on' else 'false'}",
            ]
            if row["language"] == "t5":
                values += ["stage1.text_encoder_type=t5", "model.language_len=77"]
            else:
                values += [
                    "stage1.text_encoder_type=qwen3_5",
                    "model.language_len=256",
                    "stage1.qwen_prompt_mode=current_image_instruction",
                ]
            values += [
                "loss.depth_scale_mode=" + {
                    "relative": "gam_window_pointnorm",
                    "fixed": "fixed_metric",
                    "separated": "scale_separated",
                }[row["scale"]],
                f"loss.depth_semantic_weight={row['dynamic']}",
                f"loss.depth_target_mode={row['depth_target']}",
                f"dataset.visual_anchor_stride={row['K']}",
                f"dataset.chunk_size={row['K']}",
                f"dataset.endpoint_absorbing_max_starts={row['K']}",
                f"model.action_chunk_size={row['K']}",
            ]
            if row["context"] == "multi1234":
                values += [
                    "dataset.future_steps=4",
                    "model.context_lengths=[1,2,3,4]",
                    "model.context_weights=[0.25,0.25,0.25,0.25]",
                ]
            else:
                values += [
                    "dataset.future_steps=1",
                    "model.context_lengths=[1]",
                    "model.context_weights=[1.0]",
                ]
            (overrides_root / f"{label}.txt").write_text("\n".join(values) + "\n")

    print(f"wrote {manifest}: cells={len(rows)}")


if __name__ == "__main__":
    main()
