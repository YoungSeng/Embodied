#!/usr/bin/env python3
"""Build a training-only UI5 GT-repair crop audit from immutable v3 results.

This command never runs PP-OCRv5 or OmniParser and never edits the v3 audit or
``detections/merged/detections.jsonl``.  It applies task/sample-scoped repair
detections only to training materialization, writes a distinct v4 audit, and
leaves final training authorization to ``build_ui5_crop_training_recipe.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from analyze_ui5_source_overlap import content_fingerprint
from run_ui5_crop_audit import (
    AuditPaths,
    METRIC_DEFINITIONS,
    ProgressReporter,
    REGION_TASKS,
    TASK_AWARE_CANDIDATES,
    TASK_NAMES,
    V4_FINAL_TRAINING_GATE_CONDITIONS,
    aggregate_scope,
    atomic_save_png,
    atomic_write_json,
    atomic_write_jsonl,
    audit_input_snapshot,
    audit_state_digest,
    build_preview_rows,
    digest_ids,
    load_parser_module,
    make_image_detail,
    materialize_image_record,
    materialization_reuse_metrics,
    open_raw_image,
    planned_crop_paths,
    planned_overview_path,
    proposal_crops,
    read_jsonl,
    rect_contains,
    rect_intersects,
    rectangle_union_area,
    save_overview,
    stable_id,
    write_excel_report,
    write_statistics_csv,
)


SOURCE_CONFIG = "TA_CTX015_H050"
REPAIR_CONFIG = "TA_CTX015_H050_GT_REPAIR"
EXCLUDED_SAMPLE_ID = "sample_3a3922c5762298f04c8d"
EXCLUDED_TASK = "ui_text_overflow"
EXPECTED_RAW_FAILURES = 107
EXPECTED_VALID_REPAIRS = 106
EXPECTED_REGION_GT_TOTAL_RAW = 17798
EXPECTED_REGION_GT_CONTAINED_RAW = 17691
EXPECTED_REGION_GT_TOTAL_CLEAN = 17797
EXPECTED_VALID_REPAIRS_BY_TASK = {
    "ui_occlusion": 46,
    "ui_cropping": 23,
    "ui_text_ellipsis": 35,
    "ui_text_overflow": 2,
}
V4_GATE_CONDITIONS = V4_FINAL_TRAINING_GATE_CONDITIONS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parser-root", type=Path, required=True)
    parser.add_argument("--source-audit-name", default="crop_audit_v3")
    parser.add_argument("--crop-audit-name", default="crop_audit_v4_gt_repair")
    parser.add_argument("--source-config", default=SOURCE_CONFIG)
    parser.add_argument("--repair-config", default=REPAIR_CONFIG)
    parser.add_argument("--expected-unique-images", type=int, default=17281)
    parser.add_argument("--expected-raw-failures", type=int, default=EXPECTED_RAW_FAILURES)
    parser.add_argument("--expected-valid-repairs", type=int, default=EXPECTED_VALID_REPAIRS)
    parser.add_argument("--max-crops", type=int, default=10)
    parser.add_argument("--boundary-margin-ratio", type=float, default=0.01)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _combined_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for path in sorted((value.resolve() for value in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_fingerprint(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _intersection_area(left: Sequence[int], right: Sequence[int]) -> int:
    width = max(0, min(int(left[2]), int(right[2])) - max(int(left[0]), int(right[0])))
    height = max(0, min(int(left[3]), int(right[3])) - max(int(left[1]), int(right[1])))
    return width * height


def _box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _union(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return [
        min(int(left[0]), int(right[0])),
        min(int(left[1]), int(right[1])),
        max(int(left[2]), int(right[2])),
        max(int(left[3]), int(right[3])),
    ]


def _clip(box: Sequence[int], width: int, height: int) -> list[int]:
    result = [
        max(0, min(width, math.floor(float(box[0])))),
        max(0, min(height, math.floor(float(box[1])))),
        max(0, min(width, math.ceil(float(box[2])))),
        max(0, min(height, math.ceil(float(box[3])))),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"zero-area repaired crop: {result}")
    return result


def _merge_overlaps(boxes: list[list[int]]) -> list[list[int]]:
    boxes = [list(map(int, box)) for box in boxes]
    while True:
        pair = next(
            (
                (left, right)
                for left in range(len(boxes))
                for right in range(left + 1, len(boxes))
                if rect_intersects(boxes[left], boxes[right])
            ),
            None,
        )
        if pair is None:
            break
        left, right = pair
        merged = _union(boxes[left], boxes[right])
        boxes.pop(right)
        boxes.pop(left)
        boxes.append(merged)
    return boxes


def _merge_to_limit(boxes: list[list[int]], max_crops: int) -> list[list[int]]:
    boxes = _merge_overlaps(boxes)
    while len(boxes) > max_crops:
        left, right = min(
            (
                (left, right)
                for left in range(len(boxes))
                for right in range(left + 1, len(boxes))
            ),
            key=lambda pair: (
                _box_area(_union(boxes[pair[0]], boxes[pair[1]]))
                - _box_area(boxes[pair[0]])
                - _box_area(boxes[pair[1]]),
                pair,
            ),
        )
        merged = _union(boxes[left], boxes[right])
        boxes.pop(right)
        boxes.pop(left)
        boxes.append(merged)
        boxes = _merge_overlaps(boxes)
    return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def _context_box(gt: Sequence[int], width: int, height: int, ratio: float) -> list[int]:
    gt_width = max(1, int(gt[2]) - int(gt[0]))
    gt_height = max(1, int(gt[3]) - int(gt[1]))
    pad_x = max(1, math.ceil(gt_width * ratio))
    pad_y = max(1, math.ceil(gt_height * ratio))
    return _clip(
        [gt[0] - pad_x, gt[1] - pad_y, gt[2] + pad_x, gt[3] + pad_y],
        width,
        height,
    )


def make_repair_detection(
    failure: Mapping[str, Any], *, audit_source: str, split: str
) -> dict[str, Any]:
    return {
        "bbox": [int(value) for value in failure["gt_bbox"]],
        "source": "manual_gt_repair",
        "task": str(failure["task"]),
        "sample_id": str(failure["sample_id"]),
        "gt_index": int(failure["gt_index"]),
        "failure_type": str(failure["failure_type"]),
        "split": split,
        "audit_source": audit_source,
    }


def repair_sample_geometry(
    cropper: Any,
    *,
    sample: Mapping[str, Any],
    source_result: Mapping[str, Any],
    detection: Mapping[str, Any],
    task_rule: Mapping[str, float],
    source_audit_name: str,
    max_crops: int,
    boundary_margin_ratio: float,
) -> tuple[list[list[int]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Repair one image×task without mutating task-agnostic detections."""

    width, height = int(sample["width"]), int(sample["height"])
    original_boxes = [list(map(int, box)) for box in source_result["crop_boxes"]]
    boxes = [list(box) for box in original_boxes]
    failures = [dict(row) for row in source_result.get("failures", [])]
    split = str(sample.get("split", ""))
    if failures and split != "train":
        raise RuntimeError(
            f"GT repair is training-only; refusing sample={sample['sample_id']} split={split!r}"
        )
    repair_detections = [
        make_repair_detection(row, audit_source=source_audit_name, split=split)
        for row in failures
    ]
    actions: list[dict[str, Any]] = []

    for failure, repair in zip(failures, repair_detections):
        if str(failure["sample_id"]) != str(sample["sample_id"]):
            raise ValueError("source failure crossed task sample boundary")
        gt = repair["bbox"]
        if failure["failure_type"] != "partial_intersection":
            continue
        candidates = [
            (index, _intersection_area(box, gt))
            for index, box in enumerate(boxes)
            if rect_intersects(box, gt)
        ]
        if not candidates:
            raise ValueError(f"partial failure has no intersecting crop: {repair}")
        target_index = max(candidates, key=lambda item: (item[1], -item[0]))[0]
        before = list(boxes[target_index])
        boxes[target_index] = _clip(_union(before, gt), width, height)
        actions.append(
            {
                **repair,
                "action": "expand_max_intersection_crop",
                "before_crop_bbox": before,
                "after_crop_bbox": list(boxes[target_index]),
                "gt_unchanged": True,
            }
        )

    uncovered = [
        repair
        for failure, repair in zip(failures, repair_detections)
        if failure["failure_type"] == "uncovered"
    ]
    if uncovered:
        augmented = dict(detection)
        augmented["text_detections"] = [
            dict(row) for row in detection.get("text_detections", [])
        ]
        augmented["icon_detections"] = [
            dict(row) for row in detection.get("icon_detections", [])
        ]
        destination = (
            "text_detections" if str(sample["task"]).startswith("ui_text_") else "icon_detections"
        )
        for repair in uncovered:
            augmented[destination].append(
                {"bbox": list(repair["bbox"]), "score": 1.0, **repair}
            )
        proposal = proposal_crops(
            cropper,
            augmented,
            task_rule,
            max_crops=max_crops,
            boundary_margin_ratio=boundary_margin_ratio,
            min_context_image_ratio=float(task_rule.get("min_context_image_ratio", 0.0)),
        )
        for repair in uncovered:
            gt = repair["bbox"]
            candidates = [
                list(box) for box in proposal["crop_boxes"] if rect_contains(box, gt)
            ]
            candidate = (
                min(candidates, key=_box_area)
                if candidates
                else _context_box(gt, width, height, float(task_rule["context_ratio"]))
            )
            boxes.append(candidate)
            actions.append(
                {
                    **repair,
                    "action": "add_task_scoped_regenerated_crop",
                    "before_crop_bbox": None,
                    "after_crop_bbox": list(candidate),
                    "gt_unchanged": True,
                }
            )

    boxes = _merge_to_limit(boxes, max_crops)
    for repair in repair_detections:
        gt = repair["bbox"]
        if any(rect_contains(box, gt) for box in boxes):
            continue
        intersecting = [
            (index, _intersection_area(box, gt))
            for index, box in enumerate(boxes)
            if rect_intersects(box, gt)
        ]
        if intersecting:
            target = max(intersecting, key=lambda item: (item[1], -item[0]))[0]
            boxes[target] = _clip(_union(boxes[target], gt), width, height)
        elif len(boxes) < max_crops:
            boxes.append(_context_box(gt, width, height, float(task_rule["context_ratio"])))
        else:
            target = min(
                range(len(boxes)),
                key=lambda index: _box_area(_union(boxes[index], gt)) - _box_area(boxes[index]),
            )
            boxes[target] = _clip(_union(boxes[target], gt), width, height)
        boxes = _merge_to_limit(boxes, max_crops)

    for gt in sample["gt_boxes"]:
        if not any(rect_contains(box, gt) for box in boxes):
            raise AssertionError(
                f"post-repair GT remains uncovered: sample={sample['sample_id']} gt={gt}"
            )
    if not 1 <= len(boxes) <= max_crops:
        raise AssertionError(f"post-repair crop count is invalid: {len(boxes)}")

    repair_gt_boxes = [tuple(row["bbox"]) for row in repair_detections]
    provenance = [
        (
            "manual_gt_repair"
            if any(rect_contains(box, gt) for gt in repair_gt_boxes)
            and list(box) not in original_boxes
            else "raw_detector"
        )
        for box in boxes
    ]
    for action in actions:
        action["final_containing_crop_bboxes"] = [
            list(box) for box in boxes if rect_contains(box, action["bbox"])
        ]
        if not action["final_containing_crop_bboxes"]:
            raise AssertionError(f"repair action did not produce a containing crop: {action}")
    return boxes, repair_detections, actions, provenance


def build_exclusion_evidence(
    *,
    target_audit: Path,
    source_audit: Path,
    samples_by_id: Mapping[str, Mapping[str, Any]],
    excluded_sample_id: str = EXCLUDED_SAMPLE_ID,
) -> list[dict[str, Any]]:
    sample = dict(samples_by_id[excluded_sample_id])
    if sample["task"] != EXCLUDED_TASK:
        raise ValueError(f"excluded sample task mismatch: {sample['task']}")
    visualized = {
        (str(row["sample_id"]), int(row["gt_index"])): row
        for row in read_jsonl(source_audit / "gt_failures_visualized.jsonl")
    }
    matching = [
        row for (sample_id, _), row in visualized.items() if sample_id == excluded_sample_id
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one visualized failure for excluded sample, found {len(matching)}"
        )
    failure = matching[0]
    evidence = (
        target_audit
        / "excluded_annotation_cases"
        / EXCLUDED_TASK
        / excluded_sample_id
    )
    evidence.mkdir(parents=True, exist_ok=True)
    gt_record_path = evidence / "gt_record.json"
    if gt_record_path.is_file():
        existing = json.loads(gt_record_path.read_text(encoding="utf-8"))
        excluded_at = existing["excluded_at"]
    else:
        excluded_at = datetime.now(timezone.utc).isoformat()
    gt_record = {
        "sample_id": excluded_sample_id,
        "image_id": sample["image_id"],
        "task": sample["task"],
        "gt_index": int(failure["gt_index"]),
        "gt_bbox": failure["gt_bbox"],
        "manual_conclusion": "annotation_error",
        "excluded_at": excluded_at,
        "audit_source": source_audit.name,
        "source_data_deleted": False,
    }
    atomic_write_json(gt_record_path, gt_record)
    atomic_write_json(evidence / "task_sample_record.json", sample)
    visualization = Path(str(failure["visualization_4panel"])).resolve(strict=True)
    shutil.copy2(visualization, evidence / "visualization_4panel.png")
    source_image = Path(str(sample["canonical_path"])).resolve(strict=True)
    with open_raw_image(source_image) as image:
        atomic_save_png(image, evidence / "source_image.png")
    _atomic_write_text(
        evidence / "README.md",
        "# UI5 text overflow 标注排除证据\n\n"
        f"`{excluded_sample_id}` 的 `ui_text_overflow` GT 经人工复核存在标注问题，"
        "因此只从该任务训练数据中排除。原始图片、原始 JSONL、v3 audit 和其他任务监督均未删除或修改。\n",
    )
    if not source_image.is_file():
        raise AssertionError("source image disappeared while copying evidence")
    return [
        {
            "sample_id": excluded_sample_id,
            "image_id": sample["image_id"],
            "task": EXCLUDED_TASK,
            "reason": "annotation_error",
            "evidence_dir": str(evidence.resolve()),
            "source_records": sample.get("source_records", []),
        }
    ]


def _detection_bbox(row: Mapping[str, Any]) -> list[int] | None:
    value = row.get("bbox", row.get("bbox_2d"))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return [int(round(float(item))) for item in value]


def _draw_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[int]],
    *,
    color: tuple[int, int, int],
    prefix: str,
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(boxes, 1):
        draw.rectangle(tuple(map(int, box)), outline=color, width=width)
        draw.text((int(box[0]) + width, int(box[1]) + width), f"{prefix}{index}", fill=color)


def render_repair_visualizations(
    *,
    target_audit: Path,
    actions: list[dict[str, Any]],
    unique_by_id: Mapping[str, Mapping[str, Any]],
    detections: Mapping[str, Mapping[str, Any]],
    source_results_by_sample: Mapping[str, Mapping[str, Any]],
    repaired_results_by_sample: Mapping[str, Mapping[str, Any]],
    progress_interval_seconds: float = 10.0,
    resume: bool = False,
) -> None:
    """Render training-only repair evidence without loading either detector."""

    root = target_audit / "gt_repair_visualizations"
    gallery_rows: list[str] = []
    output_paths = [
        root
        / str(action["task"])
        / str(action["failure_type"])
        / f"{action['sample_id']}__gt{int(action['gt_index'])}.png"
        for action in actions
    ]
    reused = sum(resume and path.is_file() and path.stat().st_size > 0 for path in output_paths)
    reporter = ProgressReporter(
        stage="gt-repair-visualizations",
        total=len(actions),
        output_dir=target_audit,
        interval_seconds=progress_interval_seconds,
        initial_completed=reused,
        unit="GT failures",
    )
    reporter.update(
        reused,
        detail=f"复用 {reused} 张已有四联图",
        force=True,
    )
    completed = reused
    for action, output in zip(actions, output_paths):
        sample_id = str(action["sample_id"])
        action["visualization_4panel"] = str(output.resolve())
        relative = output.relative_to(root).as_posix()
        if resume and output.is_file() and output.stat().st_size > 0:
            gallery_rows.append(
                f'<article data-task="{action["task"]}" data-failure="{action["failure_type"]}">'
                f"<h3>{sample_id} · GT {action['gt_index']}</h3>"
                f'<a href="../{relative}"><img loading="lazy" src="../{relative}"></a></article>'
            )
            continue
        repaired = repaired_results_by_sample[sample_id]
        image_id = str(repaired["image_id"])
        source_result = source_results_by_sample[sample_id]
        manifest = unique_by_id[image_id]
        image_path = Path(str(manifest["image_path"])).resolve(strict=True)
        with open_raw_image(image_path) as opened:
            original = opened.convert("RGB")
        line = max(2, round(min(original.size) / 350))
        panels = [original.copy() for _ in range(4)]
        detection = detections[image_id]
        text_boxes = [
            box
            for row in detection.get("text_detections", [])
            if (box := _detection_bbox(row)) is not None
        ]
        icon_boxes = [
            box
            for row in detection.get("icon_detections", [])
            if (box := _detection_bbox(row)) is not None
        ]
        _draw_boxes(panels[1], text_boxes, color=(20, 180, 60), prefix="T", width=line)
        _draw_boxes(panels[1], icon_boxes, color=(255, 140, 0), prefix="I", width=line)
        gt = [action["bbox"]]
        _draw_boxes(
            panels[2], source_result["crop_boxes"], color=(30, 120, 255), prefix="C", width=line
        )
        _draw_boxes(panels[2], gt, color=(255, 30, 30), prefix="GT", width=line * 2)
        _draw_boxes(
            panels[3], repaired["crop_boxes"], color=(30, 120, 255), prefix="C", width=line
        )
        _draw_boxes(panels[3], gt, color=(255, 30, 30), prefix="GT", width=line * 2)

        display_width = min(640, original.width)
        display_height = max(1, round(original.height * display_width / original.width))
        title_height, footer_height = 34, 50
        canvas = Image.new(
            "RGB", (display_width * 4, display_height + title_height + footer_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        titles = ("Original", "Raw text + icon detections", "v3 GT + crops", "v4 repaired GT + crops")
        for index, (panel, title) in enumerate(zip(panels, titles)):
            rendered = panel.resize((display_width, display_height), Image.Resampling.LANCZOS)
            canvas.paste(rendered, (index * display_width, title_height))
            draw.text((index * display_width + 5, 10), title, fill="black", font=font)
            rendered.close()
            panel.close()
        original.close()
        footer = (
            f"{action['task']} | {action['failure_type']} | {sample_id} | gt{action['gt_index']} | "
            f"GT={action['bbox']} | action={action['action']} | training only; inference GT repair disabled"
        )
        draw.text((5, title_height + display_height + 8), footer, fill="black", font=font)
        atomic_save_png(canvas, output)
        canvas.close()
        gallery_rows.append(
            f'<article data-task="{action["task"]}" data-failure="{action["failure_type"]}">'
            f"<h3>{sample_id} · GT {action['gt_index']}</h3>"
            f'<a href="../{relative}"><img loading="lazy" src="../{relative}"></a></article>'
        )
        completed += 1
        reporter.update(completed, detail=f"当前 {sample_id}")
    gallery = root / "gallery" / "index.html"
    _atomic_write_text(
        gallery,
        "<!doctype html><meta charset='utf-8'><title>UI5 GT repair evidence</title>"
        "<style>body{font-family:sans-serif}article{margin:24px 0}img{max-width:100%;height:auto}</style>"
        "<h1>UI5 training-only GT repair evidence</h1>"
        "<p>These views use saved detector JSONL only. GT repair is forbidden in validation, test and inference.</p>"
        + "\n".join(gallery_rows),
    )
    reporter.update(len(actions), status="completed", detail="四联图与 gallery 已完成", force=True)


def _geometry_shard_valid(
    source: Path, output: Path, marker: Path, state_digest: str
) -> bool:
    if not (output.is_file() and marker.is_file()):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return (
            payload.get("stage") == "gt-repair-geometry"
            and payload.get("state_digest") == state_digest
            and payload.get("source_digest") == content_fingerprint(source)
            and payload.get("count") == len(read_jsonl(output))
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _material_shard_valid(
    source: Path, output: Path, marker: Path, state_digest: str
) -> bool:
    if not (output.is_file() and marker.is_file()):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        rows = read_jsonl(output)
        paths = [Path(path) for row in rows for path in row.get("region_paths", [])]
        return (
            payload.get("stage") == "gt-repair-materialize"
            and payload.get("state_digest") == state_digest
            and payload.get("source_digest") == content_fingerprint(source)
            and payload.get("count") == len(rows)
            and all(path.is_file() and path.stat().st_size > 0 for path in paths)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _candidate_summary(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scope = {"ALL": aggregate_scope(details)}
    by_scope["REGION_ALL"] = aggregate_scope(
        [row for row in details if row["task"] in REGION_TASKS]
    )
    for task in TASK_NAMES:
        by_scope[task] = aggregate_scope([row for row in details if row["task"] == task])
    return {
        "parameters": {
            **TASK_AWARE_CANDIDATES[SOURCE_CONFIG],
            "training_only_gt_repair": True,
        },
        "by_scope": by_scope,
    }


def validate_repair_inventory(
    detections: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_by_task: Mapping[str, int] | None = None,
) -> None:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        return str(row["task"]), str(row["sample_id"]), int(row["gt_index"])

    detection_keys = [key(row) for row in detections]
    action_keys = [key(row) for row in actions]
    if len(detection_keys) != expected_count or len(set(detection_keys)) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique valid repair detections, found "
            f"rows={len(detection_keys)} unique={len(set(detection_keys))}"
        )
    if len(action_keys) != expected_count or set(action_keys) != set(detection_keys):
        raise ValueError("repair actions do not map one-to-one to repair detections")
    if any(row.get("source") != "manual_gt_repair" for row in detections):
        raise ValueError("repair detection source must be manual_gt_repair")
    if any(row.get("split") != "train" for row in detections):
        raise ValueError("GT repair is forbidden outside the training split")
    if expected_by_task is not None:
        counts = defaultdict(int)
        for task, _, _ in detection_keys:
            counts[task] += 1
        if dict(sorted(counts.items())) != dict(sorted(expected_by_task.items())):
            raise ValueError(
                f"valid repair task distribution changed: actual={dict(counts)}, "
                f"expected={dict(expected_by_task)}"
            )


def _build_v4_gate(
    *,
    details: Sequence[Mapping[str, Any]],
    region: Mapping[str, Any],
    by_scope: Mapping[str, Mapping[str, Any]],
    cross_train_val_count: int,
    content_missing_mismatch_count: int,
    input_unchanged: bool,
    excluded_count: int,
    repair_split_violations: int,
    reports_written: bool,
) -> dict[str, Any]:
    conditions = {
        "region_overall_recall_at_least_0_99": region["gt_box_containment_recall"] >= 0.99,
        "each_region_task_recall_at_least_0_98": all(
            by_scope[task]["gt_box_containment_recall"] >= 0.98 for task in REGION_TASKS
        ),
        "detector_boundary_cut_count_zero": region["detector_boundary_cut_count"] == 0,
        "region_roundtrip_error_over_1_count_zero": region["roundtrip_error_over_1_count"] == 0,
        "partial_crop_training_eligible_count_zero": all(
            int(row.get("partial_training_eligible_count", 0)) == 0
            for row in details
            if row.get("task") in REGION_TASKS
        ),
        "hard_negative_max_one_per_image_task": all(
            int(row.get("hard_negative_count", 0)) <= 1
            for row in details
            if row.get("task") in REGION_TASKS
        ),
        "same_content_cross_train_val_count_zero": int(cross_train_val_count) == 0,
        "content_missing_recall_equals_1": math.isclose(
            float(by_scope["ui_content_missing"]["gt_box_containment_recall"]),
            1.0,
            abs_tol=1e-12,
        ),
        "content_missing_normalized_gt_mismatch_count_zero": content_missing_mismatch_count == 0,
        "input_snapshot_unchanged": bool(input_unchanged),
        "all_reports_written_successfully": bool(reports_written),
        "excluded_annotation_count_equals_1": excluded_count == 1,
        "excluded_sample_absent_from_text_overflow_recipe": False,
        "valid_gt_post_repair_recall_equals_1": math.isclose(
            float(region["gt_box_containment_recall"]), 1.0, abs_tol=1e-12
        ),
        "post_repair_partial_count_zero": int(region["partial_only_gt_count"]) == 0,
        "post_repair_uncovered_count_zero": int(region["uncovered_gt_count"]) == 0,
        "crop_training_recipe_written_successfully": False,
        "crop_training_recipe_contains_crop_records": False,
        "gt_repair_not_applied_to_val_test": repair_split_violations == 0,
    }
    if set(conditions) != V4_GATE_CONDITIONS:
        raise AssertionError("v4 gate schema changed unexpectedly")
    return {
        "conditions": conditions,
        "passes": all(conditions.values()),
        "training_ready": False,
        "training_started": False,
        "failed_conditions": [name for name, passed in conditions.items() if not passed],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve(strict=True)
    parser_root = args.parser_root.expanduser().resolve(strict=True)
    source_audit = output_dir / args.source_audit_name
    target_audit = output_dir / args.crop_audit_name
    if source_audit == target_audit:
        raise ValueError("source and target crop audit directories must differ")
    if not source_audit.is_dir():
        raise FileNotFoundError(source_audit)
    marker_path = target_audit / "training_ready.json"
    marker_path.unlink(missing_ok=True)

    paths = AuditPaths(output_dir, args.crop_audit_name)
    unique = read_jsonl(paths.unique_images)
    samples = read_jsonl(paths.task_samples)
    detections_rows = read_jsonl(paths.merged)
    if args.expected_unique_images and len(unique) != args.expected_unique_images:
        raise ValueError(
            f"expected {args.expected_unique_images} unique images, found {len(unique)}"
        )
    unique_by_id = {str(row["image_id"]): row for row in unique}
    samples_by_id = {str(row["sample_id"]): row for row in samples}
    samples_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_image[str(sample["image_id"])].append(sample)
    detections = {str(row["image_id"]): row for row in detections_rows}
    if len(unique_by_id) != len(unique) or len(samples_by_id) != len(samples):
        raise ValueError("duplicate image_id/sample_id in manifest")

    source_geometry_dir = source_audit / f"candidate_{args.source_config}" / "geometry"
    source_shards = sorted(source_geometry_dir.glob("shard_*.jsonl"))
    if not source_shards:
        raise FileNotFoundError(f"no source geometry shards: {source_geometry_dir}")
    source_failures = read_jsonl(source_audit / "gt_failures.jsonl")
    if len(source_failures) != args.expected_raw_failures:
        raise ValueError(
            f"expected {args.expected_raw_failures} v3 failures, found {len(source_failures)}"
        )
    if {str(row["config"]) for row in source_failures} != {args.source_config}:
        raise ValueError("source failure config differs from requested source config")
    if EXCLUDED_SAMPLE_ID not in samples_by_id:
        raise KeyError(f"excluded sample is missing from manifest: {EXCLUDED_SAMPLE_ID}")

    input_before = audit_input_snapshot(paths, unique, detections_rows, samples)
    detection_digest_before = content_fingerprint(paths.merged)
    state = {
        "schema_version": 4,
        "audit_name": args.crop_audit_name,
        "source_audit": args.source_audit_name,
        "source_config": args.source_config,
        "repair_config": args.repair_config,
        "unique_images": len(unique),
        "image_id_digest": digest_ids(unique_by_id),
        "merged_detections_digest": detection_digest_before,
        "source_geometry_digest": _combined_digest(source_shards),
        "source_failures_digest": content_fingerprint(source_audit / "gt_failures.jsonl"),
        "excluded_sample_id": EXCLUDED_SAMPLE_ID,
        "max_crops": int(args.max_crops),
        "boundary_margin_ratio": float(args.boundary_margin_ratio),
        "detector_stages_executed": [],
        "training_only_gt_repair": True,
    }
    state_digest = audit_state_digest(state)
    state_path = target_audit / "audit_state.json"
    if state_path.is_file():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing != state:
            raise RuntimeError(
                f"existing v4 audit state differs: {state_path}; use a new audit name"
            )
    else:
        if target_audit.exists() and any(target_audit.iterdir()):
            raise RuntimeError(
                f"target audit already contains unrelated output: {target_audit}"
            )
        target_audit.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_path, state)

    exclusions = build_exclusion_evidence(
        target_audit=target_audit,
        source_audit=source_audit,
        samples_by_id=samples_by_id,
    )
    atomic_write_jsonl(target_audit / "excluded_training_samples.jsonl", exclusions)

    cropper = load_parser_module(parser_root, "ui_region_cropper")
    config_root = target_audit / f"candidate_{args.repair_config}"
    geometry_dir = config_root / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    reusable_geometry: dict[Path, bool] = {}
    geometry_reused = 0
    for source_shard in source_shards:
        output_shard = geometry_dir / source_shard.name
        done = geometry_dir / f"{source_shard.stem}.done.json"
        valid = bool(
            args.resume
            and _geometry_shard_valid(source_shard, output_shard, done, state_digest)
        )
        reusable_geometry[source_shard] = valid
        if valid:
            geometry_reused += int(json.loads(done.read_text(encoding="utf-8"))["count"])
    geometry_reporter = ProgressReporter(
        stage="gt-repair-geometry",
        total=len(unique),
        output_dir=target_audit,
        interval_seconds=args.progress_interval_seconds,
        initial_completed=geometry_reused,
        unit="images",
    )
    geometry_reporter.update(
        geometry_reused,
        detail=f"复用 {geometry_reused} 张已完成几何结果",
        force=True,
    )
    geometry_completed = geometry_reused
    for source_shard in source_shards:
        output_shard = geometry_dir / source_shard.name
        done = geometry_dir / f"{source_shard.stem}.done.json"
        if reusable_geometry[source_shard]:
            continue
        output_rows = []
        for source_record in read_jsonl(source_shard):
            image_id = str(source_record["image_id"])
            manifest = unique_by_id[image_id]
            sample_map = {
                str(row["sample_id"]): row for row in samples_by_image[image_id]
            }
            repaired_results = []
            record_repairs: list[dict[str, Any]] = []
            record_actions: list[dict[str, Any]] = []
            for source_result in source_record["sample_results"]:
                sample_id = str(source_result["sample_id"])
                sample = sample_map[sample_id]
                if sample_id == EXCLUDED_SAMPLE_ID and sample["task"] == EXCLUDED_TASK:
                    continue
                if sample["task"] == "ui_content_missing":
                    boxes = [[0, 0, int(sample["width"]), int(sample["height"])]]
                    repairs: list[dict[str, Any]] = []
                    actions: list[dict[str, Any]] = []
                    provenance = ["full_image"]
                    crop_kind = "whole"
                    task_rule = None
                else:
                    task_rule = TASK_AWARE_CANDIDATES[args.source_config]["task_rules"][sample["task"]]
                    boxes, repairs, actions, provenance = repair_sample_geometry(
                        cropper,
                        sample=sample,
                        source_result=source_result,
                        detection=detections[image_id],
                        task_rule=task_rule,
                        source_audit_name=args.source_audit_name,
                        max_crops=args.max_crops,
                        boundary_margin_ratio=args.boundary_margin_ratio,
                    )
                    crop_kind = "region"
                planned = (
                    [Path(str(manifest["image_path"])).resolve()]
                    if crop_kind == "whole"
                    else planned_crop_paths(config_root, image_id, boxes, prefix="region")
                )
                preview, preview_failures = build_preview_rows(
                    sample, boxes, planned, config_name=args.repair_config
                )
                roundtrip_errors = [
                    int(row["roundtrip_max_error_px"]) for row in preview
                ]
                detail, failures = make_image_detail(
                    sample,
                    {
                        "detection_count": int(source_result["proposal"]["detection_count"]),
                        "edge_count": int(source_result["proposal"].get("edge_count", 0)),
                        "component_count_before_merge": int(
                            source_result["proposal"].get("component_count_before_merge", 0)
                        ),
                        "forced_merge": bool(source_result["proposal"].get("forced_merge", False)),
                        "empty_detection_fallback": bool(
                            source_result["proposal"].get("empty_detection_fallback", False)
                        ),
                        "detector_boundary_cut_count": 0,
                    },
                    boxes,
                    planned,
                    planned_overview_path(config_root, sample),
                    args.repair_config,
                    roundtrip_errors,
                )
                detail["partial_training_eligible_count"] = sum(
                    row.get("failure_type") == "partial_intersection"
                    and bool(row.get("training_eligible"))
                    for row in preview_failures
                )
                detail["hard_negative_count"] = sum(
                    row.get("training_eligible") and row.get("positive") is False
                    for row in preview
                )
                detail["overview"] = ""
                detail["crop_paths"] = []
                repaired_results.append(
                    {
                        "sample_id": sample_id,
                        "task": sample["task"],
                        "gt_boxes": sample["gt_boxes"],
                        "gt_boxes_1000": sample["gt_boxes_1000"],
                        "crop_kind": crop_kind,
                        "task_geometry_rule": dict(task_rule) if task_rule else None,
                        "crop_boxes": boxes,
                        "crop_provenance": provenance,
                        "proposal": {
                            **source_result["proposal"],
                            "detector_boundary_cut_count": 0,
                        },
                        "detail": detail,
                        "failures": failures,
                        "preview_failures": preview_failures,
                        "repair_detections": repairs,
                        "repair_actions": actions,
                    }
                )
                record_repairs.extend(repairs)
                record_actions.extend(actions)
            unique_region_boxes = sorted(
                {
                    tuple(box)
                    for row in repaired_results
                    if row["crop_kind"] == "region"
                    for box in row["crop_boxes"]
                },
                key=lambda box: (box[1], box[0], box[3], box[2]),
            )
            output_rows.append(
                {
                    "image_id": image_id,
                    "config": args.repair_config,
                    "unique_region_boxes": [list(box) for box in unique_region_boxes],
                    "whole_box": [0, 0, int(manifest["width"]), int(manifest["height"])],
                    "sample_results": repaired_results,
                    "repair_detections": record_repairs,
                    "repair_actions": record_actions,
                }
            )
            geometry_completed += 1
            geometry_reporter.update(
                geometry_completed,
                detail=f"{source_shard.name}，当前 {image_id}",
            )
        atomic_write_jsonl(output_shard, output_rows)
        atomic_write_json(
            done,
            {
                "stage": "gt-repair-geometry",
                "count": len(output_rows),
                "source_digest": content_fingerprint(source_shard),
                "state_digest": state_digest,
            },
        )
        geometry_reporter.update(
            geometry_completed,
            detail=f"{source_shard.name} 已原子落盘",
            force=True,
        )
    geometry_reporter.update(
        len(unique), status="completed", detail="几何结果完整", force=True
    )

    geometry_rows = [
        row for shard in source_shards for row in read_jsonl(geometry_dir / shard.name)
    ]
    if len(geometry_rows) != len(unique) or {row["image_id"] for row in geometry_rows} != set(unique_by_id):
        raise ValueError("v4 geometry does not exactly cover the unique image manifest")
    geometry_by_image = {str(row["image_id"]): row for row in geometry_rows}
    repair_detections = [row for record in geometry_rows for row in record["repair_detections"]]
    repair_actions = [row for record in geometry_rows for row in record["repair_actions"]]
    validate_repair_inventory(
        repair_detections,
        repair_actions,
        expected_count=args.expected_valid_repairs,
        expected_by_task=(
            EXPECTED_VALID_REPAIRS_BY_TASK
            if args.expected_unique_images == 17281
            else None
        ),
    )
    atomic_write_jsonl(target_audit / "gt_repair_detections.jsonl", repair_detections)
    atomic_write_jsonl(target_audit / "gt_repair_actions.jsonl", repair_actions)

    material_dir = config_root / "materialized"
    material_dir.mkdir(parents=True, exist_ok=True)
    repaired_sample_ids = {str(row["sample_id"]) for row in repair_actions}
    reusable_material: dict[Path, bool] = {}
    material_reused = 0
    for source_shard in source_shards:
        geometry_shard = geometry_dir / source_shard.name
        output_shard = material_dir / source_shard.name
        done = material_dir / f"{source_shard.stem}.done.json"
        valid = bool(
            args.resume
            and _material_shard_valid(geometry_shard, output_shard, done, state_digest)
        )
        reusable_material[source_shard] = valid
        if valid:
            material_reused += int(json.loads(done.read_text(encoding="utf-8"))["count"])
    material_reporter = ProgressReporter(
        stage="gt-repair-materialize",
        total=len(unique),
        output_dir=target_audit,
        interval_seconds=args.progress_interval_seconds,
        initial_completed=material_reused,
        unit="images",
    )
    material_reporter.update(
        material_reused,
        detail=f"复用 {material_reused} 张已有 crop",
        force=True,
    )
    material_completed = material_reused
    for source_shard in source_shards:
        geometry_shard = geometry_dir / source_shard.name
        output_shard = material_dir / source_shard.name
        done = material_dir / f"{source_shard.stem}.done.json"
        if reusable_material[source_shard]:
            continue
        output_rows = []
        for geometry in read_jsonl(geometry_shard):
            manifest = unique_by_id[str(geometry["image_id"])]
            # Reuse the established materializer: one image decode, one physical
            # PNG per unique bbox, task references remain independent.
            output_rows.append(
                materialize_image_record(
                    manifest=manifest,
                    geometry=geometry,
                    config_root=config_root,
                    overview_sample_ids=repaired_sample_ids,
                )
            )
            material_completed += 1
            material_reporter.update(
                material_completed,
                detail=f"{source_shard.name}，当前 {geometry['image_id']}",
            )
        atomic_write_jsonl(output_shard, output_rows)
        atomic_write_json(
            done,
            {
                "stage": "gt-repair-materialize",
                "count": len(output_rows),
                "source_digest": content_fingerprint(geometry_shard),
                "state_digest": state_digest,
            },
        )
        material_reporter.update(
            material_completed,
            detail=f"{source_shard.name} 已原子落盘",
            force=True,
        )
    material_reporter.update(
        len(unique), status="completed", detail="crop 物化完整", force=True
    )

    material_rows = [
        row for shard in source_shards for row in read_jsonl(material_dir / shard.name)
    ]
    material_by_image = {str(row["image_id"]): row for row in material_rows}
    if len(material_by_image) != len(unique):
        raise ValueError("v4 materialization does not cover every unique image")

    source_results_by_sample: dict[str, dict[str, Any]] = {}
    for source_shard in source_shards:
        for source_record in read_jsonl(source_shard):
            for result in source_record["sample_results"]:
                source_results_by_sample[str(result["sample_id"])] = dict(result)
    repaired_results_by_sample = {
        str(result["sample_id"]): {**result, "image_id": str(record["image_id"])}
        for record in geometry_rows
        for result in record["sample_results"]
    }
    render_repair_visualizations(
        target_audit=target_audit,
        actions=repair_actions,
        unique_by_id=unique_by_id,
        detections=detections,
        source_results_by_sample=source_results_by_sample,
        repaired_results_by_sample=repaired_results_by_sample,
        progress_interval_seconds=args.progress_interval_seconds,
        resume=args.resume,
    )
    if not all(Path(str(row["visualization_4panel"])).is_file() for row in repair_actions):
        raise AssertionError("one or more GT repair four-panel visualizations are missing")
    # Geometry shards remain resumable and immutable.  The flattened action
    # report is republished with evidence paths after all 106 images exist.
    atomic_write_jsonl(target_audit / "gt_repair_actions.jsonl", repair_actions)

    details: list[dict[str, Any]] = []
    post_failures: list[dict[str, Any]] = []
    task_manifest: list[dict[str, Any]] = []
    preview_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_reporter = ProgressReporter(
        stage="gt-repair-reports",
        total=len(geometry_rows),
        output_dir=target_audit,
        interval_seconds=args.progress_interval_seconds,
        unit="images",
    )
    report_reporter.update(0, detail="构建 task-aware manifest 与统计", force=True)
    for geometry_index, geometry in enumerate(geometry_rows, 1):
        image_id = str(geometry["image_id"])
        sample_map = {str(row["sample_id"]): row for row in samples_by_image[image_id]}
        material = material_by_image[image_id]
        for result in geometry["sample_results"]:
            sample_id = str(result["sample_id"])
            sample = sample_map[sample_id]
            paths_for_sample = [Path(value) for value in material["sample_paths"][sample_id]]
            overview = material["overview_paths"].get(sample_id, "")
            preview, preview_failures = build_preview_rows(
                sample,
                result["crop_boxes"],
                paths_for_sample,
                config_name=args.repair_config,
            )
            preview_by_task[str(sample["task"])].extend(preview)
            detail = dict(result["detail"])
            detail["crop_paths"] = [str(path.resolve()) for path in paths_for_sample]
            detail["overview"] = overview
            details.append(detail)
            for failure in result["failures"]:
                updated = dict(failure)
                updated["visualization"] = overview
                post_failures.append(updated)
            provenance_by_box = {
                tuple(box): source
                for box, source in zip(result["crop_boxes"], result["crop_provenance"])
            }
            training_records = []
            for row in preview:
                if sample["task"] == "ui_content_missing":
                    # The full-image record is already retained by the base
                    # recipe; do not duplicate it as a synthetic crop record.
                    continue
                if not row["training_eligible"]:
                    continue
                crop_source = provenance_by_box.get(tuple(row["crop_bbox"]), "full_image")
                training_records.append(
                    {
                        "image": row["image"],
                        "conversations": row["conversations"],
                        "_ui5_sample_id": sample_id,
                        "_ui5_image_id": image_id,
                        "_ui5_task": sample["task"],
                        "_ui5_split": sample.get("split"),
                        "_ui5_record_kind": "crop",
                        "_ui5_crop_source": crop_source,
                        "_ui5_crop_bbox": row["crop_bbox"],
                        "_ui5_contained_gt_indices": row["contained_gt_indices"],
                        "_ui5_training_eligible": True,
                    }
                )
            task_manifest.append(
                {
                    **detail,
                    "config": args.repair_config,
                    "source_audit": args.source_audit_name,
                    "training_records": training_records,
                    "repair_detection_count": len(result["repair_detections"]),
                    "repair_action_count": len(result["repair_actions"]),
                    "crop_provenance": result["crop_provenance"],
                    "excluded": False,
                }
            )
        report_reporter.update(
            geometry_index,
            detail=f"构建报告数据，当前 {image_id}",
        )
    if post_failures:
        raise AssertionError(f"post-repair geometry still has {len(post_failures)} GT failures")
    atomic_write_jsonl(target_audit / "task_aware_manifest.jsonl", task_manifest)
    atomic_write_jsonl(target_audit / "gt_failures.jsonl", post_failures)
    atomic_write_jsonl(target_audit / "gt_failures_visualized.jsonl", [])
    for task in TASK_NAMES:
        atomic_write_jsonl(config_root / "preview" / f"{task}.jsonl", preview_by_task[task])

    candidate_summary = _candidate_summary(details)
    region = candidate_summary["by_scope"]["REGION_ALL"]
    raw_source_summary = json.loads((source_audit / "summary.json").read_text(encoding="utf-8"))
    raw_region = raw_source_summary["candidates"][args.source_config]["by_scope"]["REGION_ALL"]
    strict_current = args.expected_unique_images == 17281
    if strict_current:
        raw_values = (int(raw_region["gt_contained_count"]), int(raw_region["gt_count"]))
        if raw_values != (EXPECTED_REGION_GT_CONTAINED_RAW, EXPECTED_REGION_GT_TOTAL_RAW):
            raise ValueError(f"raw detector baseline changed unexpectedly: {raw_values}")
        post_values = (int(region["gt_contained_count"]), int(region["gt_count"]))
        if post_values != (EXPECTED_REGION_GT_TOTAL_CLEAN, EXPECTED_REGION_GT_TOTAL_CLEAN):
            raise ValueError(f"post-repair cleaned GT totals differ: {post_values}")

    input_after = audit_input_snapshot(paths, unique, detections_rows, samples)
    detection_digest_after = content_fingerprint(paths.merged)
    input_unchanged = input_after == input_before and detection_digest_after == detection_digest_before
    overlap = raw_source_summary.get("cross_task_supervision", {})
    cross_train_val_count = int(overlap.get("same_content_cross_train_val_count", 0))
    content_preview = preview_by_task["ui_content_missing"]
    mismatch_count = sum(
        row.get("original_gt_boxes_1000") != row.get("output_gt_boxes_1000")
        for row in content_preview
    )
    required_reports = [
        target_audit / "gt_repair_detections.jsonl",
        target_audit / "gt_repair_actions.jsonl",
        target_audit / "excluded_training_samples.jsonl",
        target_audit / "task_aware_manifest.jsonl",
        target_audit / "gt_failures.jsonl",
        target_audit / "gt_failures_visualized.jsonl",
        target_audit / "gt_repair_visualizations" / "gallery" / "index.html",
    ]
    reports_written = all(path.is_file() for path in required_reports)
    reports_written = reports_written and all(
        path.stat().st_size > 0
        for path in required_reports
        if path.name not in {"gt_failures.jsonl", "gt_failures_visualized.jsonl"}
    )
    reports_written = reports_written and all(
        Path(str(row["visualization_4panel"])).is_file() for row in repair_actions
    )
    gate = _build_v4_gate(
        details=details,
        region=region,
        by_scope=candidate_summary["by_scope"],
        cross_train_val_count=cross_train_val_count,
        content_missing_mismatch_count=mismatch_count,
        input_unchanged=input_unchanged,
        excluded_count=len(exclusions),
        repair_split_violations=sum(row.get("split") != "train" for row in repair_detections),
        reports_written=reports_written,
    )
    summary = {
        "audit_format_version": 4,
        "audit_name": args.crop_audit_name,
        "source_audit": args.source_audit_name,
        "source_commits": {
            "runtime_libgl": "de30357f73b6b393840efcb7eb3ca37182e86cc4",
            "sigbus_checkpoint": "ed5add660608b93ed4f9d5c68efb1c04478aa6bd",
        },
        "recommended_config": args.repair_config,
        "materialized_candidate": args.repair_config,
        "training_only_gt_repair": True,
        "training_started": False,
        "training_ready": False,
        "detector_stages_executed": [],
        "audit_state_digest": state_digest,
        "input_snapshot_before": input_before,
        "input_snapshot_after": input_after,
        "input_snapshot_digest": audit_state_digest(input_after),
        "input_snapshot_unchanged": input_unchanged,
        "detections_digest_before": detection_digest_before,
        "detections_digest_after": detection_digest_after,
        "detections_unchanged": detection_digest_before == detection_digest_after,
        "metric_definitions": METRIC_DEFINITIONS,
        "candidates": {args.repair_config: candidate_summary},
        "configs": {args.repair_config: candidate_summary},
        "cross_task_supervision": overlap,
        "materialization": {
            "candidate": args.repair_config,
            **materialization_reuse_metrics(material_rows, len(repaired_sample_ids)),
        },
        "repair_metrics": {
            "raw_detector_region_gt_contained": int(raw_region["gt_contained_count"]),
            "raw_detector_region_gt_total": int(raw_region["gt_count"]),
            "raw_detector_region_gt_recall": float(raw_region["gt_box_containment_recall"]),
            "training_materialization_gt_contained_after_repair": int(region["gt_contained_count"]),
            "training_materialization_gt_total_after_repair": int(region["gt_count"]),
            "training_materialization_gt_recall_after_repair": float(region["gt_box_containment_recall"]),
            "raw_failure_count": len(source_failures),
            "excluded_annotation_count": len(exclusions),
            "repaired_valid_failure_count": len(repair_detections),
            "post_repair_partial_count": int(region["partial_only_gt_count"]),
            "post_repair_uncovered_count": int(region["uncovered_gt_count"]),
            "interpretation": (
                "post-repair recall only proves that training materialization covers cleaned training GT; "
                "it is not OCR/icon detector recall and must not be reported as detector generalization."
            ),
        },
        "excluded_annotation_cases": exclusions,
        "content_missing_checks": {
            "full_image_bbox": "[0,0,W,H]",
            "gt_box_containment_recall": candidate_summary["by_scope"]["ui_content_missing"]["gt_box_containment_recall"],
            "normalized_gt_mismatch_count": mismatch_count,
            "label_transform_applied": False,
        },
        "next_stage_gate": gate,
        "recipe_state": {
            "written": False,
            "message": "Run build_ui5_crop_training_recipe.py before training authorization.",
        },
    }
    atomic_write_json(target_audit / "materialization_summary.json", summary["materialization"])
    write_statistics_csv(target_audit / "statistics.csv", details)
    atomic_write_json(target_audit / "summary.json", summary)
    report_reporter.update(
        len(geometry_rows), detail="正在写 Excel（最后一个大文件）", force=True
    )
    overlap_report = output_dir / "manifest" / "overlap" / "source_overlap.json"
    overlap_payload = json.loads(overlap_report.read_text(encoding="utf-8")) if overlap_report.is_file() else {}
    write_excel_report(
        target_audit / "ui5_crop_audit.xlsx",
        summary,
        overlap_payload,
        details,
        post_failures,
    )
    report_reporter.update(
        len(geometry_rows), status="completed", detail="JSON/CSV/Excel 全部写入", force=True
    )
    if marker_path.exists():
        raise AssertionError("GT-repair stage must not publish training_ready.json before recipe")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(
        {
            "audit": summary["audit_name"],
            "config": summary["recommended_config"],
            "repair_metrics": summary["repair_metrics"],
            "training_ready": summary["training_ready"],
            "next": "run build_ui5_crop_training_recipe.py",
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
