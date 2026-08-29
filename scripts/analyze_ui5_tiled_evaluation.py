#!/usr/bin/env python3
"""Compute tile-level error amplification diagnostics for UI5 evaluation."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from locany_ui5_common import TASK_ISSUE_NAMES, TASK_JSONL, TASKS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    return parser.parse_args(argv)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scorer(path: Path):
    spec = importlib.util.spec_from_file_location("ui5_tile_diagnostic_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_image(sample: dict[str, Any], base: Path) -> str:
    value = sample.get("images", sample.get("image"))
    if isinstance(value, list):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str):
        raise ValueError("GT sample has no image path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve(strict=False))


def _pixel_box(box: Sequence[float], width: int, height: int) -> list[float]:
    # Production UI5 GT uses LocateAnything's normalized 0..1000 space.
    if max(abs(float(value)) for value in box) <= 1.5:
        scale = 1.0
    elif max(abs(float(value)) for value in box) <= 1000.5:
        scale = 1000.0
    else:
        return [float(value) for value in box]
    return [
        float(box[0]) / scale * width,
        float(box[1]) / scale * height,
        float(box[2]) / scale * width,
        float(box[3]) / scale * height,
    ]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match(gt: list[list[float]], predictions: list[list[float]], threshold: float) -> tuple[int, int, int]:
    candidates = sorted(
        (
            (_iou(gt_box, pred_box), gt_index, pred_index)
            for gt_index, gt_box in enumerate(gt)
            for pred_index, pred_box in enumerate(predictions)
        ),
        reverse=True,
    )
    used_gt, used_pred = set(), set()
    for score, gt_index, pred_index in candidates:
        if score < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
    return len(used_gt), len(predictions) - len(used_pred), len(gt) - len(used_gt)


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _gallery(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows[:100]:
        image = Path(row["image_path"]).as_posix()
        cards.append(
            f'<article><h3>{row["task"]} | tiles={row["tile_count"]}</h3>'
            f'<img src="file:///{image}" loading="lazy"><pre>{json.dumps(row, ensure_ascii=False, indent=2)}</pre></article>'
        )
    _atomic_text(
        path,
        "<!doctype html><meta charset='utf-8'><title>" + title + "</title>"
        "<style>body{font-family:sans-serif}article{margin:20px;border:1px solid #bbb;padding:12px}"
        "img{max-width:900px;max-height:700px}pre{white-space:pre-wrap}</style>"
        f"<h1>{title}</h1>" + "".join(cards),
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    prediction_dir = args.prediction_dir.expanduser().resolve(strict=True)
    gt_dir = args.gt_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    scorer = _scorer(
        args.scorer_root.expanduser().resolve(strict=True)
        / "qwen3vl_merge_and_score_fixed_5tasks.py"
    )
    detail_rows = []
    task_totals = {}
    grouped = defaultdict(lambda: {"image": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "bbox": {"tp": 0, "fp": 0, "fn": 0}})
    for task in TASKS:
        source = gt_dir / TASK_JSONL[task]
        gt_by_image = {}
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                image_path = _sample_image(sample, source.parent)
                gt_by_image[image_path] = scorer.extract_bboxes_for_issue(
                    scorer.get_gt_payload(sample), TASK_ISSUE_NAMES[task]
                )
        raw_dir = prediction_dir / task / "raw"
        task_counts = {
            "tile": {"tp": 0, "fp": 0, "fn": 0},
            "final_bbox": {"tp": 0, "fp": 0, "fn": 0},
            "negative_images": 0,
            "false_positive_tiles_on_negative_images": 0,
            "boxes_before_global_nms": 0,
            "boxes_after_global_nms": 0,
        }
        for raw_path in sorted(raw_dir.glob("*.json")):
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            image_path = str(Path(raw["image_path"]).resolve(strict=False))
            width = int(raw["image_size"]["width"])
            height = int(raw["image_size"]["height"])
            gt_boxes = [_pixel_box(box, width, height) for box in gt_by_image.get(image_path, [])]
            tile_rows = raw.get("inference_crop", {}).get("tiles", [])
            tile_count = len(tile_rows) or 1
            tile_tp = tile_fp = tile_fn = 0
            before = 0
            false_positive_tiles = 0
            for tile in tile_rows:
                tile_bbox = tile["tile_bbox"]
                local_predictions = tile.get("local_pixel_boxes", [])
                global_predictions = [
                    [
                        float(box[0]) + tile_bbox[0],
                        float(box[1]) + tile_bbox[1],
                        float(box[2]) + tile_bbox[0],
                        float(box[3]) + tile_bbox[1],
                    ]
                    for box in local_predictions
                ]
                before += len(global_predictions)
                assigned_gt = [
                    box for box in gt_boxes
                    if float(tile_bbox[1]) <= (box[1] + box[3]) / 2 < float(tile_bbox[3])
                ]
                tp, fp, fn = _match(assigned_gt, global_predictions, args.iou_threshold)
                tile_tp += tp
                tile_fp += fp
                tile_fn += fn
                false_positive_tiles += int(not gt_boxes and bool(global_predictions))
            final_boxes = [list(map(float, box)) for box in raw["parse"]["pixel_boxes_xyxy"]]
            final_tp, final_fp, final_fn = _match(gt_boxes, final_boxes, args.iou_threshold)
            task_counts["tile"]["tp"] += tile_tp
            task_counts["tile"]["fp"] += tile_fp
            task_counts["tile"]["fn"] += tile_fn
            task_counts["final_bbox"]["tp"] += final_tp
            task_counts["final_bbox"]["fp"] += final_fp
            task_counts["final_bbox"]["fn"] += final_fn
            task_counts["boxes_before_global_nms"] += before
            task_counts["boxes_after_global_nms"] += len(final_boxes)
            if not gt_boxes:
                task_counts["negative_images"] += 1
                task_counts["false_positive_tiles_on_negative_images"] += false_positive_tiles
            bucket = "1" if tile_count == 1 else "2" if tile_count == 2 else "3" if tile_count == 3 else "4+"
            image_group = grouped[(task, bucket)]["image"]
            predicted_positive, label_positive = bool(final_boxes), bool(gt_boxes)
            image_group["tp"] += int(predicted_positive and label_positive)
            image_group["fp"] += int(predicted_positive and not label_positive)
            image_group["fn"] += int(not predicted_positive and label_positive)
            image_group["tn"] += int(not predicted_positive and not label_positive)
            bbox_group = grouped[(task, bucket)]["bbox"]
            bbox_group["tp"] += final_tp
            bbox_group["fp"] += final_fp
            bbox_group["fn"] += final_fn
            detail_rows.append(
                {
                    "task": task,
                    "image_path": image_path,
                    "tile_count": tile_count,
                    "gt_count": len(gt_boxes),
                    "tile_tp": tile_tp,
                    "tile_fp": tile_fp,
                    "tile_fn": tile_fn,
                    "false_positive_tile_count": false_positive_tiles,
                    "pre_nms_bbox_count": before,
                    "post_nms_bbox_count": len(final_boxes),
                    "final_bbox_tp": final_tp,
                    "final_bbox_fp": final_fp,
                    "final_bbox_fn": final_fn,
                    "raw_sidecar": str(raw_path),
                }
            )
        task_counts["tile"] = _metrics(task_counts["tile"])
        task_counts["final_bbox"] = _metrics(task_counts["final_bbox"])
        task_counts["source_image_fp_amplification"] = (
            task_counts["false_positive_tiles_on_negative_images"]
            / max(1, sum(row["final_bbox_fp"] > 0 for row in detail_rows if row["task"] == task and row["gt_count"] == 0))
        )
        task_totals[task] = task_counts
    grouped_output = {
        f"{task}|tiles={bucket}": {
            "image": _metrics(values["image"]),
            "bbox": _metrics(values["bbox"]),
        }
        for (task, bucket), values in sorted(grouped.items())
    }
    summary = {
        "schema_version": 1,
        "iou_threshold": args.iou_threshold,
        "tasks": task_totals,
        "by_tile_count": grouped_output,
        "detail_rows": len(detail_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "tile_error_analysis.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    fieldnames = list(detail_rows[0]) if detail_rows else []
    if fieldnames:
        csv_path = output_dir / "tile_error_detail.csv"
        temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    _gallery(
        output_dir / "text_ellipsis_fp_gallery.html",
        "text ellipsis false positives",
        [row for row in detail_rows if row["task"] == "text_ellipsis" and row["gt_count"] == 0 and row["final_bbox_fp"] > 0],
    )
    _gallery(
        output_dir / "cropping_fn_gallery.html",
        "element cropping false negatives",
        [row for row in detail_rows if row["task"] == "cropping" and row["final_bbox_fn"] > 0],
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
