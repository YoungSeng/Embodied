#!/usr/bin/env python3
"""Maintain evaluation_history.json/csv for LocateAnything UI5 runs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from locany_ui5_common import TASK_ISSUE_NAMES, TASKS
from eaglevl.train.ui5_excel_logger import (
    UI5ExcelLogger,
    build_eval_rows,
)


BASE_COLUMNS = [
    "step",
    "machine_type",
    "gpu_count",
    "max_num_tokens",
    "max_num_tokens_scope",
    "relation_gate_mode",
    "ui_model_signature",
    "checkpoint",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "image_macro_precision",
    "image_macro_recall",
    "image_macro_f1",
    "bbox_macro_precision",
    "bbox_macro_recall",
    "bbox_macro_f1",
    "evaluation_start_time",
    "evaluation_end_time",
    "evaluation_status",
    "prediction_dir",
    "evaluation_run_dir",
    "error",
]
TASK_COLUMNS = [
    f"{task}_{granularity}_{metric}"
    for task in TASKS
    for granularity in ("image", "bbox")
    for metric in ("precision", "recall", "f1")
]
CSV_COLUMNS = BASE_COLUMNS + TASK_COLUMNS


def norm1000_box_to_pixel(
    box: list[float] | tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> list[float]:
    if len(box) != 4 or image_width <= 0 or image_height <= 0:
        raise ValueError("norm1000 box conversion requires four coordinates and positive image size")
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        x1 * image_width / 1000.0,
        y1 * image_height / 1000.0,
        x2 * image_width / 1000.0,
        y2 * image_height / 1000.0,
    ]


def coarse_boxes_px_from_sidecar(record: dict[str, Any]) -> list[list[float]]:
    """Return coarse boxes in pixel space, rejecting ambiguous legacy records."""
    normalized = record.get("coarse_boxes_norm1000")
    pixel = record.get("coarse_boxes_px")
    if not normalized and not pixel:
        if record.get("coarse_boxes"):
            raise RuntimeError(
                "legacy coarse_boxes are ambiguous; migrate them with "
                "scripts/recompute_ui5_coarse_sidecars.py first"
            )
        return []
    if record.get("coordinate_space") != "norm1000":
        raise RuntimeError("coarse boxes require coordinate_space='norm1000'")
    if not isinstance(normalized, list) or not isinstance(pixel, list):
        raise RuntimeError(
            "coarse boxes require both coarse_boxes_norm1000 and "
            "coarse_boxes_px lists"
        )
    size = record.get("image_size") or {}
    width = record.get("image_width", size.get("width") if isinstance(size, dict) else None)
    height = record.get("image_height", size.get("height") if isinstance(size, dict) else None)
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise RuntimeError("coarse boxes require image_width/image_height")
    expected = [norm1000_box_to_pixel(box, float(width), float(height)) for box in normalized]
    if len(expected) != len(pixel):
        raise RuntimeError("coarse norm1000/pixel box counts differ")
    for left, right in zip(expected, pixel):
        if len(right) != 4 or any(abs(a - float(b)) > 1.0e-4 for a, b in zip(left, right)):
            raise RuntimeError("coarse_boxes_px does not match norm1000 conversion")
    return [[float(value) for value in box] for box in pixel]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("evaluations", [])
    if not isinstance(value, list):
        raise ValueError(f"Invalid evaluation history format: {path}")
    return [row for row in value if isinstance(row, dict)]


def load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metric summary: {path}")
    return value


def ui_model_signature(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    has_image_gate = False
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        has_image_gate = any(
            "relation_pyramid.image_gate_heads" in key
            for key in index.get("weight_map", {})
        )
    else:
        try:
            from safetensors import safe_open

            for path in checkpoint.glob("model*.safetensors"):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if any(
                        "relation_pyramid.image_gate_heads" in key
                        for key in handle.keys()
                    ):
                        has_image_gate = True
                        break
        except ImportError:
            pass
    signature = {
        "enable_ui_relation": bool(config.get("enable_ui_relation", False)),
        "relation_detail_layers": config.get("relation_detail_layers"),
        "relation_detail_hidden_size": config.get("relation_detail_hidden_size"),
        "relation_num_slots": config.get("relation_num_slots"),
        "relation_slot_gate_loss_weight": config.get(
            "relation_slot_gate_loss_weight"
        ),
        "relation_slot_objectness_loss_weight": config.get(
            "relation_slot_objectness_loss_weight",
            config.get("relation_slot_gate_loss_weight"),
        ),
        "tc_msed_stage": config.get("tc_msed_stage", "v4"),
        "relation_task_scale_router": config.get("relation_task_scale_router", False),
        "relation_set_localizer": config.get("relation_set_localizer", False),
        "relation_dynamic_slot_pbd": config.get("relation_dynamic_slot_pbd", False),
        "relation_coordinate_bridge": config.get("relation_coordinate_bridge", False),
        "relation_soft_gate": config.get("relation_soft_gate", False),
        "relation_overlap_adapter": config.get("relation_overlap_adapter", False),
        "relation_task_hard_router": config.get("relation_task_hard_router", False),
        "relation_task_experts": config.get("relation_task_experts", False),
        "relation_task_expert_rank": config.get("relation_task_expert_rank", 8),
        "relation_set_decoder": config.get("relation_set_decoder", False),
        "relation_set_decoder_layers": config.get("relation_set_decoder_layers", 3),
        "relation_box_l1_loss_weight": config.get("relation_box_l1_loss_weight", 0.0),
        "relation_box_giou_loss_weight": config.get("relation_box_giou_loss_weight", 0.0),
        "relation_coverage_loss_weight": config.get("relation_coverage_loss_weight", 0.0),
        "relation_coord_prior_sigma": config.get("relation_coord_prior_sigma", 0.05),
        "box_start_token_id": config.get("box_start_token_id"),
        "text_mask_token_id": (config.get("text_config") or {}).get(
            "text_mask_token_id"
        ),
        "mtp_block_size": (config.get("text_config") or {}).get("block_size"),
        "causal_attn": (config.get("text_config") or {}).get("causal_attn"),
        "has_image_gate": has_image_gate,
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _sample_image_path(sample: dict[str, Any], jsonl_dir: Path) -> str | None:
    images = sample.get("images", sample.get("image"))
    if isinstance(images, list):
        images = images[0] if images else None
    if isinstance(images, dict):
        images = images.get("path")
    if not isinstance(images, str):
        return None
    path = Path(images).expanduser()
    if not path.is_absolute():
        path = jsonl_dir / path
    return str(path.resolve(strict=False))


def collect_gate_metrics(
    prediction_dir: Path | None,
    gt_dir: Path | None,
    scorer_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Join per-image gate sidecars with the same GT parser used by scoring."""

    if prediction_dir is None or not prediction_dir.is_dir():
        return {}
    ground_truth: dict[str, dict[str, bool]] = {}
    ground_truth_boxes: dict[str, dict[str, list[Any]]] = {}
    if gt_dir is not None and gt_dir.is_dir():
        scorer_file = (
            scorer_root / "qwen3vl_merge_and_score_fixed_5tasks.py"
            if scorer_root is not None
            else Path(__file__).resolve().parents[1]
            / "qwen3vl_merge_and_score_fixed_5tasks.py"
        )
        extract_bboxes_for_issue = None
        get_gt_payload = None
        if scorer_file.is_file():
            spec = importlib.util.spec_from_file_location(
                "ui5_qwen_scorer_for_gate_metrics", scorer_file
            )
            if spec is not None and spec.loader is not None:
                scorer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(scorer_module)
                extract_bboxes_for_issue = getattr(
                    scorer_module, "extract_bboxes_for_issue", None
                )
                get_gt_payload = getattr(scorer_module, "get_gt_payload", None)
        if extract_bboxes_for_issue is not None and get_gt_payload is not None:
            from locany_ui5_common import TASK_JSONL

            for task in TASKS:
                source = gt_dir / TASK_JSONL[task]
                labels: dict[str, bool] = {}
                boxes_by_path: dict[str, list[Any]] = {}
                if source.is_file():
                    with source.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            sample = json.loads(line)
                            image_path = _sample_image_path(sample, source.parent)
                            if image_path is None:
                                continue
                            parsed_boxes = list(
                                extract_bboxes_for_issue(
                                    get_gt_payload(sample), TASK_ISSUE_NAMES[task]
                                )
                                or []
                            )
                            positive = bool(parsed_boxes)
                            labels[image_path] = positive
                            labels[Path(image_path).name] = positive
                            boxes_by_path[image_path] = parsed_boxes
                            boxes_by_path[Path(image_path).name] = parsed_boxes
                ground_truth[task] = labels
                ground_truth_boxes[task] = boxes_by_path

    result: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        gate_dir = prediction_dir / task / "gate"
        records = []
        if gate_dir.is_dir():
            for path in gate_dir.glob("*.json"):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    records.append(value)
        positives: list[float] = []
        negatives: list[float] = []
        sweep_samples: list[dict[str, Any]] = []
        gate_tp = gate_fp = gate_fn = 0
        gt_box_sum = pred_box_sum = 0
        count_absolute_error = 0.0
        coarse_iou_values: list[float] = []
        selected_iou_values: list[float] = []
        route_match_values: list[float] = []
        duplicate_slot_values: list[float] = []
        labels = ground_truth.get(task, {})
        task_boxes = ground_truth_boxes.get(task, {})

        def xyxy(value: Any) -> list[float] | None:
            if isinstance(value, dict):
                value = value.get("bbox", value.get("box", value.get("xyxy")))
            if not isinstance(value, (list, tuple)) or len(value) < 4:
                return None
            try:
                return [float(item) for item in value[:4]]
            except (TypeError, ValueError):
                return None

        def iou(left: list[float], right: list[float]) -> float:
            lx1, ly1, lx2, ly2 = left
            rx1, ry1, rx2, ry2 = right
            intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
                0.0, min(ly2, ry2) - max(ly1, ry1)
            )
            left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
            right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
            return intersection / max(left_area + right_area - intersection, 1.0e-9)
        for record in records:
            p_defect = record.get("p_defect")
            image_path = str(record.get("image_path", ""))
            label = labels.get(image_path, labels.get(Path(image_path).name))
            if isinstance(p_defect, (int, float)) and label is not None:
                (positives if label else negatives).append(float(p_defect))
                predicted = bool(record.get("would_pass", record.get("gate_passed")))
                gate_tp += int(label and predicted)
                gate_fp += int(not label and predicted)
                gate_fn += int(label and not predicted)
                sweep_samples.append(
                    {
                        "label": bool(label),
                        "raw_positive": record.get("prediction_status") == "defect",
                        "p_defect": float(p_defect),
                    }
                )
            gt_boxes = [
                parsed for value in task_boxes.get(image_path, task_boxes.get(Path(image_path).name, []))
                if (parsed := xyxy(value)) is not None
            ]
            predicted_count = int(record.get("prediction_boxes", 0) or 0)
            gt_box_sum += len(gt_boxes)
            pred_box_sum += predicted_count
            count_absolute_error += abs(predicted_count - len(gt_boxes))
            raw_coarse = coarse_boxes_px_from_sidecar(record)
            coarse = [parsed for value in raw_coarse if (parsed := xyxy(value)) is not None]
            selected_indices = [
                int(value) for value in (record.get("selected_slot_indices") or [])
                if isinstance(value, (int, float)) and 0 <= int(value) < len(coarse)
            ]
            selected = [coarse[index] for index in dict.fromkeys(selected_indices)]
            for target in gt_boxes:
                oracle_iou = max((iou(target, candidate) for candidate in coarse), default=0.0)
                selected_iou = max((iou(target, candidate) for candidate in selected), default=0.0)
                coarse_iou_values.append(oracle_iou)
                selected_iou_values.append(selected_iou)
                route_match_values.append(float(selected_iou + 1.0e-9 >= oracle_iou))
            duplicate = record.get("duplicate_slot_rate")
            if isinstance(duplicate, (int, float)):
                duplicate_slot_values.append(float(duplicate))
        gate_precision = gate_tp / (gate_tp + gate_fp) if gate_tp + gate_fp else 0.0
        gate_recall = gate_tp / (gate_tp + gate_fn) if gate_tp + gate_fn else 0.0
        gate_f1 = (
            2 * gate_precision * gate_recall / (gate_precision + gate_recall)
            if gate_precision + gate_recall
            else 0.0
        )
        if records and len(sweep_samples) != len(records):
            raise RuntimeError(
                f"Gate sidecars are incomplete for task={task}: "
                f"labeled_p_defect={len(sweep_samples)}, records={len(records)}"
            )
        result[task] = {
            "samples": len(records),
            "gate_positive": sum(
                bool(row.get("would_pass", row.get("gate_passed")))
                for row in records
            ),
            "gate_filtered": sum(bool(row.get("gate_filtered")) for row in records),
            "p_defect_pos": (
                sum(positives) / len(positives) if positives else None
            ),
            "positive_count": len(positives),
            "p_defect_neg": (
                sum(negatives) / len(negatives) if negatives else None
            ),
            "negative_count": len(negatives),
            "parse_error": sum(
                row.get("prediction_status") == "parse_error" for row in records
            ),
            "gate_tp": gate_tp,
            "gate_fp": gate_fp,
            "gate_fn": gate_fn,
            "gate_precision": gate_precision,
            "gate_recall": gate_recall,
            "gate_f1": gate_f1,
            "_sweep_samples": sweep_samples,
            "gt_average_box_count": gt_box_sum / len(records) if records else None,
            "pred_average_box_count": pred_box_sum / len(records) if records else None,
            "count_mae": count_absolute_error / len(records) if records else None,
            "coarse_recall_03": (
                sum(value >= 0.3 for value in coarse_iou_values) / len(coarse_iou_values)
                if coarse_iou_values else None
            ),
            "coarse_recall_05": (
                sum(value >= 0.5 for value in coarse_iou_values) / len(coarse_iou_values)
                if coarse_iou_values else None
            ),
            "selected_slot_iou": (
                sum(selected_iou_values) / len(selected_iou_values)
                if selected_iou_values else None
            ),
            "oracle_8slot_iou": (
                sum(coarse_iou_values) / len(coarse_iou_values)
                if coarse_iou_values else None
            ),
            "route_top1_match_accuracy": (
                sum(route_match_values) / len(route_match_values)
                if route_match_values else None
            ),
            "duplicate_slot_rate": (
                sum(duplicate_slot_values) / len(duplicate_slot_values)
                if duplicate_slot_values else None
            ),
            "pbd_enabled": (
                bool(records[0].get("pbd_enabled")) if records else None
            ),
            "coordinate_bridge_enabled": (
                bool(records[0].get("coordinate_bridge_enabled")) if records else None
            ),
            "slot_routing_enabled": (
                bool(records[0].get("slot_routing_enabled")) if records else None
            ),
        }
    return result


def build_gate_threshold_sweep(
    gate_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply image Gate thresholds to one immutable set of raw predictions."""

    output: dict[str, Any] = {
        "schema_version": 1,
        "thresholds": [index / 100.0 for index in range(61)],
        "tasks": {},
    }
    for task in TASKS:
        samples = list(gate_metrics.get(task, {}).get("_sweep_samples", []))
        rows = []
        for index in range(61):
            threshold = index / 100.0
            tp = fp = fn = tn = predicted_positive = 0
            for sample in samples:
                # t=0 is exactly the raw/no-hard-gate result.
                predicted = bool(sample["raw_positive"]) and (
                    threshold == 0.0 or float(sample["p_defect"]) >= threshold
                )
                label = bool(sample["label"])
                tp += int(label and predicted)
                fp += int(not label and predicted)
                fn += int(label and not predicted)
                tn += int(not label and not predicted)
                predicted_positive += int(predicted)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "predicted_positive": predicted_positive,
                }
            )
        best = max(rows, key=lambda row: (float(row["f1"]), -float(row["threshold"])))
        output["tasks"][task] = {
            "raw": rows[0],
            "selected": best,
            "sweep": rows,
        }
    return output


def write_gate_threshold_sweep(prediction_dir: Path, sweep: dict[str, Any]) -> None:
    atomic_write_text(
        prediction_dir / "gate_threshold_sweep.json",
        json.dumps(sweep, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = ["task\tthreshold\tprecision\trecall\tf1\ttp\tfp\tfn\tpredicted_positive"]
    for task in TASKS:
        selected = sweep.get("tasks", {}).get(task, {}).get("selected", {})
        lines.append(
            "\t".join(
                str(value)
                for value in (
                    task,
                    selected.get("threshold"),
                    selected.get("precision"),
                    selected.get("recall"),
                    selected.get("f1"),
                    selected.get("tp"),
                    selected.get("fp"),
                    selected.get("fn"),
                    selected.get("predicted_positive"),
                )
            )
        )
    atomic_write_text(prediction_dir / "gate_threshold_sweep.txt", "\n".join(lines) + "\n")


def parse_markdown_report(path: Path) -> dict[str, Any]:
    """Convert the legacy all_tasks_evaluation.txt report into metric JSON."""

    issue_to_task = {value: key for key, value in TASK_ISSUE_NAMES.items()}
    result: dict[str, Any] = {
        "schema_version": 1,
        "tasks": {},
        "macro": {"bbox": {}, "image": {}},
    }
    section: str | None = None
    headers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(">>> Bbox"):
            section = "bbox"
            headers = []
            continue
        if line.startswith(">>> Image"):
            section = "image"
            headers = []
            continue
        if section is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if not headers:
            headers = cells
            continue
        row = dict(zip(headers, cells))
        name = row.get("task", "")
        if not name:
            continue
        metric_values = {
            "precision": float(row["prec"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }
        if name == "五类平均":
            result["macro"][section] = metric_values
            continue
        task = issue_to_task.get(name)
        if task is None:
            continue
        result["tasks"].setdefault(task, {"issue_name": name})[section] = metric_values
    if set(result["tasks"]) != set(TASKS):
        raise ValueError(
            f"Legacy report does not contain all five tasks: found={sorted(result['tasks'])}"
        )
    if not result["macro"]["image"] or not result["macro"]["bbox"]:
        raise ValueError("Legacy report is missing macro image/bbox rows")
    return result


def build_row(args: argparse.Namespace) -> dict[str, Any]:
    metrics = load_metrics(args.metrics_json)
    macro = metrics.get("macro", {})
    image_macro = macro.get("image", {})
    bbox_macro = macro.get("bbox", {})
    row: dict[str, Any] = {
        "step": args.step,
        "machine_type": args.machine_type,
        "gpu_count": args.gpu_count,
        "max_num_tokens": args.max_num_tokens,
        "max_num_tokens_scope": args.max_num_tokens_scope,
        "relation_gate_mode": getattr(args, "relation_gate_mode", "observe"),
        "ui_model_signature": ui_model_signature(args.checkpoint),
        "checkpoint": str(args.checkpoint),
        "macro_precision": image_macro.get("precision"),
        "macro_recall": image_macro.get("recall"),
        "macro_f1": image_macro.get("f1"),
        "image_macro_precision": image_macro.get("precision"),
        "image_macro_recall": image_macro.get("recall"),
        "image_macro_f1": image_macro.get("f1"),
        "bbox_macro_precision": bbox_macro.get("precision"),
        "bbox_macro_recall": bbox_macro.get("recall"),
        "bbox_macro_f1": bbox_macro.get("f1"),
        "evaluation_start_time": args.start_time,
        "evaluation_end_time": args.end_time,
        "evaluation_status": args.status,
        "prediction_dir": str(args.prediction_dir or ""),
        "evaluation_run_dir": str(args.evaluation_run_dir or ""),
        "error": args.error or "",
        "tasks": metrics.get("tasks", {}),
    }
    for task in TASKS:
        task_metrics = metrics.get("tasks", {}).get(task, {})
        for granularity in ("image", "bbox"):
            group = task_metrics.get(granularity, {})
            for metric in ("precision", "recall", "f1"):
                row[f"{task}_{granularity}_{metric}"] = group.get(metric)
    return row


def append_excel_evaluation(
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> Path:
    diagnostics_path = (
        args.diagnostics_xlsx.expanduser().resolve()
        if args.diagnostics_xlsx is not None
        else args.history_dir.expanduser().resolve().parent
        / "diagnostics"
        / "ui5_training_evaluation.xlsx"
    )
    raw_metrics = (
        load_metrics(args.raw_metrics_json)
        if getattr(args, "raw_metrics_json", None) is not None
        else (metrics if args.relation_gate_mode != "soft" else {})
    )
    gate_prediction_dir = (
        args.raw_prediction_dir
        if getattr(args, "raw_prediction_dir", None) is not None
        else args.prediction_dir
    )
    gate_metrics = collect_gate_metrics(
        gate_prediction_dir, args.gt_dir, args.scorer_root
    )
    sweep = build_gate_threshold_sweep(gate_metrics)
    if args.prediction_dir is not None:
        write_gate_threshold_sweep(args.prediction_dir, sweep)
    if args.evaluation_run_dir is not None:
        write_gate_threshold_sweep(args.evaluation_run_dir, sweep)
    for task in TASKS:
        task_sweep = sweep.get("tasks", {}).get(task, {})
        raw = task_sweep.get("raw", {})
        selected = task_sweep.get("selected", {})
        gate_metrics.setdefault(task, {}).update(
            {
                "raw_precision": raw.get("precision"),
                "raw_recall": raw.get("recall"),
                "raw_f1": raw.get("f1"),
                "raw_predicted_positive": raw.get("predicted_positive"),
                "raw_fp": raw.get("fp"),
                "selected_gate_threshold": selected.get("threshold"),
                "gated_precision": selected.get("precision"),
                "gated_recall": selected.get("recall"),
                "gated_f1": selected.get("f1"),
                "gated_predicted_positive": selected.get("predicted_positive"),
                "gated_fp": selected.get("fp"),
                "gate_filter_rate": (
                    1.0
                    - float(selected.get("predicted_positive", 0))
                    / max(1, int(raw.get("predicted_positive", 0)))
                ),
            }
        )
        gate_metrics[task].pop("_sweep_samples", None)
        gate_metrics[task]["relation_gate_mode"] = args.relation_gate_mode
        sample_count = int(gate_metrics[task].get("samples", 0))
        if sample_count >= 100 and int(
            gate_metrics[task].get("raw_predicted_positive") or 0
        ) == 0:
            raise RuntimeError(
                f"observe-mode raw predictions are all negative for task={task}; "
                f"samples={sample_count}"
            )
        if sample_count and int(gate_metrics[task].get("gate_filtered", 0)) == sample_count:
            raise RuntimeError(
                f"hard Gate filtered every sample for task={task}; samples={sample_count}"
            )
    for task, values in gate_metrics.items():
        image_metrics = metrics.get("tasks", {}).get(task, {}).get("image", {})
        values["post_gate_fp"] = values.get("gated_fp")
        values["raw_fp"] = values.get("raw_fp", image_metrics.get("fp"))
        tp = image_metrics.get("tp")
        fp = image_metrics.get("fp")
        values["final_predicted_positive"] = (
            int(tp) + int(fp) if tp is not None and fp is not None else None
        )
    if args.prediction_dir is not None:
        atomic_write_text(
            args.prediction_dir / "_gate_metrics.json",
            json.dumps(gate_metrics, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
    rows = build_eval_rows(
        step=args.step,
        checkpoint=str(args.checkpoint),
        metrics=metrics,
        gate_metrics=gate_metrics,
        raw_metrics=raw_metrics,
        metadata={
            "git_commit": os.environ.get("GIT_COMMIT", ""),
            "run_name": os.environ.get("RUN_NAME", ""),
            "tc_msed_stage": json.loads(
                (args.checkpoint / "config.json").read_text(encoding="utf-8")
            ).get("tc_msed_stage", "v4") if (args.checkpoint / "config.json").is_file() else "",
            "config_hash": os.environ.get("UI5_CONFIG_HASH", ""),
        },
    )
    UI5ExcelLogger(diagnostics_path).append_eval(args.step, rows)
    return diagnostics_path


def write_history(history_dir: Path, rows: list[dict[str, Any]]) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row.get("step", 0)))
    json_path = history_dir / "evaluation_history.json"
    csv_path = history_dir / "evaluation_history.csv"
    atomic_write_text(
        json_path,
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: "" if row.get(key) is None else row.get(key)
                        for key in CSV_COLUMNS
                    }
                )
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--history-dir", type=Path, required=True)
    record.add_argument("--step", type=int, required=True)
    record.add_argument("--machine-type", required=True)
    record.add_argument("--gpu-count", type=int, required=True)
    record.add_argument("--max-num-tokens", type=int, required=True)
    record.add_argument(
        "--max-num-tokens-scope", default="per_rank_packed_batch"
    )
    record.add_argument(
        "--relation-gate-mode", choices=("observe", "hard", "soft"), default="observe"
    )
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--metrics-json", type=Path, default=None)
    record.add_argument("--raw-metrics-json", type=Path, default=None)
    record.add_argument("--start-time", required=True)
    record.add_argument("--end-time", required=True)
    record.add_argument("--status", choices=("success", "failed"), required=True)
    record.add_argument("--prediction-dir", type=Path, default=None)
    record.add_argument("--raw-prediction-dir", type=Path, default=None)
    record.add_argument("--gt-dir", type=Path, default=None)
    record.add_argument("--scorer-root", type=Path, default=None)
    record.add_argument("--diagnostics-xlsx", type=Path, default=None)
    record.add_argument("--evaluation-run-dir", type=Path, default=None)
    record.add_argument("--error", default="")

    check = subparsers.add_parser("has-success")
    check.add_argument("--history-dir", type=Path, required=True)
    check.add_argument("--step", type=int, required=True)
    check.add_argument(
        "--relation-gate-mode", choices=("observe", "hard", "soft"), default="observe"
    )

    convert = subparsers.add_parser("convert-report")
    convert.add_argument("--report", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "convert-report":
        value = parse_markdown_report(args.report.expanduser().resolve())
        atomic_write_text(
            args.output.expanduser().resolve(),
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return 0
    history_path = args.history_dir.expanduser().resolve() / "evaluation_history.json"
    rows = load_history(history_path)
    if args.command == "has-success":
        found = any(
            int(row.get("step", -1)) == args.step
            and row.get("evaluation_status") == "success"
            and row.get("relation_gate_mode", "observe") == args.relation_gate_mode
            and (
                args.step != 0
                or (
                    Path(str(row.get("checkpoint", ""))).name == "checkpoint-0"
                    and '"has_image_gate":true' in str(row.get("ui_model_signature", ""))
                    and row.get("relation_gate_mode") in {"observe", "soft"}
                )
            )
            for row in rows
        )
        if found:
            diagnostics_path = (
                args.history_dir.expanduser().resolve().parent
                / "diagnostics"
                / "ui5_training_evaluation.xlsx"
            )
            found = UI5ExcelLogger(diagnostics_path).has_eval_step(args.step)
        return 0 if found else 1

    row = build_row(args)
    previous_success = next(
        (
            existing
            for existing in rows
            if int(existing.get("step", -1)) == args.step
            and existing.get("evaluation_status") == "success"
        ),
        None,
    )
    if args.status == "failed" and previous_success is not None:
        print(
            f"[HISTORY] step={args.step} already has a successful evaluation; "
            "the failed retry remains in attempts/ and does not replace the successful metric row"
        )
        return 0
    if args.status == "success" and args.step > 0:
        step_zero = next(
            (
                existing
                for existing in rows
                if int(existing.get("step", -1)) == 0
                and existing.get("evaluation_status") == "success"
            ),
            None,
        )
        if step_zero is not None:
            if step_zero.get("relation_gate_mode") not in (None, row["relation_gate_mode"]):
                raise RuntimeError(
                    "checkpoint-0 and checkpoint-N evaluation gate modes differ: "
                    f"{step_zero.get('relation_gate_mode')} != {row['relation_gate_mode']}"
                )
            if step_zero.get("ui_model_signature") not in (None, "", row["ui_model_signature"]):
                raise RuntimeError(
                    "checkpoint-0 and checkpoint-N UI model structures differ: "
                    f"{step_zero.get('ui_model_signature')} != {row['ui_model_signature']}"
                )
    rows = [existing for existing in rows if int(existing.get("step", -1)) != args.step]
    rows.append(row)
    diagnostics_xlsx = None
    if args.status == "success":
        metrics = load_metrics(args.metrics_json)
        diagnostics_xlsx = append_excel_evaluation(args, metrics)
    write_history(args.history_dir.expanduser().resolve(), rows)
    print(
        json.dumps(
            {
                "history_json": str(history_path),
                "history_csv": str(history_path.with_suffix(".csv")),
                "recorded_step": args.step,
                "status": args.status,
                "diagnostics_xlsx": (
                    str(diagnostics_xlsx) if diagnostics_xlsx is not None else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
