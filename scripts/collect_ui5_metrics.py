#!/usr/bin/env python3
"""Maintain evaluation_history.json/csv for LocateAnything UI5 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

# Direct invocations use ``scripts/`` as sys.path[0].  Put this checkout first
# so the history repair command cannot accidentally import an older installed
# ``eaglevl`` package that lacks the UI5 Excel logger.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locany_ui5_common import TASK_ISSUE_NAMES, TASKS, image_gate_probability
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
    "evaluation_split",
    "cache_scope",
    "development_test_reuse",
    "git_sha",
    "git_dirty",
    "recipe_digest",
    "cache_digest",
    "ui_model_signature",
    "git_commit",
    "config_hash",
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
    "init_checkpoint",
    "init_cpt_step",
    "sft_step",
    "is_best_image",
    "is_best_bbox",
    "is_4000_milestone",
    "checkpoint_kept",
    "checkpoint_keep_reasons",
]
TASK_COLUMNS = [
    f"{task}_{granularity}_{metric}"
    for task in TASKS
    for granularity in ("image", "bbox")
    for metric in ("precision", "recall", "f1")
]
CSV_COLUMNS = BASE_COLUMNS + TASK_COLUMNS
PERMANENT_MILESTONE_STEPS = frozenset({4000, 8000, 12000, 16000})


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


def box_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1.0e-9)


def route_slot_diagnostic(
    target: list[float],
    coarse: list[list[float]],
    selected: list[list[float]],
    pre_mask_selected: list[list[float]],
) -> dict[str, float | None]:
    """Score one GT target without treating a zero-IoU tie as success."""

    oracle_iou = max((box_iou(target, box) for box in coarse), default=0.0)
    selected_iou = max((box_iou(target, box) for box in selected), default=0.0)
    pre_mask_iou = max(
        (box_iou(target, box) for box in pre_mask_selected), default=0.0
    )
    return {
        "oracle_iou": oracle_iou,
        "selected_iou": selected_iou,
        "route_match": float(
            oracle_iou > 0.0 and selected_iou + 1.0e-9 >= oracle_iou
        ),
        "pre_mask_route_match": float(
            oracle_iou > 0.0 and pre_mask_iou + 1.0e-9 >= oracle_iou
        ),
        "oracle_slot_hit": float(oracle_iou > 0.1),
        "selected_oracle_iou_ratio": (
            selected_iou / oracle_iou if oracle_iou > 0.0 else None
        ),
    }


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


def canonical_state_dict_keys(
    config: dict[str, Any], state_dict_keys: set[str]
) -> set[str]:
    """Compare logical model keys, independent of tied-weight serialization.

    ``save_pretrained`` may omit ``lm_head.weight`` when it aliases the input
    embedding. A DeepSpeed gathered state dict materializes that alias as a
    separate tensor, even though both checkpoints instantiate the same module
    state dict. Restore the declared tied alias before hashing while keeping
    every non-alias key under exact comparison.
    """

    canonical = set(state_dict_keys)
    text_config = config.get("text_config") or {}
    tied_embeddings = bool(
        config.get(
            "tie_word_embeddings",
            text_config.get("tie_word_embeddings", False),
        )
    )
    input_embedding = "language_model.model.embed_tokens.weight"
    output_embedding = "language_model.lm_head.weight"
    if tied_embeddings and input_embedding in canonical:
        canonical.add(output_embedding)
    return canonical


def ui_model_signature(checkpoint: Path) -> str:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict_keys: set[str] = set()
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        state_dict_keys.update(index.get("weight_map", {}))
    else:
        try:
            from safetensors import safe_open

            for path in checkpoint.glob("model*.safetensors"):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    state_dict_keys.update(handle.keys())
        except ImportError:
            pass
    state_dict_keys = canonical_state_dict_keys(config, state_dict_keys)
    state_dict_key_sha256 = (
        hashlib.sha256("\n".join(sorted(state_dict_keys)).encode("utf-8")).hexdigest()
        if state_dict_keys
        else ""
    )
    has_image_gate = any(
        "relation_pyramid.image_gate_heads" in key for key in state_dict_keys
    )
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
        "relation_aux_budget_ratio": config.get("relation_aux_budget_ratio", 1.0),
        "relation_straight_through_slot_router": config.get(
            "relation_straight_through_slot_router", False
        ),
        "relation_set_decoder_deep_supervision": config.get(
            "relation_set_decoder_deep_supervision", False
        ),
        "relation_reference_position_encoding": config.get(
            "relation_reference_position_encoding", False
        ),
        "relation_per_level_scale_router": config.get(
            "relation_per_level_scale_router", False
        ),
        "relation_constrained_bbox_decoding": config.get(
            "relation_constrained_bbox_decoding", False
        ),
        "box_start_token_id": config.get("box_start_token_id"),
        "text_mask_token_id": (config.get("text_config") or {}).get(
            "text_mask_token_id"
        ),
        "mtp_block_size": (config.get("text_config") or {}).get("block_size"),
        "causal_attn": (config.get("text_config") or {}).get("causal_attn"),
        "has_image_gate": has_image_gate,
        "state_dict_key_count": len(state_dict_keys),
        "state_dict_key_sha256": state_dict_key_sha256,
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def refresh_row_model_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Refresh cached signatures so pre-fix histories resume without re-eval."""

    checkpoint_value = str(row.get("checkpoint", "")).strip()
    if not checkpoint_value:
        return row
    checkpoint = Path(checkpoint_value).expanduser().resolve(strict=False)
    if not checkpoint.is_dir():
        return row
    signature = ui_model_signature(checkpoint)
    if signature:
        row["ui_model_signature"] = signature
    return row


def evaluation_cache_row_matches(
    row: dict[str, Any],
    *,
    step: int,
    relation_gate_mode: str,
    git_commit: str,
    config_hash: str,
    model_signature: str,
) -> bool:
    """Require code, effective config and model structure to all match."""

    if not (
        int(row.get("step", -1)) == int(step)
        and row.get("evaluation_status") == "success"
        and row.get("relation_gate_mode", "observe") == relation_gate_mode
        and row.get("git_commit") == git_commit
        and row.get("config_hash") == config_hash
        and row.get("ui_model_signature") == model_signature
    ):
        return False
    if int(step) != 0:
        return True
    return (
        Path(str(row.get("checkpoint", ""))).name == "checkpoint-0"
        and '"has_image_gate":true' in str(row.get("ui_model_signature", ""))
        and row.get("relation_gate_mode") in {"observe", "soft"}
    )


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
    task_files: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Join per-image gate sidecars with the same GT parser used by scoring."""

    if prediction_dir is None or not prediction_dir.is_dir():
        return {}
    ground_truth: dict[str, dict[str, bool]] = {}
    ground_truth_boxes: dict[str, dict[str, list[Any]]] = {}
    if task_files is not None:
        # UI9 inputs are normalized original-image annotations. Explicit source
        # keys bypass UI5 filename, Figma and prompt-alias conventions.
        for task, source in task_files.items():
            labels, boxes_by_path = {}, {}
            with Path(source).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip(): continue
                    sample = json.loads(line)
                    image_path = str(Path(sample["source_image"]).resolve())
                    labels[image_path] = bool(sample["boxes_px"])
                    boxes_by_path[image_path] = sample["boxes_px"]
            ground_truth[task], ground_truth_boxes[task] = labels, boxes_by_path
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
    for task in (task_files if task_files is not None else TASKS):
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
        pre_mask_route_match_values: list[float] = []
        selected_oracle_ratios: list[float] = []
        oracle_slot_hits: list[float] = []
        duplicate_slot_values: list[float] = []
        predicted_center_diversity_values: list[float] = []
        attention_diversity_values: list[float] = []
        slot_usage_histogram: list[int] = []
        p_defect_sources: dict[str, int] = {}
        missing_p_defect = 0
        missing_label = 0
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

        for record in records:
            p_defect, p_defect_source = image_gate_probability(record)
            p_defect_sources[p_defect_source] = (
                p_defect_sources.get(p_defect_source, 0) + 1
            )
            image_path = str(record.get("image_path", ""))
            label = labels.get(image_path, labels.get(Path(image_path).name))
            missing_p_defect += int(p_defect is None)
            missing_label += int(label is None)
            if p_defect is not None and label is not None:
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
            pre_mask_indices = [
                int(value) for value in (record.get("pre_mask_selected_slot_indices") or [])
                if isinstance(value, (int, float)) and 0 <= int(value) < len(coarse)
            ]
            pre_mask_selected = [
                coarse[index] for index in dict.fromkeys(pre_mask_indices)
            ]
            record_histogram = record.get("slot_usage_histogram")
            if isinstance(record_histogram, list):
                if len(slot_usage_histogram) < len(record_histogram):
                    slot_usage_histogram.extend(
                        [0] * (len(record_histogram) - len(slot_usage_histogram))
                    )
                for index, value in enumerate(record_histogram):
                    if isinstance(value, (int, float)):
                        slot_usage_histogram[index] += int(value)
            else:
                required = max(selected_indices, default=-1) + 1
                if len(slot_usage_histogram) < required:
                    slot_usage_histogram.extend([0] * (required - len(slot_usage_histogram)))
                for index in selected_indices:
                    slot_usage_histogram[index] += 1
            for target in gt_boxes:
                route_diagnostic = route_slot_diagnostic(
                    target, coarse, selected, pre_mask_selected
                )
                oracle_iou = float(route_diagnostic["oracle_iou"] or 0.0)
                selected_iou = float(route_diagnostic["selected_iou"] or 0.0)
                coarse_iou_values.append(oracle_iou)
                selected_iou_values.append(selected_iou)
                route_match_values.append(float(route_diagnostic["route_match"] or 0.0))
                pre_mask_route_match_values.append(
                    float(route_diagnostic["pre_mask_route_match"] or 0.0)
                )
                oracle_slot_hits.append(
                    float(route_diagnostic["oracle_slot_hit"] or 0.0)
                )
                ratio = route_diagnostic["selected_oracle_iou_ratio"]
                if ratio is not None:
                    selected_oracle_ratios.append(float(ratio))
            duplicate = record.get("duplicate_slot_rate")
            if isinstance(duplicate, (int, float)):
                duplicate_slot_values.append(float(duplicate))
            center_diversity = record.get("predicted_center_diversity")
            if isinstance(center_diversity, (int, float)):
                predicted_center_diversity_values.append(float(center_diversity))
            attention_diversity = record.get("attention_diversity")
            if isinstance(attention_diversity, (int, float)):
                attention_diversity_values.append(float(attention_diversity))
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
                f"labeled_p_defect={len(sweep_samples)}, records={len(records)}, "
                f"missing_p_defect={missing_p_defect}, missing_label={missing_label}, "
                f"p_defect_sources={p_defect_sources}"
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
            "p_defect_sources": p_defect_sources,
            "legacy_tile_gate_recovered": p_defect_sources.get(
                "legacy_tile_gates_max", 0
            ),
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
            "pre_mask_route_top1_accuracy": (
                sum(pre_mask_route_match_values) / len(pre_mask_route_match_values)
                if pre_mask_route_match_values else None
            ),
            "oracle_slot_hit_rate": (
                sum(oracle_slot_hits) / len(oracle_slot_hits)
                if oracle_slot_hits else None
            ),
            "selected_oracle_iou_ratio": (
                sum(selected_oracle_ratios) / len(selected_oracle_ratios)
                if selected_oracle_ratios else None
            ),
            "per_slot_usage_histogram": slot_usage_histogram,
            "predicted_center_diversity": (
                sum(predicted_center_diversity_values) / len(predicted_center_diversity_values)
                if predicted_center_diversity_values else None
            ),
            "attention_diversity": (
                sum(attention_diversity_values) / len(attention_diversity_values)
                if attention_diversity_values else None
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


def score_gate_samples(samples: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = predicted_positive = 0
    for sample in samples:
        predicted = bool(sample["raw_positive"]) and (
            threshold <= 0.0 or float(sample["p_defect"]) >= threshold
        )
        label = bool(sample["label"])
        tp += int(label and predicted)
        fp += int(not label and predicted)
        fn += int(label and not predicted)
        tn += int(not label and not predicted)
        predicted_positive += int(predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": predicted_positive,
    }


def apply_frozen_gate_thresholds(
    gate_metrics: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    if set(thresholds) != set(TASKS):
        raise ValueError(
            f"frozen gate thresholds must contain exactly five tasks: {sorted(thresholds)}"
        )
    output = {"schema_version": 1, "selection": "frozen_external", "tasks": {}}
    for task in TASKS:
        samples = list(gate_metrics.get(task, {}).get("_sweep_samples", []))
        threshold = float(thresholds[task])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"invalid frozen gate threshold for {task}: {threshold}")
        output["tasks"][task] = {
            "raw": score_gate_samples(samples, 0.0),
            "selected": score_gate_samples(samples, threshold),
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
        "evaluation_split": getattr(args, "evaluation_split", "test"),
        "cache_scope": getattr(args, "cache_scope", "full_test"),
        "development_test_reuse": bool(
            getattr(args, "development_test_reuse", False)
        ),
        "git_sha": getattr(args, "git_sha", ""),
        "git_dirty": str(getattr(args, "git_dirty", "0")) == "1",
        "recipe_digest": getattr(args, "recipe_digest", ""),
        "cache_digest": getattr(args, "cache_digest", ""),
        "ui_model_signature": ui_model_signature(args.checkpoint),
        "git_commit": os.environ.get("GIT_COMMIT", ""),
        "config_hash": os.environ.get("UI5_CONFIG_HASH", ""),
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
        "init_checkpoint": os.environ.get("INIT_CHECKPOINT", ""),
        "init_cpt_step": int(os.environ.get("INIT_CPT_STEP", "0")),
        "sft_step": args.step,
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
    retention: dict[str, Any],
) -> Path:
    checkpoint_config = (
        json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
        if (args.checkpoint / "config.json").is_file()
        else {}
    )
    tc_msed_stage = str(checkpoint_config.get("tc_msed_stage", "v4")).lower()
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
    genuinely_rescored_metrics = (
        load_metrics(args.gated_metrics_json)
        if getattr(args, "gated_metrics_json", None) is not None
        else None
    )
    if args.frozen_gate_thresholds is not None:
        frozen_path = args.frozen_gate_thresholds.expanduser().resolve(strict=True)
        frozen_value = json.loads(frozen_path.read_text(encoding="utf-8"))
        thresholds = frozen_value.get("thresholds", frozen_value)
        sweep = apply_frozen_gate_thresholds(gate_metrics, thresholds)
    else:
        # Test metrics remain raw unless an explicit frozen threshold file is supplied.
        # In particular, do not manufacture "gated" Image/BBox values by copying
        # the raw metrics when no gated prediction set was actually rescored.
        sweep = {
            "schema_version": 1,
            "selection": "raw_only_no_frozen_gate_thresholds",
            "tasks": {
                task: {
                    "raw": score_gate_samples(
                        list(gate_metrics.get(task, {}).get("_sweep_samples", [])), 0.0
                    ),
                    "selected": {},
                }
                for task in TASKS
            },
        }
    for task in TASKS:
        task_sweep = sweep.get("tasks", {}).get(task, {})
        raw = task_sweep.get("raw", {})
        selected = task_sweep.get("selected", {})
        genuine_task_metrics = (
            genuinely_rescored_metrics.get("tasks", {}).get(task, {})
            if genuinely_rescored_metrics is not None
            else {}
        )
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
                "gated_tp": selected.get("tp"),
                "gated_fp": selected.get("fp"),
                "gated_fn": selected.get("fn"),
                "gated_tn": selected.get("tn"),
                "gate_filter_rate": (
                    1.0
                    - float(selected.get("predicted_positive", 0))
                    / max(1, int(raw.get("predicted_positive", 0)))
                    if selected.get("predicted_positive") is not None
                    else None
                ),
                "gated_metrics_by_granularity": genuine_task_metrics,
                "bbox_metrics_genuinely_rescored": bool(
                    genuine_task_metrics.get("bbox")
                ),
            }
        )
        gate_metrics[task].pop("_sweep_samples", None)
        gate_metrics[task]["relation_gate_mode"] = args.relation_gate_mode
        sample_count = int(gate_metrics[task].get("samples", 0))
        if sample_count >= 100 and int(
            gate_metrics[task].get("raw_predicted_positive") or 0
        ) == 0:
            message = (
                f"observe-mode raw predictions are all negative for task={task}; "
                f"samples={sample_count}"
            )
            if tc_msed_stage == "m32":
                print(f"[m32 diagnostic] {message}")
            else:
                raise RuntimeError(message)
        if sample_count and int(gate_metrics[task].get("gate_filtered", 0)) == sample_count:
            message = (
                f"hard Gate filtered every sample for task={task}; "
                f"samples={sample_count}"
            )
            if tc_msed_stage == "m32":
                print(f"[m32 diagnostic] {message}")
            else:
                raise RuntimeError(message)
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
            "tc_msed_stage": tc_msed_stage,
            "config_hash": os.environ.get("UI5_CONFIG_HASH", ""),
            "model_signature": ui_model_signature(args.checkpoint),
            "init_checkpoint": os.environ.get("INIT_CHECKPOINT", ""),
            "init_cpt_step": int(os.environ.get("INIT_CPT_STEP", "0")),
            "is_best_image": retention["is_best_image"],
            "is_best_bbox": retention["is_best_bbox"],
            "is_4000_milestone": retention["is_4000_milestone"],
            "checkpoint_kept": retention["checkpoint_kept"],
        },
        audit_context={
            "evaluation_split": getattr(args, "evaluation_split", "test"),
            "cache_scope": getattr(args, "cache_scope", "full_test"),
            "development_test_reuse": bool(
                getattr(args, "development_test_reuse", False)
            ),
            "git_sha": getattr(args, "git_sha", ""),
            "git_dirty": str(getattr(args, "git_dirty", "0")) == "1",
            "recipe_digest": getattr(args, "recipe_digest", ""),
            "cache_digest": getattr(args, "cache_digest", ""),
            "code_digest": getattr(args, "git_sha", "")
            or os.environ.get("GIT_COMMIT", ""),
            "crop_train_mode": os.environ.get("UI5_CROP_TRAIN_MODE", ""),
            "ui_sampling_mode": os.environ.get("UI5_UI_SAMPLING_MODE", ""),
            "eval_inference_crop_mode": os.environ.get(
                "EVAL_INFERENCE_CROP_MODE", ""
            ),
            "scan_name": os.environ.get("EVAL_SCAN_NAME", ""),
        },
    )
    excel = UI5ExcelLogger(diagnostics_path)
    excel.append_eval(args.step, rows)
    excel.update_checkpoint_status(
        args.step,
        is_best_image=retention["is_best_image"],
        is_best_bbox=retention["is_best_bbox"],
        is_4000_milestone=retention["is_4000_milestone"],
        checkpoint_kept=retention["checkpoint_kept"],
    )
    return diagnostics_path


def build_best_checkpoints_document(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Select strict scorer-metric improvements and all permanent checkpoints."""

    successful = sorted(
        (
            row
            for row in rows
            if row.get("evaluation_status") == "success"
        ),
        key=lambda row: int(row.get("step", 0)),
    )
    best_image: float | None = None
    best_bbox: float | None = None
    current_image: dict[str, Any] | None = None
    current_bbox: dict[str, Any] | None = None
    selections: dict[int, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    for row in successful:
        step = int(row["step"])
        image_f1 = row.get("image_macro_f1")
        bbox_f1 = row.get("bbox_macro_f1")
        is_best_image = image_f1 is not None and (
            best_image is None or float(image_f1) > best_image
        )
        is_best_bbox = bbox_f1 is not None and (
            best_bbox is None or float(bbox_f1) > best_bbox
        )
        milestone = step in PERMANENT_MILESTONE_STEPS
        reasons = []
        if step == 0:
            reasons.append("checkpoint-0")
        if milestone:
            reasons.append(f"milestone-{step}")
        if is_best_image:
            reasons.append("strict-best-image-macro-f1")
        if is_best_bbox:
            reasons.append("strict-best-bbox-macro-f1@0.1")
        selection = {
            "step": step,
            "image_macro_f1": image_f1,
            "bbox_macro_f1_at_0_1": bbox_f1,
            "checkpoint": row.get("checkpoint"),
            "is_best_image": bool(is_best_image),
            "is_best_bbox": bool(is_best_bbox),
            "is_4000_milestone": milestone,
            "checkpoint_kept": bool(reasons),
            "keep_reasons": reasons,
        }
        selections[step] = selection
        history.append(selection)
        if is_best_image:
            best_image = float(image_f1)
            current_image = selection
        if is_best_bbox:
            best_bbox = float(bbox_f1)
            current_bbox = selection

    document = {
        "schema_version": 1,
        "primary_best_metric": "bbox_macro_f1@0.1",
        "init_checkpoint": os.environ.get("INIT_CHECKPOINT", ""),
        "init_cpt_step": int(os.environ.get("INIT_CPT_STEP", "0")),
        "current_best": {
            "image": current_image,
            "bbox_at_0_1": current_bbox,
        },
        "best_history": [
            entry
            for entry in history
            if entry["is_best_image"] or entry["is_best_bbox"]
        ],
        "evaluation_history": history,
        "permanently_kept_steps": [
            entry["step"] for entry in history if entry["checkpoint_kept"]
        ],
    }
    return document, selections


def print_metric_summary(
    *,
    step: int,
    metrics: dict[str, Any],
    retention: dict[str, Any],
    diagnostics_xlsx: Path,
) -> None:
    def metric_line(
        values: dict[str, Any],
        *,
        count_rows: list[dict[str, Any]] | None = None,
    ) -> str:
        def count(name: str) -> int:
            value = values.get(name)
            if value is not None:
                return int(value)
            return sum(int(row.get(name) or 0) for row in (count_rows or []))

        return "P={:.6f} R={:.6f} F1={:.6f} TP={} FP={} FN={}".format(
            float(values.get("precision") or 0.0),
            float(values.get("recall") or 0.0),
            float(values.get("f1") or 0.0),
            count("tp"),
            count("fp"),
            count("fn"),
        )

    print(f"===== UI5 full-test scorer metrics: SFT step {step} =====")
    for task in TASKS:
        values = metrics.get("tasks", {}).get(task, {})
        issue = TASK_ISSUE_NAMES[task]
        print(f"{issue} Image {metric_line(values.get('image', {}))}")
        print(f"{issue} BBox@0.1 {metric_line(values.get('bbox', {}))}")
    macro = metrics.get("macro", {})
    image_task_rows = [
        metrics.get("tasks", {}).get(task, {}).get("image", {}) for task in TASKS
    ]
    bbox_task_rows = [
        metrics.get("tasks", {}).get(task, {}).get("bbox", {}) for task in TASKS
    ]
    print(
        "five_task_macro Image "
        + metric_line(macro.get("image", {}), count_rows=image_task_rows)
    )
    print(
        "five_task_macro BBox@0.1 "
        + metric_line(macro.get("bbox", {}), count_rows=bbox_task_rows)
    )
    print(
        "checkpoint 是否永久保留及原因: "
        f"{'是' if retention['checkpoint_kept'] else '否'}; "
        f"{','.join(retention['keep_reasons']) or 'temporary-evaluated-checkpoint'}"
    )
    print(f"Excel 路径: {diagnostics_xlsx}")
    print("=====================================================")


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
    record.add_argument("--evaluation-split", choices=("test",), default="test")
    record.add_argument(
        "--cache-scope",
        choices=("full_test",),
        default="full_test",
    )
    record.add_argument("--development-test-reuse", action="store_true")
    record.add_argument("--git-sha", default="")
    record.add_argument("--git-dirty", choices=("0", "1"), default="0")
    record.add_argument("--recipe-digest", default="")
    record.add_argument("--cache-digest", default="")
    record.add_argument("--frozen-gate-thresholds", type=Path, default=None)
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--metrics-json", type=Path, default=None)
    record.add_argument("--raw-metrics-json", type=Path, default=None)
    record.add_argument("--gated-metrics-json", type=Path, default=None)
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
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--git-commit", required=True)
    check.add_argument("--config-hash", required=True)

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
        requested_signature = ui_model_signature(args.checkpoint)
        found = any(
            evaluation_cache_row_matches(
                refresh_row_model_signature(row),
                step=args.step,
                relation_gate_mode=args.relation_gate_mode,
                git_commit=args.git_commit,
                config_hash=args.config_hash,
                model_signature=requested_signature,
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
            refresh_row_model_signature(step_zero)
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
    best_checkpoints_path = None
    if args.status == "success":
        metrics = load_metrics(args.metrics_json)
        best_document, selections = build_best_checkpoints_document(rows)
        retention = selections[args.step]
        row.update(
            {
                "is_best_image": retention["is_best_image"],
                "is_best_bbox": retention["is_best_bbox"],
                "is_4000_milestone": retention["is_4000_milestone"],
                "checkpoint_kept": retention["checkpoint_kept"],
                "checkpoint_keep_reasons": retention["keep_reasons"],
            }
        )
        diagnostics_xlsx = append_excel_evaluation(args, metrics, retention)
    write_history(args.history_dir.expanduser().resolve(), rows)
    if args.status == "success":
        best_checkpoints_path = (
            args.history_dir.expanduser().resolve() / "best_checkpoints.json"
        )
        atomic_write_text(
            best_checkpoints_path,
            json.dumps(best_document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        print_metric_summary(
            step=args.step,
            metrics=metrics,
            retention=retention,
            diagnostics_xlsx=diagnostics_xlsx,
        )
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
                "best_checkpoints_json": (
                    str(best_checkpoints_path)
                    if best_checkpoints_path is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
