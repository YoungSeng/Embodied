#!/usr/bin/env python3
"""Build fail-closed LocateAnything full/full+crop recipes from a v4 audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from analyze_ui5_source_overlap import content_fingerprint
from run_ui5_crop_audit import (
    ProgressReporter,
    atomic_write_json,
    atomic_write_jsonl,
    audit_state_digest,
    read_jsonl,
)
from run_ui5_gt_repair import (
    EXCLUDED_SAMPLE_ID,
    EXCLUDED_TASK,
    V4_GATE_CONDITIONS,
)


BOX_RE = re.compile(
    r"<box><(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)>"
    r"<(-?\d+(?:\.\d+)?)><(-?\d+(?:\.\d+)?)></box>"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--base-meta", type=Path, required=True)
    parser.add_argument("--task-aware-manifest", type=Path, required=True)
    parser.add_argument("--excluded-samples", type=Path, required=True)
    parser.add_argument("--mode", choices=("full_only", "full_plus_crop"), default="full_plus_crop")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-valid-gt-recall", type=float, default=1.0)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _answer(record: Mapping[str, Any]) -> str:
    conversations = record.get("conversations", [])
    for row in reversed(conversations):
        if row.get("from") in {"gpt", "assistant"}:
            return str(row.get("value", ""))
    return ""


def _positive(record: Mapping[str, Any]) -> bool:
    answer = _answer(record)
    return bool(BOX_RE.search(answer)) and "<box>none</box>" not in answer


def _resolve_path(value: str | Path, *, base: Path, cwd: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    candidates = ((base / path), (cwd / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(
        f"cannot resolve recipe path {value!r}; tried {[str(row) for row in candidates]}"
    )


def _resolve_media_path(
    image: str,
    *,
    root: str,
    meta_path: Path,
    paths_relative_to_meta: bool,
    cwd: Path,
) -> Path:
    value = Path(image).expanduser()
    if value.is_absolute():
        return value.resolve(strict=True)
    if root:
        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            root_path = (
                meta_path.parent / root_path
                if paths_relative_to_meta
                else cwd / root_path
            )
        return (root_path / value).resolve(strict=True)
    return _resolve_path(value, base=meta_path.parent, cwd=cwd)


def _iter_base_records(
    base_meta_path: Path,
) -> Iterable[tuple[dict[str, Any], Path, int, Mapping[str, Any]]]:
    payload = json.loads(base_meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("base meta must be a non-empty dataset mapping")
    cwd = Path.cwd().resolve()
    for dataset_name, raw_entry in payload.items():
        entry = dict(raw_entry)
        annotations = entry.get("annotation")
        if not annotations:
            raise ValueError(f"base meta entry lacks annotation: {dataset_name}")
        values = annotations if isinstance(annotations, list) else [annotations]
        paths_relative = bool(entry.get("paths_relative_to_meta", False))
        for value in values:
            annotation = _resolve_path(
                value,
                base=base_meta_path.parent if paths_relative else cwd,
                cwd=cwd,
            )
            for line_no, record in enumerate(read_jsonl(annotation), 1):
                yield entry, annotation, line_no, record


def _source_record_map(
    manifest_rows: Sequence[Mapping[str, Any]],
    parent_task_samples: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    final_sample_ids = {str(row["sample_id"]) for row in manifest_rows}
    mapping: dict[tuple[str, int], Mapping[str, Any]] = {}
    for sample in parent_task_samples:
        sample_id = str(sample["sample_id"])
        # The excluded sample is deliberately absent from the repaired task
        # manifest but still needs to map so it can be explicitly filtered.
        if sample_id not in final_sample_ids and sample_id != EXCLUDED_SAMPLE_ID:
            continue
        for source in sample.get("source_records", []):
            key = (str(Path(str(source["source_file"])).resolve()), int(source["line_no"]))
            if key in mapping and mapping[key]["sample_id"] != sample_id:
                raise ValueError(f"source record maps to multiple task samples: {key}")
            mapping[key] = sample
    return mapping


def _validate_crop_record(record: Mapping[str, Any]) -> None:
    if record.get("_ui5_record_kind") != "crop":
        raise ValueError("task-aware training record is not marked as crop")
    if record.get("_ui5_training_eligible") is not True:
        raise ValueError("ineligible crop record reached recipe builder")
    image = Path(str(record["image"])).expanduser().resolve(strict=True)
    if not image.is_file() or image.stat().st_size <= 0:
        raise FileNotFoundError(f"crop image is missing or empty: {image}")
    answer = _answer(record)
    for match in BOX_RE.findall(answer):
        values = [float(value) for value in match]
        if not all(0 <= value <= 1000 for value in values):
            raise ValueError(f"crop label is outside normalized coordinate space: {values}")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"crop label has zero/negative area: {values}")


def _recipe_entry(
    annotation_name: str,
    *,
    excluded_manifest_relative: str,
    crop_recipe: bool,
    recipe_summary_name: str,
) -> dict[str, Any]:
    return {
        "annotation": [annotation_name],
        "root": "",
        "repeat_time": 1.0,
        "data_augment": False,
        "paths_relative_to_meta": True,
        "ui5_crop_recipe": bool(crop_recipe),
        "excluded_samples": excluded_manifest_relative,
        "recipe_summary": recipe_summary_name,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = args.audit_dir.expanduser().resolve(strict=True)
    base_meta = args.base_meta.expanduser().resolve(strict=True)
    task_manifest_path = args.task_aware_manifest.expanduser().resolve(strict=True)
    excluded_path = args.excluded_samples.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.parent != audit_dir and audit_dir not in output_dir.parents:
        raise ValueError("training recipe output must remain inside the v4 audit directory")
    marker_path = audit_dir / "training_ready.json"
    marker_path.unlink(missing_ok=True)

    summary_path = audit_dir / "summary.json"
    state_path = audit_dir / "audit_state.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    repair_recall = float(
        summary["repair_metrics"]["training_materialization_gt_recall_after_repair"]
    )
    if repair_recall + 1e-12 < float(args.require_valid_gt_recall):
        raise RuntimeError(
            f"post-repair GT recall {repair_recall:.8f} is below required "
            f"{args.require_valid_gt_recall:.8f}"
        )

    manifest_rows = read_jsonl(task_manifest_path)
    exclusions = read_jsonl(excluded_path)
    if len(exclusions) != 1 or exclusions[0]["sample_id"] != EXCLUDED_SAMPLE_ID:
        raise ValueError("v4 recipe requires exactly the confirmed annotation exclusion")
    parent_task_samples = read_jsonl(audit_dir.parent / "manifest" / "task_samples.jsonl")
    source_map = _source_record_map(manifest_rows, parent_task_samples)
    reporter = ProgressReporter(
        stage="gt-repair-recipe",
        total=len(parent_task_samples) + len(manifest_rows),
        output_dir=audit_dir,
        interval_seconds=getattr(args, "progress_interval_seconds", 10.0),
        unit="records",
    )
    reporter.update(0, detail="读取 full-image 训练记录", force=True)
    processed = 0

    full_records: list[dict[str, Any]] = []
    unmatched: list[tuple[str, int]] = []
    excluded_matches = 0
    for entry, annotation, line_no, raw_record in _iter_base_records(base_meta):
        processed += 1
        key = (str(annotation.resolve()), line_no)
        sample = source_map.get(key)
        if sample is None:
            unmatched.append(key)
            reporter.update(processed, detail="匹配 full-image 训练记录")
            continue
        sample_id, task = str(sample["sample_id"]), str(sample["task"])
        if sample_id == EXCLUDED_SAMPLE_ID and task == EXCLUDED_TASK:
            excluded_matches += 1
            reporter.update(processed, detail=f"排除 {EXCLUDED_SAMPLE_ID}")
            continue
        record = dict(raw_record)
        raw_images = record.get("image")
        if not isinstance(raw_images, str):
            raise ValueError(f"UI5 base record must contain one image string: {key}")
        record["image"] = str(
            _resolve_media_path(
                raw_images,
                root=str(entry.get("root", "")),
                meta_path=base_meta,
                paths_relative_to_meta=bool(entry.get("paths_relative_to_meta", False)),
                cwd=Path.cwd().resolve(),
            )
        )
        record.update(
            {
                "_ui5_sample_id": sample_id,
                "_ui5_image_id": sample["image_id"],
                "_ui5_task": task,
                "_ui5_split": sample.get("split"),
                "_ui5_record_kind": "full_image",
                "_ui5_crop_source": None,
            }
        )
        full_records.append(record)
        reporter.update(processed, detail="匹配 full-image 训练记录")
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} base-meta records do not map to the task-aware manifest; "
            f"first={unmatched[:5]}"
        )
    if excluded_matches < 1:
        raise ValueError("confirmed annotation error was not found in base full-image records")

    crop_records: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        for raw_record in manifest.get("training_records", []):
            record = dict(raw_record)
            _validate_crop_record(record)
            if (
                record.get("_ui5_sample_id") == EXCLUDED_SAMPLE_ID
                and record.get("_ui5_task") == EXCLUDED_TASK
            ):
                raise ValueError("excluded text-overflow sample reached crop records")
            if record.get("_ui5_crop_source") == "manual_gt_repair" and record.get(
                "_ui5_split"
            ) != "train":
                raise RuntimeError("manual_gt_repair reached validation/test recipe")
            crop_records.append(record)
        processed += 1
        reporter.update(processed, detail="校验 crop 记录与图片路径")
    if not crop_records:
        raise ValueError("full_plus_crop recipe would contain zero crop records")
    if not any(row.get("_ui5_crop_source") == "manual_gt_repair" for row in crop_records):
        raise ValueError("crop recipe contains no manual_gt_repair record")
    if not any(row.get("_ui5_crop_source") == "raw_detector" for row in crop_records):
        raise ValueError("crop recipe contains no ordinary detector crop record")

    all_referenced_images = [
        Path(str(row["image"])).resolve() for row in (*full_records, *crop_records)
    ]
    missing_images = [str(path) for path in all_referenced_images if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"recipe references {len(missing_images)} missing images; first={missing_images[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    full_jsonl = output_dir / "ui_defect_5class_train_full_only.jsonl"
    combined_jsonl = output_dir / "ui_defect_5class_train_full_plus_crop.jsonl"
    full_meta = output_dir / "ui_defect_5class_train_full_only.json"
    combined_meta = output_dir / "ui_defect_5class_train_full_plus_crop.json"
    recipe_summary_path = output_dir / "recipe_summary.json"
    atomic_write_jsonl(full_jsonl, full_records)
    atomic_write_jsonl(combined_jsonl, [*full_records, *crop_records])
    excluded_relative = Path(os.path.relpath(excluded_path, output_dir)).as_posix()
    _atomic_write_json(
        full_meta,
        {
            "ui_defect_5class_train_full_only": _recipe_entry(
                full_jsonl.name,
                excluded_manifest_relative=excluded_relative,
                crop_recipe=False,
                recipe_summary_name=recipe_summary_path.name,
            )
        },
    )
    _atomic_write_json(
        combined_meta,
        {
            "ui_defect_5class_train_full_plus_crop": _recipe_entry(
                combined_jsonl.name,
                excluded_manifest_relative=excluded_relative,
                crop_recipe=True,
                recipe_summary_name=recipe_summary_path.name,
            )
        },
    )

    counts_by_task = Counter(str(row["_ui5_task"]) for row in (*full_records, *crop_records))
    positive_by_kind = Counter(
        (str(row["_ui5_record_kind"]), "positive" if _positive(row) else "negative")
        for row in (*full_records, *crop_records)
    )
    recipe_summary = {
        "schema_version": 1,
        "mode": args.mode,
        "base_meta": str(base_meta),
        "full_image_records": len(full_records),
        "crop_records": len(crop_records),
        "ordinary_detector_crop_records": sum(
            row.get("_ui5_crop_source") == "raw_detector" for row in crop_records
        ),
        "gt_repair_crop_records": sum(
            row.get("_ui5_crop_source") == "manual_gt_repair" for row in crop_records
        ),
        "records_by_task": dict(sorted(counts_by_task.items())),
        "positive_negative_records": {
            f"{kind}_{polarity}": count
            for (kind, polarity), count in sorted(positive_by_kind.items())
        },
        "excluded_records": excluded_matches,
        "excluded_sample_ids": [EXCLUDED_SAMPLE_ID],
        "all_referenced_images_exist": not missing_images,
        "full_only_meta": str(full_meta.resolve()),
        "full_only_jsonl": str(full_jsonl.resolve()),
        "full_plus_crop_meta": str(combined_meta.resolve()),
        "full_plus_crop_jsonl": str(combined_jsonl.resolve()),
        "full_only_recipe_digest": content_fingerprint(full_meta),
        "full_only_jsonl_digest": content_fingerprint(full_jsonl),
        "full_plus_crop_recipe_digest": content_fingerprint(combined_meta),
        "full_plus_crop_jsonl_digest": content_fingerprint(combined_jsonl),
        "excluded_samples_digest": content_fingerprint(excluded_path),
    }
    atomic_write_json(recipe_summary_path, recipe_summary)

    selected_meta = combined_meta if args.mode == "full_plus_crop" else full_meta
    conditions = dict(summary["next_stage_gate"]["conditions"])
    final_report_files = (
        audit_dir / "summary.json",
        audit_dir / "audit_state.json",
        audit_dir / "materialization_summary.json",
        audit_dir / "statistics.csv",
        audit_dir / "ui5_crop_audit.xlsx",
        audit_dir / "task_aware_manifest.jsonl",
        audit_dir / "gt_repair_detections.jsonl",
        audit_dir / "gt_repair_actions.jsonl",
        audit_dir / "excluded_training_samples.jsonl",
        audit_dir / "gt_repair_visualizations" / "gallery" / "index.html",
    )
    conditions.update(
        {
            "excluded_sample_absent_from_text_overflow_recipe": not any(
                row.get("_ui5_sample_id") == EXCLUDED_SAMPLE_ID
                and row.get("_ui5_task") == EXCLUDED_TASK
                for row in (*full_records, *crop_records)
            ),
            "crop_training_recipe_written_successfully": all(
                path.is_file() and path.stat().st_size > 0
                for path in (full_meta, combined_meta, combined_jsonl, recipe_summary_path)
            ),
            "crop_training_recipe_contains_crop_records": len(crop_records) > 0,
            "all_reports_written_successfully": all(
                path.is_file() and path.stat().st_size > 0 for path in final_report_files
            ),
            "gt_repair_not_applied_to_val_test": not any(
                row.get("_ui5_crop_source") == "manual_gt_repair"
                and row.get("_ui5_split") != "train"
                for row in crop_records
            ),
        }
    )
    if set(conditions) != V4_GATE_CONDITIONS:
        raise RuntimeError(
            f"v4 training gate schema differs: missing={sorted(V4_GATE_CONDITIONS - set(conditions))}, "
            f"extra={sorted(set(conditions) - V4_GATE_CONDITIONS)}"
        )
    passes = all(bool(value) for value in conditions.values())
    summary["next_stage_gate"] = {
        "conditions": conditions,
        "passes": passes,
        "training_ready": passes,
        "training_started": False,
        "failed_conditions": [name for name, value in conditions.items() if not value],
    }
    summary["training_ready"] = passes
    summary["training_started"] = False
    summary["recipe_state"] = {
        "written": True,
        "mode": args.mode,
        "selected_meta": str(selected_meta.resolve()),
        "recipe_summary": str(recipe_summary_path.resolve()),
        "recipe_digest": content_fingerprint(selected_meta),
        "recipe_jsonl_digest": content_fingerprint(combined_jsonl),
        "recipe_summary_digest": content_fingerprint(recipe_summary_path),
        "excluded_samples_digest": content_fingerprint(excluded_path),
        **recipe_summary,
    }
    atomic_write_json(summary_path, summary)
    if not passes:
        raise RuntimeError(
            f"v4 recipe was written but final training gate failed: "
            f"{summary['next_stage_gate']['failed_conditions']}"
        )

    marker = {
        "schema_version": 2,
        "training_ready": True,
        "recommended_config": summary["recommended_config"],
        "training_started": False,
        "audit_state_digest": audit_state_digest(state),
        "input_snapshot_digest": summary["input_snapshot_digest"],
        "summary_file_digest": content_fingerprint(summary_path),
        "excluded_samples_digest": content_fingerprint(excluded_path),
        "training_recipe_digest": content_fingerprint(selected_meta),
        "training_recipe_jsonl_digest": content_fingerprint(combined_jsonl),
        "full_only_recipe_digest": content_fingerprint(full_meta),
        "full_only_recipe_jsonl_digest": content_fingerprint(full_jsonl),
        "recipe_summary_digest": content_fingerprint(recipe_summary_path),
        "training_recipe": str(selected_meta.resolve()),
        "created_after_all_checks": True,
    }
    reporter.update(
        len(parent_task_samples) + len(manifest_rows),
        status="completed",
        detail="recipe 与 digest 已完成，下一步原子发布 training_ready marker",
        force=True,
    )
    # This marker must remain the final write in the audit transaction.  The
    # shell wrapper prints the user-facing completion line after this returns.
    atomic_write_json(marker_path, marker)
    return recipe_summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = build(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
