#!/usr/bin/env python3
"""Create cumulative, time-triggered snapshots for the eight UI5 rollouts."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from run_ui5_train_rollout_worker import fixed_interleaved_samples


SCHEMA_VERSION = 1
MODELS = ("m31", "crop")
ROLLOUT_IDS = (0, 1, 2, 3)
DIFFICULTIES = ("easy", "medium", "hard", "incomplete_or_runtime_error")
CLASSIFICATION_FILES = {
    "easy": "easy.jsonl",
    "medium": "medium.jsonl",
    "hard": "hard.jsonl",
    "incomplete_or_runtime_error": "incomplete_or_runtime_error.jsonl",
}
GRPO_FILES = {
    "m31": "grpo_m31_ready.jsonl",
    "crop": "grpo_crop_ready.jsonl",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("hourly", "final"), required=True)
    parser.add_argument("--scheduled-hour", type=int)
    parser.add_argument("--started-at-epoch", type=float, required=True)
    parser.add_argument("--export-selection-dir", type=Path)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_token(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def visible_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read only newline-terminated records while a worker may still append."""
    payload = path.read_bytes()
    final_newline = payload.rfind(b"\n")
    if final_newline < 0:
        return []
    payload = payload[: final_newline + 1]
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid completed JSONL record at {path}:{line_no}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_no}")
        rows.append(value)
    return rows


def bundle_samples(bundle_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (bundle_root / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    samples = fixed_interleaved_samples(
        read_jsonl(bundle_root / "manifest" / "task_samples.jsonl")
    )
    expected_total = int(manifest["rollout_samples"])
    if len(samples) != expected_total:
        raise RuntimeError(
            f"bundle sample count mismatch: manifest={expected_total} rows={len(samples)}"
        )
    record_ids = [str(row["record_id"]) for row in samples]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("bundle contains duplicate record_id values")
    return samples


def load_visible_stream(
    output_root: Path,
    model: str,
    rollout_id: int,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    directory = output_root / "raw" / model / f"rollout_{rollout_id}"
    rows: dict[str, dict[str, Any]] = {}
    for part in sorted(directory.glob("part-*.jsonl")) if directory.is_dir() else []:
        for row in visible_jsonl_rows(part):
            record_id = str(row.get("record_id"))
            if record_id not in expected_ids:
                raise RuntimeError(
                    f"{model} rollout {rollout_id} has unknown record_id={record_id}"
                )
            if str(row.get("model_id")) != model or int(row.get("rollout_id", -1)) != rollout_id:
                raise RuntimeError(
                    f"raw stream identity mismatch: {part} record_id={record_id}"
                )
            if record_id in rows:
                raise RuntimeError(
                    f"duplicate record_id in {model} rollout {rollout_id}: {record_id}"
                )
            rows[record_id] = row
    return rows


def compact_runtime_error(row: Mapping[str, Any]) -> dict[str, Any] | None:
    runtime_error = row.get("runtime_error")
    if not isinstance(runtime_error, Mapping):
        return None
    return {
        "type": runtime_error.get("type"),
        "python_type": runtime_error.get("python_type"),
        "message": runtime_error.get("message"),
    }


def model_group_payload(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["rollout_id"]))
    return {
        **base,
        "group_model_id": model,
        "group_key": str(base["record_id"]),
        "answers": [row.get("raw_output") for row in ordered],
        "rewards_exact": [bool(row.get("exact_correct")) for row in ordered],
        "rollout_ids": [int(row["rollout_id"]) for row in ordered],
        "seeds": [int(row["seed"]) for row in ordered],
        "group_size": 4,
        "cross_model_group": False,
    }


def cumulative_metrics(
    streams: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    counters: dict[tuple[str, int, str], Counter[str]] = defaultdict(Counter)
    for (model, rollout_id), stream in streams.items():
        for row in stream.values():
            task = str(row.get("task"))
            for scope in (task, "micro"):
                counter = counters[(model, rollout_id, scope)]
                counter["attempted"] += 1
                if row.get("runtime_error"):
                    counter["runtime_error"] += 1
                    continue
                counter["inference_success"] += 1
                if row.get("parse_status") == "parse_error" or row.get(
                    "contains_crop_parse_error"
                ):
                    counter["parse_error"] += 1
                confusion = row.get("image_confusion")
                if confusion in {"TP", "TN", "FP", "FN"}:
                    counter[f"image_{confusion}"] += 1
                counter["bbox_TP"] += int(row.get("TP_box") or 0)
                counter["bbox_FP"] += int(row.get("FP_box") or 0)
                counter["bbox_FN"] += int(row.get("FN_box") or 0)
                counter["exact_correct"] += int(bool(row.get("exact_correct")))
    rows: list[dict[str, Any]] = []
    scopes = sorted({key[2] for key in counters})
    for model in MODELS:
        for rollout_id in ROLLOUT_IDS:
            for scope in scopes:
                counter = counters[(model, rollout_id, scope)]
                attempted = counter["attempted"]
                success = counter["inference_success"]
                rows.append(
                    {
                        "model_id": model,
                        "rollout_id": rollout_id,
                        "scope": scope,
                        "attempted": attempted,
                        "inference_success": success,
                        "runtime_error": counter["runtime_error"],
                        "parse_error": counter["parse_error"],
                        "image_TP": counter["image_TP"],
                        "image_TN": counter["image_TN"],
                        "image_FP": counter["image_FP"],
                        "image_FN": counter["image_FN"],
                        "image_TP_ratio": (
                            counter["image_TP"] / success if success else 0.0
                        ),
                        "image_TN_ratio": (
                            counter["image_TN"] / success if success else 0.0
                        ),
                        "image_FP_ratio": (
                            counter["image_FP"] / success if success else 0.0
                        ),
                        "image_FN_ratio": (
                            counter["image_FN"] / success if success else 0.0
                        ),
                        "bbox_TP": counter["bbox_TP"],
                        "bbox_FP": counter["bbox_FP"],
                        "bbox_FN": counter["bbox_FN"],
                        "exact_correct": counter["exact_correct"],
                        "inference_success_ratio": success / attempted if attempted else 0.0,
                        "runtime_error_ratio": (
                            counter["runtime_error"] / attempted if attempted else 0.0
                        ),
                        "exact_correct_ratio_of_inference_success": (
                            counter["exact_correct"] / success if success else 0.0
                        ),
                    }
                )
    return rows


def build_difficulty_records(
    output_root: Path, bundle_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = bundle_samples(bundle_root)
    expected_ids = {str(row["record_id"]) for row in samples}
    streams = {
        (model, rollout_id): load_visible_stream(
            output_root, model, rollout_id, expected_ids
        )
        for model in MODELS
        for rollout_id in ROLLOUT_IDS
    }
    records: list[dict[str, Any]] = []
    for sample in samples:
        record_id = str(sample["record_id"])
        by_model = {
            model: [
                streams[(model, rollout_id)][record_id]
                for rollout_id in ROLLOUT_IDS
                if record_id in streams[(model, rollout_id)]
            ]
            for model in MODELS
        }
        runtime_errors = [
            {
                "model_id": model,
                "rollout_id": int(row["rollout_id"]),
                **(compact_runtime_error(row) or {}),
            }
            for model in MODELS
            for row in by_model[model]
            if row.get("runtime_error")
        ]
        exact_available = all(
            isinstance(row.get("exact_correct"), bool)
            for model in MODELS
            for row in by_model[model]
            if not row.get("runtime_error")
        )
        completed = {model: len(by_model[model]) for model in MODELS}
        correct = {
            model: sum(row.get("exact_correct") is True for row in by_model[model])
            for model in MODELS
        }
        complete8 = (
            all(completed[model] == 4 for model in MODELS)
            and not runtime_errors
            and exact_available
        )
        total_correct = correct["m31"] + correct["crop"]
        if not complete8:
            difficulty = "incomplete_or_runtime_error"
        elif total_correct == 8:
            difficulty = "easy"
        elif total_correct == 0:
            difficulty = "hard"
        else:
            difficulty = "medium"
        grpo_m31 = bool(difficulty == "medium" and 1 <= correct["m31"] <= 3)
        grpo_crop = bool(difficulty == "medium" and 1 <= correct["crop"] <= 3)
        first_raw = next(
            (row for model in MODELS for row in by_model[model]), None
        )
        base = {
            "record_id": record_id,
            "sample_id": str(sample.get("sample_id", record_id)),
            "source_image_id": sample.get("source_image_id", sample.get("image_id")),
            "task": sample.get("task"),
            "image_relpath": sample.get("image_relpath"),
            "prompt": sample.get("prompt"),
            "gt_global": sample.get("gt_global"),
            "source_records": sample.get("source_records", []),
            "original_training_record": sample.get("original_training_record"),
            "m31_correct_count": correct["m31"],
            "crop_correct_count": correct["crop"],
            "total_correct_count": total_correct,
            "success_rate": total_correct / 8.0,
            "difficulty": difficulty,
            "grpo_ready_m31": grpo_m31,
            "grpo_ready_crop": grpo_crop,
            "m31_completed_rollout_count": completed["m31"],
            "crop_completed_rollout_count": completed["crop"],
            "completed_rollout_count": completed["m31"] + completed["crop"],
            "runtime_error_count": len(runtime_errors),
            "runtime_errors": runtime_errors,
            "parse_error_count": sum(
                bool(
                    row.get("parse_status") == "parse_error"
                    or row.get("contains_crop_parse_error")
                )
                for model in MODELS
                for row in by_model[model]
                if not row.get("runtime_error")
            ),
            "cross_model_complete8": complete8,
            "pipeline_coverage_failure": bool(
                (first_raw or sample).get("pipeline_coverage_failure")
            ),
            "annotation_anomaly": bool(
                (first_raw or sample).get("annotation_anomaly")
            ),
            "coordinate_transform_anomaly": bool(
                (first_raw or sample).get("coordinate_transform_anomaly")
            ),
            "visualization_rollouts": {
                model: [
                    {
                        "model_id": model,
                        "rollout_id": int(row["rollout_id"]),
                        "pred_global": row.get("pred_global") or [],
                        "matched_pairs": row.get("matched_pairs") or [],
                        "error_type": (
                            f"{model}:{row.get('error_type') or 'INCOMPLETE'}"
                        ),
                        "exact_correct": row.get("exact_correct") is True,
                        "crop_boundaries": [
                            item["crop_xyxy"]
                            for item in row.get("crop_outputs", [])
                            if item.get("crop_xyxy")
                        ],
                    }
                    for row in sorted(
                        by_model[model], key=lambda item: int(item["rollout_id"])
                    )
                ]
                for model in MODELS
            },
        }
        group_base = {
            key: value for key, value in base.items() if key != "visualization_rollouts"
        }
        if grpo_m31:
            base["grpo_m31_group"] = model_group_payload(
                "m31", by_model["m31"], group_base
            )
        if grpo_crop:
            base["grpo_crop_group"] = model_group_payload(
                "crop", by_model["crop"], group_base
            )
        records.append(base)
    stream_rows = []
    for model in MODELS:
        for rollout_id in ROLLOUT_IDS:
            stream = streams[(model, rollout_id)]
            stream_rows.append(
                {
                    "model_id": model,
                    "rollout_id": rollout_id,
                    "visible_records": len(stream),
                    "expected_records": len(samples),
                    "complete": len(stream) == len(samples),
                }
            )
    return records, {
        "expected_total": len(samples),
        "raw_streams": stream_rows,
        "cumulative_metrics": cumulative_metrics(streams),
    }


def classification_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"grpo_m31_group", "grpo_crop_group", "visualization_rollouts"}
    return {key: value for key, value in row.items() if key not in excluded}


def write_difficulty_exports(
    destination: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    previous_records: Mapping[str, Mapping[str, Any]] | None = None,
    write_delta: bool = False,
) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for difficulty, filename in CLASSIFICATION_FILES.items():
        selected = [
            classification_projection(row)
            for row in records
            if row["difficulty"] == difficulty
        ]
        counts[difficulty] = atomic_jsonl(destination / filename, selected)
    for model, filename in GRPO_FILES.items():
        flag = f"grpo_ready_{model}"
        group = f"grpo_{model}_group"
        selected = [row[group] for row in records if row.get(flag)]
        counts[f"grpo_{model}_ready"] = atomic_jsonl(destination / filename, selected)
    forbidden = destination / "cross_model_complete8.jsonl"
    if forbidden.exists():
        raise RuntimeError(
            f"obsolete cross_model_complete8.jsonl must not exist: {forbidden}"
        )
    if write_delta:
        previous = previous_records or {}
        delta_rows = []
        for row in records:
            projected = classification_projection(row)
            old = previous.get(str(row["record_id"]))
            if old == projected:
                continue
            delta_rows.append(
                {
                    **projected,
                    "delta_change": "new" if old is None else "updated",
                    "previous_state": (
                        None
                        if old is None
                        else {
                            key: old.get(key)
                            for key in (
                                "m31_correct_count",
                                "crop_correct_count",
                                "total_correct_count",
                                "success_rate",
                                "difficulty",
                                "grpo_ready_m31",
                                "grpo_ready_crop",
                                "completed_rollout_count",
                                "runtime_error_count",
                            )
                        }
                    ),
                }
            )
        counts["delta_since_previous"] = atomic_jsonl(
            destination / "delta_since_previous.jsonl", delta_rows
        )
    return counts


def read_snapshot_records(snapshot: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for filename in CLASSIFICATION_FILES.values():
        path = snapshot / filename
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            records[str(row["record_id"])] = row
    return records


def previous_snapshot(snapshots_root: Path) -> Path | None:
    candidates = [
        path
        for path in snapshots_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / "summary.json").is_file()
    ] if snapshots_root.is_dir() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def write_snapshot_workbook(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for incremental snapshot tables") from exc
    book = openpyxl.Workbook()
    book.remove(book.active)
    tables: dict[str, list[dict[str, Any]]] = {
        "overview": [
            {"metric": "snapshot_kind", "value": summary["snapshot_kind"]},
            {"metric": "scheduled_hour", "value": summary["scheduled_hour"]},
            {"metric": "started_at", "value": summary["started_at"]},
            {"metric": "created_at", "value": summary["created_at"]},
            {"metric": "elapsed_seconds", "value": summary["elapsed_seconds"]},
            {"metric": "expected_total", "value": summary["expected_total"]},
            {"metric": "previous_snapshot", "value": summary["previous_snapshot"]},
        ],
        "difficulty": [
            {
                "difficulty": difficulty,
                "samples": summary["difficulty_counts"][difficulty],
                "proportion": summary["difficulty_ratios"][difficulty],
            }
            for difficulty in DIFFICULTIES
        ],
        "per_task_difficulty": list(summary["per_task_difficulty"]),
        "raw_streams": list(summary["raw_streams"]),
        "cumulative_metrics": list(summary["cumulative_metrics"]),
        "selection_files": [
            {"file": filename, "records": count}
            for filename, count in sorted(summary["file_counts"].items())
        ],
    }
    fill = PatternFill("solid", fgColor="1F4E78")
    font = Font(color="FFFFFF", bold=True)
    for sheet_name, rows in tables.items():
        sheet = book.create_sheet(sheet_name[:31])
        columns = list(rows[0]) if rows else ["note"]
        sheet.append(columns)
        for row in rows or [{"note": "no rows"}]:
            sheet.append(
                [
                    json.dumps(row.get(column), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(column), (dict, list))
                    else row.get(column)
                    for column in columns
                ]
            )
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = fill
            cell.font = font
        for cells in sheet.columns:
            width = min(48, max(10, max(len(str(cell.value or "")) for cell in cells) + 2))
            sheet.column_dimensions[cells[0].column_letter].width = width
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.xlsx")
    book.save(temporary)
    verified = openpyxl.load_workbook(temporary, read_only=True, data_only=False)
    if set(tables) != set(verified.sheetnames):
        raise RuntimeError("incremental snapshot workbook verification failed")
    verified.close()
    os.replace(temporary, path)


def render_snapshot_gallery(
    snapshot_root: Path,
    bundle_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    panel_long_side: int = 420,
) -> dict[str, Any]:
    from PIL import Image
    from render_ui5_train_rollout_gallery import render_panel

    visual_root = snapshot_root / "visualizations"
    counts: Counter[tuple[str, str]] = Counter()
    seen_images: dict[tuple[str, str], set[str]] = defaultdict(set)
    cards: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        categories = [str(record["difficulty"])]
        if record.get("grpo_ready_m31"):
            categories.append("grpo_m31_ready")
        if record.get("grpo_ready_crop"):
            categories.append("grpo_crop_ready")
        for category in categories:
            task = str(record["task"])
            key = (category, task)
            image_id = str(record["source_image_id"])
            if len(seen_images[key]) >= 10 or image_id in seen_images[key]:
                continue
            seen_images[key].add(image_id)
            model_filter = (
                "m31" if category == "grpo_m31_ready" else
                "crop" if category == "grpo_crop_ready" else None
            )
            rollout_map = record.get("visualization_rollouts", {})
            rollouts = [
                rollout
                for model in MODELS
                if model_filter is None or model == model_filter
                for rollout in rollout_map.get(model, [])
            ]
            source_path = bundle_root / str(record["image_relpath"])
            try:
                with Image.open(source_path) as opened:
                    source = opened.convert("RGB")
                if not rollouts:
                    rollouts = [
                        {
                            "rollout_id": "none",
                            "pred_global": [],
                            "matched_pairs": [],
                            "error_type": "INCOMPLETE",
                            "exact_correct": False,
                            "crop_boundaries": [],
                        }
                    ]
                panels = [
                    render_panel(
                        source,
                        record.get("gt_global") or [],
                        rollout,
                        panel_long_side,
                    )
                    for rollout in rollouts
                ]
                source.close()
                columns = min(4, len(panels))
                rows = (len(panels) + columns - 1) // columns
                gap = 10
                cell_width = max(panel.width for panel in panels)
                cell_height = max(panel.height for panel in panels)
                composite = Image.new(
                    "RGB",
                    (
                        columns * cell_width + (columns - 1) * gap,
                        rows * cell_height + (rows - 1) * gap,
                    ),
                    (238, 242, 247),
                )
                for index, panel in enumerate(panels):
                    x = (index % columns) * (cell_width + gap)
                    y = (index // columns) * (cell_height + gap)
                    composite.paste(panel, (x, y))
                    panel.close()
                relative = (
                    Path("assets") / category / task / f"{record['record_id']}.jpg"
                )
                destination = visual_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.tmp-{os.getpid()}"
                )
                composite.save(temporary, format="JPEG", quality=88, subsampling=0)
                composite.close()
                os.replace(temporary, destination)
                cards.append(
                    {
                        "category": category,
                        "task": task,
                        "record_id": record["record_id"],
                        "source_image_id": image_id,
                        "difficulty": record["difficulty"],
                        "m31_correct_count": record["m31_correct_count"],
                        "crop_correct_count": record["crop_correct_count"],
                        "total_correct_count": record["total_correct_count"],
                        "visual_relpath": relative.as_posix(),
                    }
                )
                counts[key] += 1
            except Exception as exc:
                failures.append(
                    {
                        "record_id": record.get("record_id"),
                        "category": category,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    sections = []
    for key in sorted(counts):
        category, task = key
        selected = [
            card
            for card in cards
            if card["category"] == category and card["task"] == task
        ]
        cards_html = "".join(
            "<article><img loading='lazy' src='"
            + html.escape(str(card["visual_relpath"]))
            + "'><div><code>"
            + html.escape(str(card["record_id"]))
            + "</code> m31="
            + str(card["m31_correct_count"])
            + "/4 crop="
            + str(card["crop_correct_count"])
            + "/4 total="
            + str(card["total_correct_count"])
            + "/8</div></article>"
            for card in selected
        )
        sections.append(
            f"<section><h2>{html.escape(category)} / {html.escape(task)} "
            f"({len(selected)})</h2><div class='grid'>{cards_html}</div></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>UI5 incremental rollout snapshot</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f8fb;color:#172033}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}
article{{background:white;border:1px solid #dbe2ea;border-radius:8px;padding:9px}}
img{{width:100%;height:auto}} code{{font-size:12px}}
</style></head><body><h1>UI5 incremental rollout snapshot</h1>
<p>Green=GT, blue=matched prediction, red=unmatched prediction, yellow=crop.</p>
{''.join(sections)}</body></html>"""
    visual_root.mkdir(parents=True, exist_ok=True)
    (visual_root / "index.html").write_text(document, encoding="utf-8")
    result = {
        "index": "visualizations/index.html",
        "rendered": len(cards),
        "failures": failures,
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
    }
    atomic_json(visual_root / "gallery_summary.json", result)
    return result


def create_snapshot(
    output_root: Path,
    bundle_root: Path,
    *,
    kind: str,
    scheduled_hour: int | None,
    started_at_epoch: float,
    export_selection_dir: Path | None = None,
    created_at_epoch: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    if kind == "hourly":
        if scheduled_hour is None or scheduled_hour < 3 or scheduled_hour % 3:
            raise ValueError("hourly snapshots require --scheduled-hour 3, 6, 9, ...")
    elif kind != "final":
        raise ValueError(f"unsupported snapshot kind: {kind}")
    now_epoch = time.time() if created_at_epoch is None else created_at_epoch
    snapshots_root = output_root / "incremental_snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    previous = previous_snapshot(snapshots_root)
    previous_records = read_snapshot_records(previous) if previous else {}
    records, cumulative = build_difficulty_records(output_root, bundle_root)
    prefix = f"hour_{scheduled_hour:03d}" if kind == "hourly" else "final"
    base_name = f"{prefix}_{timestamp_token(now_epoch)}"
    destination = snapshots_root / base_name
    suffix = 1
    while destination.exists():
        destination = snapshots_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    temporary = snapshots_root / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        file_counts = write_difficulty_exports(
            temporary,
            records,
            previous_records=previous_records,
            write_delta=True,
        )
        difficulty_counts = Counter(str(row["difficulty"]) for row in records)
        task_counts = Counter(
            (str(row["task"]), str(row["difficulty"])) for row in records
        )
        task_totals = Counter(str(row["task"]) for row in records)
        expected_total = int(cumulative["expected_total"])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_kind": kind,
            "scheduled_hour": scheduled_hour,
            "started_at_epoch": started_at_epoch,
            "started_at": datetime.fromtimestamp(
                started_at_epoch, timezone.utc
            ).isoformat(),
            "created_at_epoch": now_epoch,
            "created_at": datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(),
            "elapsed_seconds": max(0.0, now_epoch - started_at_epoch),
            "previous_snapshot": previous.name if previous else None,
            "expected_total": expected_total,
            "classification_policy": {
                "complete8_required": True,
                "runtime_error_excluded": True,
                "easy": "total_correct_count == 8",
                "medium": "1 <= total_correct_count <= 7",
                "hard": "total_correct_count == 0",
                "incomplete_or_runtime_error": (
                    "fewer than eight raw records, a runtime_error, or missing exact_correct"
                ),
                "grpo_ready_subset": "medium only",
                "grpo_ready_m31": "1 <= m31_correct_count <= 3",
                "grpo_ready_crop": "1 <= crop_correct_count <= 3",
            },
            "difficulty_counts": {
                difficulty: difficulty_counts[difficulty]
                for difficulty in DIFFICULTIES
            },
            "difficulty_ratios": {
                difficulty: difficulty_counts[difficulty] / expected_total
                if expected_total
                else 0.0
                for difficulty in DIFFICULTIES
            },
            "per_task_difficulty": [
                {
                    "task": task,
                    "difficulty": difficulty,
                    "samples": task_counts[(task, difficulty)],
                    "proportion": (
                        task_counts[(task, difficulty)] / task_totals[task]
                        if task_totals[task]
                        else 0.0
                    ),
                }
                for task in sorted(task_totals)
                for difficulty in DIFFICULTIES
            ],
            "grpo_ready_counts": {
                "m31": file_counts["grpo_m31_ready"],
                "crop": file_counts["grpo_crop_ready"],
            },
            "file_counts": file_counts,
            **cumulative,
        }
        summary["visualizations"] = render_snapshot_gallery(
            temporary, bundle_root, records
        )
        summary["statistics_workbook"] = "snapshot_statistics.xlsx"
        write_snapshot_workbook(temporary / "snapshot_statistics.xlsx", summary)
        atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if export_selection_dir is not None:
        write_difficulty_exports(export_selection_dir, records)
    print(
        json.dumps(
            {
                "snapshot": str(destination),
                "kind": kind,
                "scheduled_hour": scheduled_hour,
                "counts": summary["difficulty_counts"],
                "delta": summary["file_counts"]["delta_since_previous"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return destination, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_snapshot(
        args.output_root.expanduser().resolve(strict=True),
        args.bundle_root.expanduser().resolve(strict=True),
        kind=args.kind,
        scheduled_hour=args.scheduled_hour,
        started_at_epoch=args.started_at_epoch,
        export_selection_dir=(
            args.export_selection_dir.expanduser().resolve(strict=False)
            if args.export_selection_dir is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
