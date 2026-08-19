#!/usr/bin/env python3
"""Maintain evaluation_history.json/csv for LocateAnything UI5 runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from locany_ui5_common import TASK_ISSUE_NAMES, TASKS


BASE_COLUMNS = [
    "step",
    "machine_type",
    "gpu_count",
    "max_num_tokens",
    "max_num_tokens_scope",
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
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--metrics-json", type=Path, default=None)
    record.add_argument("--start-time", required=True)
    record.add_argument("--end-time", required=True)
    record.add_argument("--status", choices=("success", "failed"), required=True)
    record.add_argument("--prediction-dir", type=Path, default=None)
    record.add_argument("--evaluation-run-dir", type=Path, default=None)
    record.add_argument("--error", default="")

    check = subparsers.add_parser("has-success")
    check.add_argument("--history-dir", type=Path, required=True)
    check.add_argument("--step", type=int, required=True)

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
            for row in rows
        )
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
    rows = [existing for existing in rows if int(existing.get("step", -1)) != args.step]
    rows.append(row)
    write_history(args.history_dir.expanduser().resolve(), rows)
    print(
        json.dumps(
            {
                "history_json": str(history_path),
                "history_csv": str(history_path.with_suffix(".csv")),
                "recorded_step": args.step,
                "status": args.status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
