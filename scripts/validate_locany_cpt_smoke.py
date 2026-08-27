#!/usr/bin/env python3
"""Validate completed CPT smoke outputs and cross-profile metric schemas."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


CPT_TASKS = {
    "ui_caption",
    "agent_action",
    "agent_grounding",
    "ui_defect",
    "all_ui_elements",
    "single_grounding",
    "ocr",
    "referring_kg",
    "referring",
    "vqa",
}

REQUIRED_TRAIN_FIELDS = {
    "step",
    "scope",
    "task",
    "attempted_samples",
    "accepted_samples",
    "trained_samples",
    "oversize_skipped_samples",
    "main_supervised_tokens",
    "mtp_supervised_tokens",
    "total_supervised_tokens",
    "train_main_token_ce",
    "train_mtp_token_ce",
    "train_total_token_ce",
    "row_coverage",
    "group_coverage",
    "effective_epoch",
    "repeat_factor",
    "packing_efficiency",
    "window_oversize_record_hashes",
    "global_attempted_samples",
    "global_trained_samples",
    "global_total_supervised_tokens",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def workbook_sheets(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [str(value.attrib["name"]) for value in root.findall("x:sheets/x:sheet", namespace)]


def _checkpoint_report(run_dir: Path, step: int, world_size: int) -> dict[str, Any]:
    checkpoint = run_dir / f"checkpoint-{step}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing checkpoint-{step}: {checkpoint}")
    marker = checkpoint / "checkpoint_complete.json"
    if not marker.is_file():
        raise RuntimeError(f"checkpoint-{step} has no completion marker")
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    if int(marker_value.get("global_step", -1)) != step:
        raise RuntimeError(f"checkpoint marker step mismatch: {marker_value}")
    missing_states = [
        rank
        for rank in range(world_size)
        if not (checkpoint / f"dataloader_state_rank{rank}.pt").is_file()
    ]
    if missing_states:
        raise RuntimeError(
            f"checkpoint-{step} missing dataloader state for ranks={missing_states}"
        )
    return {
        "path": str(checkpoint),
        "marker": str(marker),
        "rank_states": world_size,
    }


def validate_run(
    run_dir: Path,
    *,
    min_step: int = 20,
    resume_step: int = 10,
    require_excel: bool = True,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    diagnostics = run_dir / "diagnostics"
    config_path = diagnostics / "cpt_run_config.json"
    metrics_path = diagnostics / "cpt_train_metrics.jsonl"
    if not config_path.is_file() or not metrics_path.is_file():
        raise RuntimeError(f"missing CPT smoke diagnostics under {diagnostics}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    world_size = int(config.get("world_size") or 0)
    if world_size <= 0:
        raise RuntimeError(f"invalid world_size in {config_path}")
    configured_tasks = {str(row.get("task")) for row in config.get("datasets", [])}
    if configured_tasks != CPT_TASKS:
        raise RuntimeError(
            f"run config task mismatch: missing={sorted(CPT_TASKS - configured_tasks)}, "
            f"unexpected={sorted(configured_tasks - CPT_TASKS)}"
        )

    rows = read_jsonl(metrics_path)
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task"))
        step = int(row.get("step") or -1)
        missing = sorted(REQUIRED_TRAIN_FIELDS.difference(row))
        if missing:
            raise RuntimeError(f"step={step} task={task} missing fields={missing}")
        if int(row["attempted_samples"]) != int(row["accepted_samples"]) + int(
            row["oversize_skipped_samples"]
        ):
            raise RuntimeError(f"sample identity failed at step={step} task={task}")
        if int(row["main_supervised_tokens"]) + int(row["mtp_supervised_tokens"]) != int(
            row["total_supervised_tokens"]
        ):
            raise RuntimeError(f"token identity failed at step={step} task={task}")
        by_step[step].append(row)
        by_task[task].append(row)

    final_step = max(by_step, default=-1)
    if final_step < min_step:
        raise RuntimeError(f"smoke ended at step={final_step}, expected >= {min_step}")
    final_rows = by_step[final_step]
    final_tasks = {str(row["task"]) for row in final_rows}
    if final_tasks != CPT_TASKS or len(final_rows) != len(CPT_TASKS):
        raise RuntimeError(
            f"final step task rows invalid: tasks={sorted(final_tasks)}, rows={len(final_rows)}"
        )
    final_schemas = {tuple(sorted(row)) for row in final_rows}
    if len(final_schemas) != 1:
        raise RuntimeError("per-task final metric schemas differ within the run")
    for task, task_rows in by_task.items():
        ordered = sorted(task_rows, key=lambda row: int(row["step"]))
        for left, right in zip(ordered, ordered[1:]):
            for field in (
                "attempted_samples",
                "accepted_samples",
                "trained_samples",
                "oversize_skipped_samples",
                "main_supervised_tokens",
                "mtp_supervised_tokens",
                "total_supervised_tokens",
            ):
                if int(right[field]) < int(left[field]):
                    raise RuntimeError(
                        f"non-monotonic {field}: task={task}, "
                        f"steps={left['step']}->{right['step']}"
                    )
        final = ordered[-1]
        if int(final["trained_samples"]) <= 0:
            raise RuntimeError(f"task={task} received no trained samples")
        if int(final.get("main_loss_tokens") or 0) <= 0:
            raise RuntimeError(f"task={task} has no main CE tokens")
        if final.get("train_main_token_ce") is None:
            raise RuntimeError(f"task={task} has no train main CE")
        for field in ("row_coverage", "group_coverage", "repeat_factor"):
            if final.get(field) is None:
                raise RuntimeError(f"task={task} final {field} is missing")

    queue_path = diagnostics / "cpt_eval_queue.jsonl"
    queue_steps = {
        int(row["step"])
        for row in read_jsonl(queue_path)
    } if queue_path.is_file() else set()
    required_queue_steps = {resume_step, final_step}
    if not required_queue_steps.issubset(queue_steps):
        raise RuntimeError(
            f"eval queue missing steps={sorted(required_queue_steps - queue_steps)}"
        )

    checkpoints = {
        str(step): _checkpoint_report(run_dir, step, world_size)
        for step in sorted(required_queue_steps)
    }
    workbook = diagnostics / "cpt_training_evaluation.xlsx"
    sheets = None
    if require_excel:
        if not workbook.is_file():
            raise RuntimeError(f"missing smoke workbook: {workbook}")
        sheets = workbook_sheets(workbook)
        if sheets != ["Overview", "TrainMetrics", "EvalMetrics"]:
            raise RuntimeError(f"unexpected workbook sheets={sheets}")
    if not (run_dir / "done.txt").is_file():
        raise RuntimeError("final smoke segment did not write done.txt")

    return {
        "run_dir": str(run_dir),
        "world_size": world_size,
        "final_step": final_step,
        "metric_steps": sorted(by_step),
        "tasks": sorted(final_tasks),
        "final_schema": list(next(iter(final_schemas))),
        "checkpoints": checkpoints,
        "eval_queue_steps": sorted(queue_steps),
        "workbook_sheets": sheets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for A800/H20 smoke run directories",
    )
    parser.add_argument("--min-step", type=int, default=20)
    parser.add_argument("--resume-step", type=int, default=10)
    parser.add_argument("--no-require-excel", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = {}
    for value in args.run:
        if "=" not in value:
            parser.error(f"--run must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        reports[name] = validate_run(
            Path(raw_path),
            min_step=args.min_step,
            resume_step=args.resume_step,
            require_excel=not args.no_require_excel,
        )
    schemas = {name: report["final_schema"] for name, report in reports.items()}
    if len({tuple(value) for value in schemas.values()}) > 1:
        raise RuntimeError(f"cross-profile final metric schemas differ: {schemas}")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "cross_profile_schema_equal": True,
        "runs": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
