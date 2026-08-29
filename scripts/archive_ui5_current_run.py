#!/usr/bin/env python3
"""Validate and archive a stopped UI5 run without deleting or moving artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from analyze_ui5_source_overlap import content_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.ui5_checkpoint_utils import list_training_checkpoints, validate_checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--scheduler-state",
        required=True,
        choices=("STOPPED", "CANCELLED", "COMPLETED", "FAILED"),
        help="Exact state observed from the scheduler after stopping this job ID.",
    )
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--crop-audit-dir", type=Path, required=True)
    parser.add_argument("--best-metrics-json", type=Path, required=True)
    parser.add_argument("--best-step", type=int, default=4000)
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _tree_digest(paths: list[Path]) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_fingerprint(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _recipe_inventory(meta_path: Path) -> dict[str, Any]:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"training meta must contain exactly one dataset: {meta_path}")
    dataset_name, entry = next(iter(payload.items()))
    annotations = entry.get("annotation", [])
    if isinstance(annotations, str):
        annotations = [annotations]
    root = Path(str(entry.get("root") or meta_path.parent))
    if not root.is_absolute():
        root = meta_path.parent / root
    counts = Counter()
    task_polarity = Counter()
    unique_sources = set()
    annotation_paths = []
    for value in annotations:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = meta_path.parent / path if entry.get("paths_relative_to_meta") else root / path
        path = path.resolve(strict=True)
        annotation_paths.append(str(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                counts["records"] += 1
                kind = str(record.get("_ui5_record_kind") or "unknown")
                source = str(record.get("_ui5_crop_source") or "full_image")
                task = str(record.get("_ui5_task") or "unknown")
                answer = "\n".join(
                    str(turn.get("value", ""))
                    for turn in record.get("conversations", [])
                    if turn.get("from") == "gpt"
                )
                positive = bool(record.get("_ui5_positive", "<box><" in answer))
                counts[f"kind:{kind}"] += 1
                counts[f"source:{source}"] += 1
                task_polarity[(task, "positive" if positive else "negative")] += 1
                unique_sources.add(
                    str(
                        record.get("_ui5_image_id")
                        or record.get("_ui5_source_image")
                        or record.get("image")
                    )
                )
    return {
        "dataset_name": dataset_name,
        "annotation_files": annotation_paths,
        "records": counts.pop("records", 0),
        "record_kinds": {
            key.removeprefix("kind:"): value
            for key, value in sorted(counts.items())
            if key.startswith("kind:")
        },
        "crop_sources": {
            key.removeprefix("source:"): value
            for key, value in sorted(counts.items())
            if key.startswith("source:")
        },
        "task_positive_negative": {
            task: {
                polarity: task_polarity[(task, polarity)]
                for polarity in ("positive", "negative")
            }
            for task in sorted({task for task, _ in task_polarity})
        },
        "unique_source_images": len(unique_sources),
        "sampling": {
            "balance_ui_defects": entry.get("balance_ui_defects"),
            "ui_sampling_mode": entry.get("ui_sampling_mode", "fixed_ratio"),
            "ui_records_per_class": entry.get("ui_records_per_class"),
            "ui_negative_to_positive_ratio": entry.get("ui_negative_to_positive_ratio"),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    meta_path = args.meta_path.expanduser().resolve(strict=True)
    audit_dir = args.crop_audit_dir.expanduser().resolve(strict=True)
    best_metrics_path = args.best_metrics_json.expanduser().resolve(strict=True)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_dir / "current_run_archive_summary.json"
    )
    if output.parent != run_dir:
        raise ValueError("archive summary must be written directly inside --run-dir")
    temporary_writes = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and (".tmp" in path.name or path.name.startswith(".training_args"))
    )
    if temporary_writes:
        raise RuntimeError(
            "run directory still contains temporary checkpoint writes; wait for atomic save "
            f"completion before archiving: {temporary_writes[:20]}"
        )

    checkpoints = []
    last_complete_step = 0
    for step, path in list_training_checkpoints(run_dir):
        report = validate_checkpoint(
            path, mode="resume", expected_ranks=int(args.expected_ranks)
        )
        complete_marker = path / "checkpoint_complete.json"
        is_complete = bool(report["valid"] and complete_marker.is_file())
        if is_complete:
            last_complete_step = max(last_complete_step, step)
        checkpoints.append(
            {
                "step": step,
                "path": str(path),
                "resume_validation": report,
                "checkpoint_complete_marker": complete_marker.is_file(),
                "complete": is_complete,
            }
        )
    by_step = {row["step"]: row for row in checkpoints}
    best = by_step.get(int(args.best_step))
    if best is None or not best["complete"]:
        raise RuntimeError(
            f"best checkpoint-{args.best_step} is absent or not resumable/complete"
        )

    marker_path = audit_dir / "training_ready.json"
    recipe_summary_candidates = (
        meta_path.parent / "crop_only_recipe_summary.json",
        meta_path.parent / "recipe_summary.json",
    )
    recipe_summary_path = next(
        (path for path in recipe_summary_candidates if path.is_file()),
        recipe_summary_candidates[-1],
    )
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    recipe_summary = (
        json.loads(recipe_summary_path.read_text(encoding="utf-8"))
        if recipe_summary_path.is_file() else {}
    )
    evaluation_files = [
        path for path in (run_dir / "evaluation").rglob("*") if path.is_file()
    ] if (run_dir / "evaluation").is_dir() else []
    prediction_files = [
        path
        for prediction_root in run_dir.glob("inference-checkpoint-*")
        if prediction_root.is_dir()
        for path in prediction_root.rglob("*")
        if path.is_file()
    ]
    environment_files = [
        path for path in (run_dir / "environment").rglob("*") if path.is_file()
    ] if (run_dir / "environment").is_dir() else []
    log_files = sorted(
        {
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in {".log", ".out", ".err"}
                or "log" in path.name.lower()
            )
            and "checkpoint-" not in str(path.relative_to(run_dir)).replace("\\", "/")
        },
        key=str,
    )
    checkpoint_metadata_files = [
        path
        for row in checkpoints
        for name in (
            "training_args.bin",
            "trainer_state.json",
            "checkpoint_save_trace.jsonl",
            "checkpoint_complete.json",
        )
        if (path := Path(row["path"]) / name).is_file()
    ]
    diagnostics = run_dir / "diagnostics" / "ui5_training_evaluation.xlsx"
    if not diagnostics.is_file():
        raise FileNotFoundError(
            f"required diagnostics workbook is missing: {diagnostics}"
        )
    best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    recipe_inventory = _recipe_inventory(meta_path)
    payload = {
        "schema_version": 1,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "job_id": str(args.job_id),
        "scheduler_state": str(args.scheduler_state),
        "run_dir": str(run_dir),
        "non_destructive_archive": True,
        "stopped_by_this_script": False,
        "last_complete_step": last_complete_step,
        "best_step": int(args.best_step),
        "best_checkpoint": str(best["path"]),
        "best_raw_metrics_file": str(best_metrics_path),
        "best_raw_metrics_file_digest": content_fingerprint(best_metrics_path),
        "best_raw_metrics": best_metrics,
        "checkpoints": checkpoints,
        "code_sha": _git_sha(),
        "meta_path": str(meta_path),
        "meta_path_digest": content_fingerprint(meta_path),
        "crop_audit_dir": str(audit_dir),
        "training_ready_marker": str(marker_path),
        "training_ready_marker_digest": content_fingerprint(marker_path),
        "recipe_summary": str(recipe_summary_path) if recipe_summary_path.is_file() else "",
        "recipe_summary_digest": (
            content_fingerprint(recipe_summary_path) if recipe_summary_path.is_file() else ""
        ),
        "recipe_counts": {
            key: recipe_summary.get(key)
            for key in (
                "full_image_records",
                "crop_records",
                "ordinary_detector_crop_records",
                "gt_repair_crop_records",
                "positive_negative_records",
                "crop_only_records",
                "crop_only_region_records",
                "crop_only_content_missing_global_records",
                "crop_only_positive_negative_by_task",
                "crop_only_sampling_mode",
            )
        },
        "actual_recipe_inventory": recipe_inventory,
        "diagnostics_xlsx": str(diagnostics) if diagnostics.is_file() else "",
        "diagnostics_xlsx_digest": (
            content_fingerprint(diagnostics) if diagnostics.is_file() else ""
        ),
        "evaluation_file_count": len(evaluation_files),
        "evaluation_tree_digest": _tree_digest(evaluation_files),
        "prediction_file_count": len(prediction_files),
        "prediction_tree_digest": _tree_digest(prediction_files),
        "environment_fingerprint_file_count": len(environment_files),
        "environment_fingerprint_tree_digest": _tree_digest(environment_files),
        "training_log_file_count": len(log_files),
        "training_log_tree_digest": _tree_digest(log_files),
        "checkpoint_metadata_file_count": len(checkpoint_metadata_files),
        "checkpoint_metadata_tree_digest": _tree_digest(checkpoint_metadata_files),
        "temporary_write_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
