#!/usr/bin/env python3
"""Apply validation-frozen image Gate thresholds and genuinely rescore image/bbox metrics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from collect_ui5_metrics import (
    apply_frozen_gate_thresholds,
    build_gate_threshold_sweep,
    collect_gate_metrics,
)
from locany_ui5_common import TASKS, image_gate_probability


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"), required=True
    )
    parser.add_argument("--frozen-gate-thresholds", type=Path, default=None)
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_thresholds(args: argparse.Namespace, gate_metrics: dict[str, Any]) -> dict[str, float]:
    if args.evaluation_split == "validation":
        if args.frozen_gate_thresholds is not None:
            raise ValueError("validation selects thresholds itself; do not pass --frozen-gate-thresholds")
        sweep = build_gate_threshold_sweep(gate_metrics)
        return {
            task: float(sweep["tasks"][task]["selected"]["threshold"])
            for task in TASKS
        }
    if args.frozen_gate_thresholds is None:
        raise ValueError("test frozen-gated scoring requires validation --frozen-gate-thresholds")
    payload = json.loads(
        args.frozen_gate_thresholds.expanduser().resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    thresholds = payload.get("thresholds", payload)
    # Reuse the strict five-task/range validation.
    apply_frozen_gate_thresholds(gate_metrics, thresholds)
    return {task: float(thresholds[task]) for task in TASKS}


def _publish_gated_predictions(
    prediction_dir: Path, gated_dir: Path, thresholds: dict[str, float]
) -> dict[str, Any]:
    if gated_dir.exists():
        raise FileExistsError(f"refusing to mix frozen-gated predictions: {gated_dir}")
    counts: dict[str, Any] = {}
    for task in TASKS:
        source_task = prediction_dir / task
        gate_dir = source_task / "gate"
        destination_task = gated_dir / task
        destination_task.mkdir(parents=True, exist_ok=False)
        prediction_files = {
            path.name: path for path in source_task.glob("*.json") if path.is_file()
        }
        gate_files = {
            path.name: path for path in gate_dir.glob("*.json") if path.is_file()
        }
        if set(prediction_files) != set(gate_files):
            raise RuntimeError(
                f"prediction/gate sidecar mismatch for {task}: "
                f"predictions={len(prediction_files)}, gates={len(gate_files)}, "
                f"missing_gate={sorted(set(prediction_files) - set(gate_files))[:5]}, "
                f"missing_prediction={sorted(set(gate_files) - set(prediction_files))[:5]}"
            )
        kept = filtered = 0
        for name in sorted(prediction_files):
            gate = json.loads(gate_files[name].read_text(encoding="utf-8"))
            probability, source = image_gate_probability(gate)
            if probability is None:
                raise RuntimeError(f"missing p_defect for {task}/{name}; source={source}")
            raw_prediction = json.loads(prediction_files[name].read_text(encoding="utf-8"))
            if not isinstance(raw_prediction, list):
                raise ValueError(f"invalid yolo prediction list: {prediction_files[name]}")
            passed = float(probability) >= thresholds[task]
            destination = destination_task / name
            if passed:
                try:
                    os.link(prediction_files[name], destination)
                except OSError:
                    shutil.copy2(prediction_files[name], destination)
                kept += 1
            else:
                _atomic_json(destination, [])
                filtered += 1
        counts[task] = {
            "records": len(prediction_files),
            "kept_by_frozen_gate": kept,
            "filtered_by_frozen_gate": filtered,
            "threshold": thresholds[task],
        }
    return counts


def build(args: argparse.Namespace) -> dict[str, Any]:
    prediction_dir = args.prediction_dir.expanduser().resolve(strict=True)
    gt_dir = args.gt_dir.expanduser().resolve(strict=True)
    scorer_root = args.scorer_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    scorer = scorer_root / "qwen3vl_merge_and_score_fixed_5tasks.py"
    if not scorer.is_file():
        raise FileNotFoundError(scorer)
    gate_metrics = collect_gate_metrics(prediction_dir, gt_dir, scorer_root)
    thresholds = _load_thresholds(args, gate_metrics)
    gated_predictions = output_dir / "predictions"
    counts = _publish_gated_predictions(prediction_dir, gated_predictions, thresholds)
    score_root = output_dir / "score"
    command = [
        sys.executable,
        str(scorer),
        "--all_tasks",
        "--input_mode",
        "yolo_dir",
        "--gt_dir",
        str(gt_dir),
        "--pred_root",
        str(gated_predictions),
        "--output_root",
        str(score_root),
        "--run_name",
        "frozen-gated",
        "--yolo_bbox_format",
        "xyxy",
        "--iou_thresh",
        "0.1",
    ]
    subprocess.run(command, cwd=scorer_root, check=True)
    metrics_path = score_root / "frozen-gated" / "all_tasks_evaluation.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    threshold_payload = {
        "schema_version": 1,
        "selected_on": "external_frozen_thresholds",
        "thresholds": thresholds,
    }
    _atomic_json(output_dir / "frozen_gate_thresholds.json", threshold_payload)
    summary = {
        "schema_version": 1,
        "evaluation_split": args.evaluation_split,
        "threshold_selection": "validation" if args.evaluation_split == "validation" else "frozen_validation",
        "thresholds": thresholds,
        "counts": counts,
        "raw_prediction_dir": str(prediction_dir),
        "gated_prediction_dir": str(gated_predictions),
        "gated_metrics_json": str(metrics_path),
        "scorer_command": command,
        "bbox_metrics_genuinely_rescored": True,
    }
    _atomic_json(output_dir / "frozen_gate_rescore.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
