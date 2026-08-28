#!/usr/bin/env python3
"""Recompute CPT summaries from predictions.jsonl without model inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.cpt_eval_metrics import (
    aggregate_scores,
    micro_primary,
    score_task,
    task_macro_primary,
)


def recompute(predictions: Path, *, iou_threshold: float = 0.1) -> dict[str, Any]:
    by_model_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: dict[str, int] = defaultdict(int)
    token_losses: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
        lambda: defaultdict(lambda: {"sum": 0.0, "tokens": 0})
    )
    split = None
    rewritten = []
    with predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row.get("model", "checkpoint"))
            task = str(row["task"])
            row_split = str(row.get("split", ""))
            if split is not None and row_split != split:
                raise ValueError(
                    f"{predictions}:{line_number}: mixed split labels {split!r}/{row_split!r}"
                )
            split = split or row_split
            metrics = score_task(
                task,
                str(row.get("prediction", "")),
                str(row["target"]),
                iou_threshold=iou_threshold,
            )
            if row.get("error"):
                metrics["evaluation_error"] = 1.0
                metrics["primary_metric"] = 0.0
                primary_name = metrics.get("primary_name")
                if isinstance(primary_name, str):
                    metrics[primary_name] = 0.0
                counts[f"{model}:errors"] += 1
            else:
                counts[f"{model}:successful"] += 1
            row["metrics"] = metrics
            by_model_task[model][task].append(metrics)
            loss_sum = row.get("teacher_forced_main_loss_sum")
            loss_tokens = row.get("teacher_forced_main_tokens")
            if isinstance(loss_sum, (int, float)) and isinstance(loss_tokens, int):
                token_losses[model][task]["sum"] += float(loss_sum)
                token_losses[model][task]["tokens"] += int(loss_tokens)
            rewritten.append(row)

    models = {}
    for model, task_values in by_model_task.items():
        per_task = {
            task: aggregate_scores(
                task,
                scores,
                iou_threshold=iou_threshold,
            )
            for task, scores in task_values.items()
        }
        for task, metrics in per_task.items():
            totals = token_losses[model][task]
            metrics["eval_main_loss_sum"] = totals["sum"] if totals["tokens"] else None
            metrics["eval_main_loss_tokens"] = totals["tokens"]
            metrics["eval_main_token_ce"] = (
                totals["sum"] / totals["tokens"] if totals["tokens"] else None
            )
        macro = task_macro_primary(per_task)
        total_loss = sum(value["sum"] for value in token_losses[model].values())
        total_tokens = sum(value["tokens"] for value in token_losses[model].values())
        models[model] = {
            "per_task": per_task,
            "micro_primary": micro_primary(per_task),
            "task_macro_primary": macro,
            "eval_main_token_ce": total_loss / total_tokens if total_tokens else None,
            "eval_main_loss_tokens": total_tokens,
            "successful": counts[f"{model}:successful"],
            "errors": counts[f"{model}:errors"],
        }
    return {
        "schema_version": 2,
        "source": str(predictions.resolve()),
        "split": split,
        "iou_threshold": iou_threshold,
        "models": models,
        "rows": rewritten,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument(
        "--rewrite-predictions",
        type=Path,
        default=None,
        help="optional JSONL with refreshed parsed metrics",
    )
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    result = recompute(
        args.predictions.expanduser().resolve(),
        iou_threshold=args.iou_threshold,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if args.rewrite_predictions:
        args.rewrite_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.rewrite_predictions.open("w", encoding="utf-8") as handle:
            for row in result["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
