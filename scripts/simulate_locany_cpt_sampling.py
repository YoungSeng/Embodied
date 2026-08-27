#!/usr/bin/env python3
"""Offline exposure simulation for the four fixed CPT sampling policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.cpt_sampling import resolve_cpt_sampling


MODES = ("sample_equal", "hybrid", "sqrt_size", "token_balanced")


def simulate(
    recipe: dict[str, dict[str, Any]],
    data_stats: dict[str, Any],
    *,
    exposure: float,
    min_probability: float,
    max_probability: float,
) -> dict[str, Any]:
    tasks = []
    for name, meta in recipe.items():
        task = str(meta.get("cpt_task") or name.removeprefix("locany_cpt_"))
        static = data_stats.get("tasks", {}).get(task, {})
        mean_tokens = meta.get("mean_total_supervised_tokens")
        if mean_tokens is None:
            mean_tokens = static.get("lengths", {}).get("total_supervised_tokens", {}).get("mean")
        tasks.append(
            {
                "name": task,
                "rows": int(meta.get("dataset_rows") or static.get("rows") or 0),
                "groups": int(meta.get("dataset_groups") or static.get("groups") or 0),
                "mean_total_supervised_tokens": mean_tokens,
                "oversize_rate": float(static.get("oversize_rate") or 0.0),
            }
        )
    if any(task["rows"] <= 0 for task in tasks):
        raise ValueError("all tasks require positive dataset_rows from split/static statistics")
    output = {}
    for mode in MODES:
        config = resolve_cpt_sampling(
            tasks,
            mode=mode,
            min_task_prob=min_probability,
            max_task_prob=max_probability,
        )
        probabilities = {item["name"]: item["probability"] for item in config["tasks"]}
        rows = []
        for task in tasks:
            attempted = exposure * probabilities[task["name"]]
            trained = attempted * (1.0 - task["oversize_rate"])
            usable_rows = task["rows"] * (1.0 - task["oversize_rate"])
            unique_records = min(trained, usable_rows)
            supervised_tokens = trained * float(task["mean_total_supervised_tokens"] or 0.0)
            rows.append(
                {
                    **task,
                    "sampling_probability": probabilities[task["name"]],
                    "expected_attempted_samples": attempted,
                    "expected_trained_samples": trained,
                    "expected_skipped_samples": attempted - trained,
                    "expected_unique_records": unique_records,
                    "expected_row_coverage": unique_records / usable_rows if usable_rows else None,
                    "expected_effective_epoch": trained / task["rows"],
                    "expected_repeat_factor": trained / max(unique_records, 1.0),
                    "expected_supervised_tokens": supervised_tokens,
                }
            )
        total_trained = sum(row["expected_trained_samples"] for row in rows)
        total_tokens = sum(row["expected_supervised_tokens"] for row in rows)
        for row in rows:
            row["post_skip_sample_share"] = (
                row["expected_trained_samples"] / total_trained if total_trained else 0.0
            )
            row["post_skip_token_share"] = (
                row["expected_supervised_tokens"] / total_tokens if total_tokens else 0.0
            )
        output[mode] = {"config": config, "tasks": rows}
    return {
        "schema_version": 1,
        "total_sample_exposure": exposure,
        "modes": output,
        "note": "unique coverage assumes each task iterator exhausts a deterministic epoch before repeating",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=20000)
    parser.add_argument("--samples-per-step", type=float, default=23.57)
    parser.add_argument("--total-sample-exposure", type=float)
    parser.add_argument("--min-task-prob", type=float, default=0.0)
    parser.add_argument("--max-task-prob", type=float, default=1.0)
    args = parser.parse_args()
    exposure = (
        args.total_sample_exposure
        if args.total_sample_exposure is not None
        else args.optimizer_steps * args.samples_per_step
    )
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    data_stats = json.loads(args.data_stats.read_text(encoding="utf-8"))
    payload = simulate(
        recipe,
        data_stats,
        exposure=exposure,
        min_probability=args.min_task_prob,
        max_probability=args.max_task_prob,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
