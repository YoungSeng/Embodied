"""Evidence-gated CPT overfitting diagnostics over checkpoint curves."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _latest_train_at_or_before(
    rows: list[Mapping[str, Any]], task: str, step: int
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("task") == task and int(row.get("step") or -1) <= step
    ]
    return max(candidates, key=lambda row: int(row.get("step") or -1)) if candidates else None


def analyze_overfitting(
    train_rows: Iterable[Mapping[str, Any]],
    eval_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [dict(row) for row in train_rows]
    evaluation = [dict(row) for row in eval_rows]
    heldout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_pool_by_task_step = {}
    for row in evaluation:
        task = str(row.get("task", ""))
        if not task or task == "__task_macro__":
            continue
        split = row.get("split")
        step = row.get("step")
        if not isinstance(step, int):
            continue
        if split == "heldout":
            heldout[task].append(row)
        elif split == "train_pool":
            train_pool_by_task_step[(task, step)] = row

    per_task = {}
    first_events = []
    for task, raw_curve in sorted(heldout.items()):
        by_step = {int(row["step"]): row for row in raw_curve}
        curve = [by_step[step] for step in sorted(by_step)]
        points = []
        for row in curve:
            step = int(row["step"])
            train_row = _latest_train_at_or_before(train, task, step)
            points.append(
                {
                    "step": step,
                    "train_ce": train_row.get("train_main_token_ce") if train_row else None,
                    "val_ce": row.get("eval_token_ce"),
                    "heldout_primary": row.get("primary_metric"),
                    "train_pool_primary": train_pool_by_task_step.get((task, step), {}).get("primary_metric"),
                    "repeat_factor": train_row.get("repeat_factor") if train_row else None,
                }
            )
        events = []
        for index in range(2, len(points)):
            left, middle, right = points[index - 2 : index + 1]
            if all(
                isinstance(point.get(key), (int, float))
                for point in (left, middle, right)
                for key in ("train_ce", "val_ce")
            ) and (
                left["train_ce"] > middle["train_ce"] > right["train_ce"]
                and left["val_ce"] < middle["val_ce"] < right["val_ce"]
            ):
                events.append(
                    {
                        "step": right["step"],
                        "criterion": "train_ce_down_val_ce_up_two_milestones",
                    }
                )
            if all(
                isinstance(point.get(key), (int, float))
                for point in (left, middle, right)
                for key in ("heldout_primary", "train_pool_primary")
            ) and (
                left["train_pool_primary"] < middle["train_pool_primary"] < right["train_pool_primary"]
                and left["heldout_primary"] > middle["heldout_primary"] > right["heldout_primary"]
            ):
                events.append(
                    {
                        "step": right["step"],
                        "criterion": "train_pool_up_heldout_down_two_milestones",
                    }
                )
            if all(
                isinstance(point.get(key), (int, float))
                for point in (left, middle, right)
                for key in ("repeat_factor", "heldout_primary")
            ) and (
                left["repeat_factor"] < middle["repeat_factor"] < right["repeat_factor"]
                and left["heldout_primary"] >= middle["heldout_primary"] >= right["heldout_primary"]
            ):
                events.append(
                    {
                        "step": right["step"],
                        "criterion": "repeat_factor_up_heldout_non_improving",
                    }
                )
        first = min(events, key=lambda item: item["step"]) if events else None
        if first:
            first_events.append({"task": task, **first})
        per_task[task] = {
            "milestones": len(points),
            "status": (
                "overfitting_risk"
                if events
                else "insufficient_evidence"
                if len(points) < 3
                else "no_trigger"
            ),
            "first_risk": first,
            "events": events,
            "curve": points,
        }
    first_global = min(first_events, key=lambda item: item["step"]) if first_events else None
    return {
        "schema_version": 1,
        "first_overfitting_risk": first_global,
        "per_task": per_task,
        "interpretation": (
            "risk_detected"
            if first_global
            else "insufficient_heldout_curve"
            if not per_task or all(value["milestones"] < 3 for value in per_task.values())
            else "no_configured_trigger"
        ),
    }
