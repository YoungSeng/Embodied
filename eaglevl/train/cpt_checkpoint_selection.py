"""Pure held-out CPT checkpoint selection policy."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


CRITICAL_TASKS = ("referring_kg", "ui_defect", "vqa")


def select_checkpoint(
    current_macro: Mapping[str, Any],
    current_tasks: Mapping[str, Mapping[str, Any]],
    history: Iterable[Mapping[str, Any]],
    *,
    regression_tolerance: float = 0.03,
) -> dict[str, Any]:
    current_is_complete_heldout = (
        current_macro.get("split") == "heldout"
        and current_macro.get("complete_ten_task_heldout") is True
    )
    manifest_id = current_macro.get("manifest_id")
    protocol_id = current_macro.get("evaluation_protocol_id")
    rows = [
        dict(row)
        for row in history
        if row.get("split") == "heldout"
        and (manifest_id is None or row.get("manifest_id") == manifest_id)
        and (
            protocol_id is None
            or row.get("evaluation_protocol_id") == protocol_id
        )
    ]
    macro_rows = [
        row
        for row in rows
        if row.get("task") == "__task_macro__"
        and row.get("complete_ten_task_heldout") is True
    ]
    complete_checkpoints = {row.get("checkpoint") for row in macro_rows}
    historical_task_best = {}
    for row in rows:
        task, primary = row.get("task"), row.get("primary_metric")
        if (
            task in CRITICAL_TASKS
            and row.get("checkpoint") in complete_checkpoints
            and isinstance(primary, (int, float))
        ):
            historical_task_best[task] = max(float(primary), historical_task_best.get(task, float("-inf")))
    regressions = {}
    for task in CRITICAL_TASKS:
        previous = historical_task_best.get(task)
        current = current_tasks.get(task, {}).get("primary_metric")
        if previous is not None and isinstance(current, (int, float)):
            drop = previous - float(current)
            if drop > regression_tolerance:
                regressions[task] = drop

    primary = current_macro.get("primary_metric")
    ce = current_macro.get("eval_token_ce")
    comparable = [
        row
        for row in macro_rows
        if isinstance(row.get("primary_metric"), (int, float))
    ]
    best_previous = None
    if comparable:
        best_previous = max(
            comparable,
            key=lambda row: (
                float(row["primary_metric"]),
                -float(row["eval_token_ce"])
                if isinstance(row.get("eval_token_ce"), (int, float))
                else float("-inf"),
            ),
        )
    better = isinstance(primary, (int, float))
    if better and best_previous is not None:
        previous_primary = float(best_previous["primary_metric"])
        if float(primary) > previous_primary:
            better = True
        elif float(primary) < previous_primary:
            better = False
        else:
            previous_ce = best_previous.get("eval_token_ce")
            better = (
                isinstance(ce, (int, float))
                and isinstance(previous_ce, (int, float))
                and float(ce) < float(previous_ce)
            )
    eligible = current_is_complete_heldout and better and not regressions
    return {
        "is_best_overall": bool(eligible),
        "current_is_complete_heldout": current_is_complete_heldout,
        "critical_regressions": regressions,
        "regression_tolerance": regression_tolerance,
        "previous_best_checkpoint": best_previous.get("checkpoint") if best_previous else None,
        "previous_best_primary": best_previous.get("primary_metric") if best_previous else None,
        "previous_best_eval_token_ce": best_previous.get("eval_token_ce") if best_previous else None,
    }
