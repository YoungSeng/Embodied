"""Fixed, resumable sampling policies for the UI CPT mixture."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


MODE_EXPONENTS = {
    "sample_equal": (0.0, 0.0),
    "sqrt_size": (0.5, 0.0),
    "token_balanced": (0.0, 1.0),
    "hybrid": (0.5, 0.5),
}


def _bounded_probabilities(
    raw: Sequence[float], minimum: float, maximum: float
) -> list[float]:
    count = len(raw)
    if count == 0:
        raise ValueError("at least one CPT task is required")
    if minimum < 0.0 or maximum <= 0.0 or minimum > maximum:
        raise ValueError("invalid CPT probability bounds")
    if minimum * count > 1.0 + 1.0e-12 or maximum * count < 1.0 - 1.0e-12:
        raise ValueError(
            f"infeasible CPT probability bounds for {count} tasks: "
            f"min={minimum}, max={maximum}"
        )
    total = sum(raw)
    probabilities = [value / total for value in raw]
    fixed: dict[int, float] = {}
    while True:
        remaining = 1.0 - sum(fixed.values())
        free = [index for index in range(count) if index not in fixed]
        if not free:
            break
        free_weight = sum(probabilities[index] for index in free)
        projected = {
            index: remaining * probabilities[index] / free_weight
            for index in free
        }
        violations = False
        for index, value in projected.items():
            if value < minimum - 1.0e-15:
                fixed[index] = minimum
                violations = True
            elif value > maximum + 1.0e-15:
                fixed[index] = maximum
                violations = True
        if not violations:
            for index, value in projected.items():
                fixed[index] = value
            break
    output = [fixed[index] for index in range(count)]
    output[-1] += 1.0 - sum(output)
    return output


def resolve_cpt_sampling(
    tasks: Sequence[Mapping[str, Any]],
    *,
    mode: str = "sample_equal",
    size_alpha: float | None = None,
    token_beta: float | None = None,
    min_task_prob: float = 0.0,
    max_task_prob: float = 1.0,
) -> dict[str, Any]:
    if mode not in MODE_EXPONENTS:
        raise ValueError(f"unknown CPT sampling mode: {mode!r}")
    default_alpha, default_beta = MODE_EXPONENTS[mode]
    alpha = default_alpha if size_alpha is None else float(size_alpha)
    beta = default_beta if token_beta is None else float(token_beta)
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("CPT sampling exponents must be finite")

    raw = []
    normalized_tasks = []
    for task in tasks:
        name = str(task["name"])
        rows = int(task["rows"])
        mean_tokens = task.get("mean_total_supervised_tokens")
        if rows <= 0:
            raise ValueError(f"{name}: rows must be positive")
        if beta != 0.0:
            if mean_tokens is None or float(mean_tokens) <= 0.0:
                raise ValueError(
                    f"{name}: mean_total_supervised_tokens is required for token_beta={beta}"
                )
            mean_tokens = float(mean_tokens)
        else:
            mean_tokens = float(mean_tokens) if mean_tokens is not None else None
        score = rows**alpha
        if beta != 0.0:
            score *= float(mean_tokens) ** (-beta)
        if not math.isfinite(score) or score <= 0.0:
            raise ValueError(f"{name}: invalid sampling score={score}")
        raw.append(score)
        normalized_tasks.append(
            {
                "name": name,
                "rows": rows,
                "mean_total_supervised_tokens": mean_tokens,
                "raw_score": score,
            }
        )
    probabilities = _bounded_probabilities(raw, min_task_prob, max_task_prob)
    for task, probability in zip(normalized_tasks, probabilities):
        task["probability"] = probability
    config = {
        "schema_version": 1,
        "mode": mode,
        "size_alpha": alpha,
        "token_beta": beta,
        "min_task_prob": float(min_task_prob),
        "max_task_prob": float(max_task_prob),
        "tasks": normalized_tasks,
    }
    config["config_hash"] = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return config


def assert_sampling_resume_compatible(current: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    if current.get("config_hash") != saved.get("config_hash"):
        raise RuntimeError(
            "CPT sampling configuration changed across resume: "
            f"saved={saved.get('config_hash')}, current={current.get('config_hash')}"
        )
