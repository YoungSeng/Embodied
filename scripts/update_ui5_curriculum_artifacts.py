#!/usr/bin/env python3
"""Update UI5 curriculum metrics, best checkpoints, and the six-sheet Excel file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.ui5_curriculum_artifacts import (  # noqa: E402
    train_curve_rows_from_trainer_state,
    update_curriculum_artifacts,
)


CURRICULUM_PHASES = (
    (0.60, 0.25, 0.15, 1.0e-6),
    (0.45, 0.35, 0.20, 7.0e-7),
    (0.30, 0.30, 0.40, 5.0e-7),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def _read_rows(
    path: Path | None,
    key: str,
    *,
    expected_step: int | None = None,
    total_steps: int = 1200,
) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = _read_json(resolved)
        if isinstance(value, dict) and isinstance(value.get(key), list):
            rows = value[key]
        elif key == "train_curve" and isinstance(value, dict) and isinstance(
            value.get("log_history"), list
        ):
            return train_curve_rows_from_trainer_state(
                value,
                expected_step=expected_step,
                total_steps=total_steps,
            )
        elif isinstance(value, dict) and isinstance(value.get("rows"), list):
            rows = value["rows"]
        elif isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = [value]
        else:
            raise ValueError(f"{resolved} must contain an object or list of objects")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{resolved} contains a non-object row")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--evaluation-seconds", type=float, required=True)
    parser.add_argument("--checkpoints-json", type=Path, default=None)
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--formal-checkpoint-root", type=Path, default=None)
    parser.add_argument("--expected-ranks", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=1200)
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=int(os.environ.get("EVAL_INTERVAL_STEPS", "200")),
    )
    parser.add_argument("--train-curve-json", type=Path, default=None)
    parser.add_argument("--hard-transition-json", type=Path, default=None)
    parser.add_argument("--anchor-retention-json", type=Path, default=None)
    return parser.parse_args()


def _phase_profile(step: int, total_steps: int) -> tuple[int | str, tuple[float, ...]]:
    if total_steps <= 0 or total_steps % len(CURRICULUM_PHASES):
        raise ValueError("total_steps must be positive and divisible by three")
    if step < 0 or step > total_steps:
        raise ValueError("step must be within the curriculum horizon")
    if step == 0:
        return "baseline", CURRICULUM_PHASES[0]
    width = total_steps // len(CURRICULUM_PHASES)
    index = min((step - 1) // width, len(CURRICULUM_PHASES) - 1)
    return index + 1, CURRICULUM_PHASES[index]


def build_curriculum_status(
    result: dict[str, Any],
    *,
    train_curve_rows: list[dict[str, Any]] | None,
    total_steps: int,
    eval_interval_steps: int,
) -> dict[str, Any]:
    step = int(result["step"])
    phase, profile = _phase_profile(step, total_steps)
    candidates = [
        row
        for row in (train_curve_rows or [])
        if int(row.get("step", -1)) <= step
    ]
    latest = max(candidates, key=lambda row: int(row.get("step", -1))) if candidates else {}
    metrics = result["metrics"]
    next_step = min(total_steps, step + eval_interval_steps)
    next_action = (
        f"train_step_{step}_to_{next_step}"
        if step < total_steps
        else "finalize_pipeline"
    )
    return {
        "event": "curriculum_evaluation_registered",
        "step": step,
        "phase": phase,
        "curriculum_target": {
            "hard_ratio": profile[0],
            "anchor_ratio": profile[1],
            "global_replay_ratio": profile[2],
            "llm_lr": profile[3],
        },
        "training": {
            "log_step": latest.get("step"),
            "learning_rate": latest.get("learning_rate"),
            "loss_total": latest.get("loss_total"),
            "loss_lm": latest.get("loss_lm"),
            "grad_norm": latest.get("grad_norm"),
            "pool_samples_cumulative": {
                "hard": latest.get("hard_samples"),
                "anchor": latest.get("anchor_samples"),
                "global_replay": latest.get("global_replay_samples"),
            },
        },
        "evaluation": {
            "tasks": {
                task: {
                    "image_f1": values["image"]["f1"],
                    "bbox_f1": values["bbox"]["f1"],
                }
                for task, values in metrics["tasks"].items()
            },
            "macro": {
                "image_f1": metrics["macro"]["image"]["f1"],
                "bbox_f1": metrics["macro"]["bbox"]["f1"],
            },
            "micro": {
                "image_f1": metrics["micro"]["image"]["f1"],
                "bbox_f1": metrics["micro"]["bbox"]["f1"],
            },
            "joint_score": metrics["overall"]["joint_score"],
            "evaluation_seconds": result["evaluation_seconds"],
        },
        "checkpoint": {
            "candidate_path": result["candidate_checkpoint"],
            "improved_image": result["improved_image"],
            "improved_bbox": result["improved_bbox"],
            "improved_joint": result["improved_joint"],
            "preserved": result["checkpoint_preserved"],
            "path": (
                result["checkpoint_path"]
                if result["checkpoint_preserved"]
                else None
            ),
        },
        "best": {
            "image": result["best_image"],
            "bbox": result["best_bbox"],
            "joint": result["best_joint"],
        },
        "artifacts": {
            "checkpoints_json": result["checkpoints_json"],
            "workbook": result["workbook"],
        },
        "next_action": next_action,
    }


def _display(value: Any, *, digits: int = 8) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def print_curriculum_status(status: dict[str, Any]) -> None:
    target = status["curriculum_target"]
    training = status["training"]
    pools = training["pool_samples_cumulative"]
    print(
        f"[TRAIN SNAPSHOT] step={status['step']} phase={status['phase']} "
        f"lr={_display(training['learning_rate'])} target_lr={target['llm_lr']:.8g} "
        f"ratios=hard:{target['hard_ratio']:.2f},anchor:{target['anchor_ratio']:.2f},"
        f"global_replay:{target['global_replay_ratio']:.2f} "
        f"loss={_display(training['loss_total'])} loss_lm={_display(training['loss_lm'])} "
        f"grad_norm={_display(training['grad_norm'])} "
        f"pool_cumulative=hard:{_display(pools['hard'])},"
        f"anchor:{_display(pools['anchor'])},global_replay:{_display(pools['global_replay'])}"
    )
    for task, values in status["evaluation"]["tasks"].items():
        print(
            f"[UI5 TASK METRICS] step={status['step']} task={task} "
            f"image_f1={values['image_f1']:.8f} bbox_f1={values['bbox_f1']:.8f}"
        )
    evaluation = status["evaluation"]
    print(
        f"[UI5 AGGREGATE] step={status['step']} "
        f"image_macro_f1={evaluation['macro']['image_f1']:.8f} "
        f"bbox_macro_f1={evaluation['macro']['bbox_f1']:.8f} "
        f"image_micro_f1={evaluation['micro']['image_f1']:.8f} "
        f"bbox_micro_f1={evaluation['micro']['bbox_f1']:.8f} "
        f"joint_score={evaluation['joint_score']:.8f} "
        f"evaluation_seconds={evaluation['evaluation_seconds']:.1f}"
    )
    checkpoint = status["checkpoint"]
    print(
        f"[CHECKPOINT DECISION] step={status['step']} "
        f"improved=image:{checkpoint['improved_image']},bbox:{checkpoint['improved_bbox']},"
        f"joint:{checkpoint['improved_joint']} preserved={checkpoint['preserved']} "
        f"path={_display(checkpoint['path'])}"
    )
    print(
        "[BEST CHECKPOINTS] "
        + " ".join(
            f"{name}=step:{_display(value.get('step') if value else None)},"
            f"score:{_display(value.get('score') if value else None)},"
            f"path:{_display(value.get('checkpoint_path') if value else None)}"
            for name, value in status["best"].items()
        )
    )
    print(
        f"[NEXT ACTION] {status['next_action']} | workbook={status['artifacts']['workbook']}"
    )
    print(
        "[CURRICULUM STATUS] "
        + json.dumps(status, ensure_ascii=False, separators=(",", ":"))
    )


def main() -> int:
    args = parse_args()
    if args.eval_interval_steps <= 0:
        raise ValueError("--eval-interval-steps must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    metrics = _read_json(args.metrics_json)
    if not isinstance(metrics, dict):
        raise ValueError("--metrics-json must contain an object")
    train_curve_rows = _read_rows(
        args.train_curve_json,
        "train_curve",
        expected_step=args.step,
        total_steps=args.total_steps,
    )
    if train_curve_rows is None:
        trainer_state = args.candidate_checkpoint / "trainer_state.json"
        if trainer_state.is_file():
            train_curve_rows = train_curve_rows_from_trainer_state(
                trainer_state,
                expected_step=args.step,
                total_steps=args.total_steps,
            )
        elif args.step == 0:
            train_curve_rows = train_curve_rows_from_trainer_state(
                {"global_step": 0, "log_history": []},
                expected_step=0,
                total_steps=args.total_steps,
            )
    result = update_curriculum_artifacts(
        step=args.step,
        scorer_metrics=metrics,
        candidate_checkpoint=args.candidate_checkpoint,
        checkpoints_json=args.checkpoints_json or run_dir / "checkpoints.json",
        workbook_path=(
            args.workbook
            or run_dir
            / "diagnostics"
            / "ui5_crop_rollout4_curriculum_evaluation.xlsx"
        ),
        formal_checkpoint_root=(
            args.formal_checkpoint_root or run_dir / "checkpoints"
        ),
        resume_from=args.resume_from,
        evaluation_seconds=args.evaluation_seconds,
        train_curve_rows=train_curve_rows,
        hard_transition_rows=_read_rows(
            args.hard_transition_json, "hard_transition"
        ),
        anchor_retention_rows=_read_rows(
            args.anchor_retention_json, "anchor_retention"
        ),
        expected_ranks=args.expected_ranks,
    )
    status = build_curriculum_status(
        result,
        train_curve_rows=train_curve_rows,
        total_steps=args.total_steps,
        eval_interval_steps=args.eval_interval_steps,
    )
    result["status"] = status
    print_curriculum_status(status)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
