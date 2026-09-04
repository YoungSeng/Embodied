#!/usr/bin/env python3
"""Print grep-friendly, truthful status at formal curriculum segment boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.ui5_curriculum_artifacts import (  # noqa: E402
    train_curve_rows_from_trainer_state,
)


PHASES = (
    (0.60, 0.25, 0.15, 1.0e-6),
    (0.45, 0.35, 0.20, 7.0e-7),
    (0.30, 0.30, 0.40, 5.0e-7),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=("start", "complete"), required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=1200)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    if args.total_steps <= 0 or args.total_steps % len(PHASES):
        raise ValueError("--total-steps must be positive and divisible by three")
    if not 0 <= args.start_step < args.target_step <= args.total_steps:
        raise ValueError("segment must satisfy 0 <= start < target <= total")
    width = args.total_steps // len(PHASES)
    phase_index = min(args.start_step // width, len(PHASES) - 1)
    profile = PHASES[phase_index]
    latest: dict[str, Any] = {}
    checkpoint: str | None = None
    if args.event == "complete":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for event=complete")
        resolved = args.checkpoint.expanduser().resolve(strict=True)
        rows = train_curve_rows_from_trainer_state(
            resolved,
            expected_step=args.target_step,
            total_steps=args.total_steps,
        )
        candidates = [row for row in rows if int(row["step"]) <= args.target_step]
        latest = max(candidates, key=lambda row: int(row["step"])) if candidates else {}
        checkpoint = str(resolved)
    elif args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid for event=complete")

    pools = {
        "hard": latest.get("hard_samples"),
        "anchor": latest.get("anchor_samples"),
        "global_replay": latest.get("global_replay_samples"),
    }
    return {
        "event": f"train_segment_{args.event}",
        "step": args.start_step if args.event == "start" else args.target_step,
        "segment": {"start_step": args.start_step, "target_step": args.target_step},
        "phase": phase_index + 1,
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
            "pool_samples_cumulative": pools,
        },
        "checkpoint": checkpoint,
        "next_action": (
            f"train_to_step_{args.target_step}"
            if args.event == "start"
            else f"evaluate_step_{args.target_step}"
        ),
    }


def display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def print_status(status: dict[str, Any]) -> None:
    target = status["curriculum_target"]
    training = status["training"]
    pools = training["pool_samples_cumulative"]
    segment = status["segment"]
    print(
        f"[TRAIN SEGMENT] event={status['event']} "
        f"steps={segment['start_step']}->{segment['target_step']} "
        f"phase={status['phase']} target_lr={target['llm_lr']:.8g} "
        f"ratios=hard:{target['hard_ratio']:.2f},anchor:{target['anchor_ratio']:.2f},"
        f"global_replay:{target['global_replay_ratio']:.2f}"
    )
    print(
        f"[TRAIN METRICS] log_step={display(training['log_step'])} "
        f"lr={display(training['learning_rate'])} "
        f"loss={display(training['loss_total'])} loss_lm={display(training['loss_lm'])} "
        f"grad_norm={display(training['grad_norm'])} "
        f"pool_cumulative=hard:{display(pools['hard'])},"
        f"anchor:{display(pools['anchor'])},global_replay:{display(pools['global_replay'])}"
    )
    print(f"[NEXT ACTION] {status['next_action']}")
    print(
        "[CURRICULUM STATUS] "
        + json.dumps(status, ensure_ascii=False, separators=(",", ":"))
    )


def main(argv: list[str] | None = None) -> int:
    status = build_status(parse_args(argv))
    print_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
