#!/usr/bin/env python3
"""Validate, locate, and safely clean UI5 training checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eaglevl.train.ui5_checkpoint_utils import (  # noqa: E402
    list_checkpoints,
    list_training_checkpoints,
    safe_remove_checkpoint,
    validate_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--mode", choices=("eval", "resume"), default="eval")
    validate.add_argument("--expected-ranks", type=int, default=None)

    latest = subparsers.add_parser("latest")
    latest.add_argument("--output-dir", type=Path, required=True)
    latest.add_argument("--require-resume", action="store_true")
    latest.add_argument("--expected-ranks", type=int, default=None)
    latest.add_argument("--field", choices=("json", "path", "step"), default="json")

    training_candidates = subparsers.add_parser("training-candidates")
    training_candidates.add_argument("--output-dir", type=Path, required=True)
    training_candidates.add_argument(
        "--field", choices=("json", "count", "paths"), default="json"
    )

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--output-dir", type=Path, required=True)
    cleanup.add_argument("--formal-interval", type=int, required=True)
    cleanup.add_argument("--latest-step", type=int, required=True)
    cleanup.add_argument("--expected-ranks", type=int, default=None)
    cleanup.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        report = validate_checkpoint(
            args.checkpoint, mode=args.mode, expected_ranks=args.expected_ranks
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 1

    if args.command == "latest":
        candidates = list_checkpoints(args.output_dir)
        selected: tuple[int, Path] | None = None
        if candidates:
            step, path = candidates[-1]
            if not args.require_resume or validate_checkpoint(
                path, mode="resume", expected_ranks=args.expected_ranks
            )["valid"]:
                selected = (step, path)
        payload: dict[str, Any] = {
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "found": selected is not None,
            "step": selected[0] if selected else 0,
            "path": str(selected[1]) if selected else "",
        }
        if args.field == "path":
            print(payload["path"])
        elif args.field == "step":
            print(payload["step"])
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.command == "training-candidates":
        candidates = list_training_checkpoints(args.output_dir)
        payload = {
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "count": len(candidates),
            "steps": [step for step, _ in candidates],
            "paths": [str(path) for _, path in candidates],
        }
        if args.field == "count":
            print(payload["count"])
        elif args.field == "paths":
            print("\n".join(payload["paths"]))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.formal_interval <= 0:
        raise ValueError("--formal-interval must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    latest_path = output_dir / f"checkpoint-{args.latest_step}"
    latest_report = validate_checkpoint(
        latest_path,
        mode="resume",
        expected_ranks=args.expected_ranks,
    )
    if not latest_report["valid"]:
        raise RuntimeError(
            "Refusing cleanup because the latest checkpoint is not resumable: "
            + "; ".join(latest_report["errors"])
        )
    removed: list[str] = []
    kept: list[str] = []
    for step, path in list_checkpoints(output_dir):
        should_keep = step == args.latest_step or step % args.formal_interval == 0
        if should_keep or step > args.latest_step:
            kept.append(str(path))
            continue
        removed.append(str(path))
        if not args.dry_run:
            safe_remove_checkpoint(path, output_dir)
    print(
        json.dumps(
            {"removed": removed, "kept": kept, "dry_run": args.dry_run},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
