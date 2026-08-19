#!/usr/bin/env python3
"""Validate, locate, and safely clean UI5 training checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Not a checkpoint directory name: {path.name}")
    return int(match.group(1))


def list_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    if not output_dir.is_dir():
        return []
    result = []
    for path in output_dir.iterdir():
        if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name):
            result.append((checkpoint_step(path), path.resolve()))
    return sorted(result)


def has_model_weights(checkpoint: Path) -> bool:
    names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((checkpoint / name).is_file() for name in names) or any(
        checkpoint.glob("model-*.safetensors")
    )


def validate_checkpoint(
    checkpoint: Path,
    *,
    mode: str,
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    errors: list[str] = []
    if not checkpoint.is_dir():
        errors.append("checkpoint directory does not exist")
    else:
        try:
            step = checkpoint_step(checkpoint)
        except ValueError:
            step = None
            if mode == "resume":
                errors.append("resume checkpoint name must be checkpoint-<step>")
        if not (checkpoint / "config.json").is_file():
            errors.append("missing config.json")
        if not has_model_weights(checkpoint):
            errors.append("missing model weights")
        if mode == "resume":
            trainer_state = checkpoint / "trainer_state.json"
            if not trainer_state.is_file():
                errors.append("missing trainer_state.json")
            else:
                try:
                    state = json.loads(trainer_state.read_text(encoding="utf-8"))
                    if step is not None and int(state.get("global_step", -1)) != step:
                        errors.append(
                            "trainer_state global_step does not match checkpoint directory"
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid trainer_state.json: {exc}")
            optimizer_present = (checkpoint / "optimizer.pt").is_file() or any(
                checkpoint.glob("global_step*/**/*optim_states.pt")
            )
            deepspeed_state_present = any(checkpoint.glob("global_step*/*model_states.pt"))
            if not optimizer_present and not deepspeed_state_present:
                errors.append("missing optimizer/DeepSpeed resume state")
            rank_states = list(checkpoint.glob("dataloader_state_rank*.pt"))
            legacy_rank_states = list(checkpoint.glob("*.pth"))
            if expected_ranks and max(len(rank_states), len(legacy_rank_states)) < expected_ranks:
                errors.append(
                    f"rank state count is smaller than expected: "
                    f"dataloader={len(rank_states)}, legacy={len(legacy_rank_states)}, "
                    f"expected={expected_ranks}"
                )
    return {
        "checkpoint": str(checkpoint),
        "mode": mode,
        "expected_ranks": expected_ranks,
        "valid": not errors,
        "errors": errors,
    }


def safe_remove_checkpoint(path: Path, output_dir: Path) -> None:
    resolved = path.resolve()
    root = output_dir.resolve()
    if resolved.parent != root or CHECKPOINT_PATTERN.fullmatch(resolved.name) is None:
        raise ValueError(f"Refusing to remove checkpoint outside output directory: {resolved}")
    shutil.rmtree(resolved)


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

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--output-dir", type=Path, required=True)
    cleanup.add_argument("--formal-interval", type=int, required=True)
    cleanup.add_argument("--latest-step", type=int, required=True)
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
        for step, path in reversed(candidates):
            if not args.require_resume or validate_checkpoint(
                path, mode="resume", expected_ranks=args.expected_ranks
            )["valid"]:
                selected = (step, path)
                break
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

    if args.formal_interval <= 0:
        raise ValueError("--formal-interval must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    latest_path = output_dir / f"checkpoint-{args.latest_step}"
    latest_report = validate_checkpoint(latest_path, mode="resume")
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
