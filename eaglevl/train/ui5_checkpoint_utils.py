"""Shared validation primitives for UI5 evaluation and resumable checkpoints."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def atomic_save_with_fsync(save_callable, obj: Any, target: str | Path, *args, **kwargs):
    """Publish a torch-style binary only after a durable non-empty temp write."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        result = save_callable(obj, temporary, *args, **kwargs)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"atomic save produced an empty file: {temporary}")
        os.replace(temporary, destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


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


def list_training_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Exclude the model-only checkpoint-0 from training-resume candidates."""

    return [(step, path) for step, path in list_checkpoints(output_dir) if step > 0]


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def has_model_weights(checkpoint: Path) -> bool:
    if _nonempty(checkpoint / "model.safetensors") or _nonempty(
        checkpoint / "pytorch_model.bin"
    ):
        return True
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = checkpoint / index_name
        if not _nonempty(index_path):
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(index.get("weight_map", {}).values())
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not shard_names:
            return False
        return all(_nonempty(checkpoint / name) for name in shard_names)
    return False


def validate_checkpoint(
    checkpoint: Path,
    *,
    mode: str,
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    if mode not in {"eval", "resume"}:
        raise ValueError(f"Unsupported checkpoint validation mode: {mode}")
    checkpoint = checkpoint.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    if not checkpoint.is_dir():
        errors.append("checkpoint directory does not exist")
    else:
        try:
            step = checkpoint_step(checkpoint)
        except ValueError:
            step = None
            if mode == "resume":
                errors.append("resume checkpoint name must be checkpoint-<step>")
        if not _nonempty(checkpoint / "config.json"):
            errors.append("missing or empty config.json")
        if not has_model_weights(checkpoint):
            errors.append("missing or empty model weights")
        if mode == "resume":
            training_args = checkpoint / "training_args.bin"
            if not _nonempty(training_args):
                errors.append("missing or empty training_args.bin")

            trainer_state = checkpoint / "trainer_state.json"
            if not _nonempty(trainer_state):
                errors.append("missing or empty trainer_state.json")
            else:
                try:
                    state = json.loads(trainer_state.read_text(encoding="utf-8"))
                    if step is not None and int(state.get("global_step", -1)) != step:
                        errors.append(
                            "trainer_state global_step does not match checkpoint directory"
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid trainer_state.json: {exc}")

            optimizer_files = [
                path
                for path in checkpoint.glob("global_step*/**/*optim_states.pt")
                if _nonempty(path)
            ]
            optimizer_pt = checkpoint / "optimizer.pt"
            if not _nonempty(optimizer_pt) and not optimizer_files:
                errors.append("missing optimizer/DeepSpeed optimizer state")
            if (
                expected_ranks
                and not _nonempty(optimizer_pt)
                and len(optimizer_files) < expected_ranks
            ):
                errors.append(
                    "DeepSpeed optimizer state count is smaller than expected: "
                    f"found={len(optimizer_files)}, expected={expected_ranks}"
                )

            model_state_files = [
                path
                for path in checkpoint.glob("global_step*/**/*model_states.pt")
                if _nonempty(path)
            ]
            if optimizer_files and not model_state_files:
                errors.append("missing DeepSpeed model state")

            rank_states = [
                path
                for path in checkpoint.glob("dataloader_state_rank*.pt")
                if _nonempty(path)
            ]
            legacy_rank_states = [
                path for path in checkpoint.glob("*.pth") if _nonempty(path)
            ]
            if expected_ranks and max(len(rank_states), len(legacy_rank_states)) < expected_ranks:
                errors.append(
                    "rank state count is smaller than expected: "
                    f"dataloader={len(rank_states)}, legacy={len(legacy_rank_states)}, "
                    f"expected={expected_ranks}"
                )

            rng_files = [
                path for path in checkpoint.glob("rng_state*.pth") if _nonempty(path)
            ]
            if not rng_files:
                warnings.append("missing RNG state; resume would not be bitwise reproducible")
            completion = checkpoint / "checkpoint_complete.json"
            if not _nonempty(completion):
                warnings.append("missing checkpoint_complete.json (legacy or interrupted save)")
            details.update(
                {
                    "optimizer_state_files": len(optimizer_files)
                    + int(_nonempty(optimizer_pt)),
                    "deepspeed_model_state_files": len(model_state_files),
                    "dataloader_state_files": len(rank_states),
                    "legacy_rank_state_files": len(legacy_rank_states),
                    "rng_state_files": len(rng_files),
                    "training_args_size": (
                        training_args.stat().st_size if training_args.is_file() else 0
                    ),
                }
            )
    return {
        "checkpoint": str(checkpoint),
        "mode": mode,
        "expected_ranks": expected_ranks,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def safe_remove_checkpoint(path: Path, output_dir: Path) -> None:
    resolved = path.resolve()
    root = output_dir.resolve()
    if resolved.parent != root or CHECKPOINT_PATTERN.fullmatch(resolved.name) is None:
        raise ValueError(
            f"Refusing to remove checkpoint outside output directory: {resolved}"
        )
    shutil.rmtree(resolved)
