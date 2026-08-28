"""Compatibility files required to load local LocateAnything checkpoints."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


REQUIRED_REMOTE_CODE_FILES = (
    "configuration_locateanything.py",
    "modeling_locateanything.py",
)


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _copy_missing_file(source: Path, destination: Path) -> bool:
    """Create destination without overwriting a file from the checkpoint."""

    if _is_nonempty_file(destination):
        return False
    if destination.exists():
        destination.unlink()
    try:
        with source.open("rb") as source_handle, destination.open("xb") as output:
            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
            output.flush()
            try:
                os.fsync(output.fileno())
            except OSError:
                # Some NAS filesystems used by Merlin do not implement fsync.
                pass
        shutil.copystat(source, destination)
    except FileExistsError:
        # A concurrent evaluator repaired the same checkpoint first.
        if _is_nonempty_file(destination):
            return False
        raise
    except BaseException:
        # Never leave a partial file that a later run could mistake for valid
        # Hugging Face dynamic-module code.
        destination.unlink(missing_ok=True)
        raise
    return True


def ensure_local_checkpoint_files(
    checkpoint: str | os.PathLike[str],
    base_model: str | os.PathLike[str],
) -> dict[str, Any]:
    """Copy missing config/remote-code files from Base into a local checkpoint.

    Trainer checkpoints from the legacy CPT run contain full weights but may
    omit the Python files referenced by ``config.json:auto_map``.  Transformers
    then rejects the local directory before weight loading.  Existing files are
    intentionally preserved so a checkpoint-specific config or implementation
    is never replaced with its Base counterpart.
    """

    checkpoint_value = str(checkpoint)
    checkpoint_dir = Path(checkpoint_value).expanduser()
    if not checkpoint_dir.is_dir():
        if checkpoint_dir.is_absolute():
            raise FileNotFoundError(
                f"checkpoint directory does not exist: {checkpoint_dir}"
            )
        # A Hugging Face repository ID is not a local checkpoint to repair.
        return {
            "local_checkpoint": False,
            "checkpoint": checkpoint_value,
            "base_model": str(base_model),
            "copied": [],
        }

    checkpoint_dir = checkpoint_dir.resolve()
    base_dir = Path(str(base_model)).expanduser()
    config_missing = not _is_nonempty_file(checkpoint_dir / "config.json")
    remote_code_missing = [
        name
        for name in REQUIRED_REMOTE_CODE_FILES
        if not _is_nonempty_file(checkpoint_dir / name)
    ]
    base_python_files = sorted(base_dir.glob("*.py")) if base_dir.is_dir() else []
    dependency_files_missing = [
        source.name
        for source in base_python_files
        if not _is_nonempty_file(checkpoint_dir / source.name)
    ]
    if not config_missing and not remote_code_missing and not dependency_files_missing:
        return {
            "local_checkpoint": True,
            "checkpoint": str(checkpoint_dir),
            "base_model": str(base_dir),
            "copied": [],
        }

    if not base_dir.is_dir():
        raise FileNotFoundError(
            "checkpoint is missing config/remote-code files and the Base source "
            f"is not a local directory: checkpoint={checkpoint_dir}, base={base_dir}"
        )
    base_dir = base_dir.resolve()

    copied: list[str] = []
    base_config = base_dir / "config.json"
    if config_missing:
        if not _is_nonempty_file(base_config):
            raise FileNotFoundError(f"Base model is missing config.json: {base_dir}")
        if _copy_missing_file(base_config, checkpoint_dir / "config.json"):
            copied.append("config.json")

    base_python_files = sorted(base_dir.glob("*.py"))
    for source in base_python_files:
        destination = checkpoint_dir / source.name
        if _copy_missing_file(source, destination):
            copied.append(source.name)

    still_missing = [
        name
        for name in REQUIRED_REMOTE_CODE_FILES
        if not _is_nonempty_file(checkpoint_dir / name)
    ]
    if still_missing:
        raise FileNotFoundError(
            "Base model cannot repair LocateAnything remote code; "
            f"missing={still_missing}, base={base_dir}"
        )

    return {
        "local_checkpoint": True,
        "checkpoint": str(checkpoint_dir),
        "base_model": str(base_dir),
        "copied": copied,
    }
