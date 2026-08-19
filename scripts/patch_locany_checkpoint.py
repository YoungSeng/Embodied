#!/usr/bin/env python3
"""Idempotently patch a LocateAnything checkpoint for trust_remote_code inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


AUTO_MAP = {
    "AutoConfig": "configuration_locateanything.LocateAnythingConfig",
    "AutoModel": "modeling_locateanything.LocateAnythingForConditionalGeneration",
    "AutoModelForCausalLM": "modeling_locateanything.LocateAnythingForConditionalGeneration",
    "AutoImageProcessor": "image_processing_locateanything.LocateAnythingImageProcessor",
    "AutoProcessor": "processing_locateanything.LocateAnythingProcessor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_has_weights(checkpoint: Path) -> bool:
    direct_candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if any((checkpoint / name).is_file() for name in direct_candidates):
        return True
    return any(checkpoint.glob("model-*.safetensors")) or any(
        checkpoint.glob("pytorch_model-*.bin")
    )


def patch_checkpoint(
    *,
    base_model: Path,
    checkpoint: Path,
    project_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    base_model = base_model.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {base_model}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint}")
    if not checkpoint_has_weights(checkpoint):
        raise FileNotFoundError(f"Checkpoint is missing model weights: {checkpoint}")

    selected_sources: dict[str, Path] = {
        source.name: source for source in sorted(base_model.glob("*.py"))
    }
    inference_source = project_root / "eaglevl" / "utils" / "locany"
    if inference_source.is_dir():
        for source in sorted(inference_source.glob("*.py")):
            if source.name.startswith("__"):
                continue
            selected_sources[source.name] = source
    relation_source = (
        project_root / "eaglevl" / "model" / "locany" / "relation_modules.py"
    )
    if relation_source.is_file():
        selected_sources[relation_source.name] = relation_source
    if not selected_sources:
        raise FileNotFoundError(
            f"No LocateAnything Python sources found under {base_model} or {project_root}"
        )

    copied: list[str] = []
    skipped: list[str] = []
    stale: list[str] = []
    for name, source in sorted(selected_sources.items()):
        destination = checkpoint / name
        if destination.exists() and not force:
            skipped.append(name)
            if destination.is_file() and sha256(destination) != sha256(source):
                stale.append(name)
            continue
        shutil.copy2(source.resolve(), destination)
        copied.append(name)

    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_changed = config.get("auto_map") != AUTO_MAP
    if config_changed:
        config["auto_map"] = AUTO_MAP
        atomic_write_json(config_path, config)

    manifest = {
        "schema_version": 1,
        "base_model": str(base_model),
        "checkpoint": str(checkpoint),
        "project_root": str(project_root),
        "force": force,
        "copied": copied,
        "skipped": skipped,
        "stale_skipped": stale,
        "config_auto_map_updated": config_changed,
        "files": {
            name: {"source": str(source), "sha256": sha256(source)}
            for name, source in sorted(selected_sources.items())
        },
    }
    atomic_write_json(checkpoint / "locany_patch_manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    report = patch_checkpoint(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        project_root=args.project_root,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["stale_skipped"]:
        print(
            "[WARN] Existing files differ from the selected sources and were skipped. "
            "Use --force to overwrite them."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
