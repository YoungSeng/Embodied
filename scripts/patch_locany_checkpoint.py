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
    parser.add_argument(
        "--validate-relation-weights",
        action="store_true",
        help="Fail unless checkpoint weights contain every Relation/Gate/PBD group",
    )
    parser.add_argument(
        "--allow-legacy-slot-gate",
        action="store_true",
        help="Observe-only compatibility for pre-image-gate checkpoints",
    )
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


REQUIRED_RELATION_WEIGHT_GROUPS = (
    "relation_pyramid.level_projections.",
    "relation_pyramid.scale_logits",
    "relation_pyramid.evidence_queries",
    "relation_pyramid.context_queries",
    "relation_pyramid.family_adapters.",
    "relation_pyramid.gate_heads.",
    "relation_pyramid.image_gate_heads.",
    "relation_pbd.semantic_projection.",
    "relation_pbd.box_projection.",
    "relation_pbd.semantic_scale",
    "relation_pbd.box_scale",
)


def checkpoint_weight_keys(checkpoint: Path) -> set[str]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / name
        if index_path.is_file():
            value = json.loads(index_path.read_text(encoding="utf-8"))
            return set(value.get("weight_map", {}))

    safetensor_files = list(checkpoint.glob("model*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to validate relation checkpoint keys"
            ) from exc
        keys: set[str] = set()
        for path in safetensor_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys.update(handle.keys())
        return keys

    bin_files = list(checkpoint.glob("pytorch_model*.bin"))
    if bin_files:
        import torch

        keys: set[str] = set()
        for path in bin_files:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, dict):
                keys.update(value)
        return keys
    return set()


def validate_relation_weight_keys(keys: set[str]) -> dict[str, Any]:
    missing = [
        group
        for group in REQUIRED_RELATION_WEIGHT_GROUPS
        if not any(group in key for key in keys)
    ]
    return {
        "valid": not missing,
        "missing_groups": missing,
        "relation_key_count": sum(
            "relation_pyramid" in key or "relation_pbd" in key for key in keys
        ),
    }


def validate_pbd_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the token-driven PBD selector settings saved by training."""

    text_config = config.get("text_config")
    missing: list[str] = []
    if not isinstance(text_config, dict):
        missing.extend(("text_config.block_size", "text_config.text_mask_token_id"))
        text_config = {}
    else:
        for name in ("block_size", "text_mask_token_id"):
            if name not in text_config:
                missing.append(f"text_config.{name}")
    if "box_start_token_id" not in config:
        missing.append("box_start_token_id")
    if missing:
        return {"valid": False, "missing": missing}

    try:
        block_size = int(text_config["block_size"])
        text_mask_token_id = int(text_config["text_mask_token_id"])
        box_start_token_id = int(config["box_start_token_id"])
    except (TypeError, ValueError) as exc:
        return {"valid": False, "missing": [], "error": str(exc)}
    if block_size <= 0:
        return {
            "valid": False,
            "missing": [],
            "error": f"text_config.block_size must be positive, got {block_size}",
        }
    return {
        "valid": True,
        "missing": [],
        "block_size": block_size,
        "text_mask_token_id": text_mask_token_id,
        "box_start_token_id": box_start_token_id,
    }


def patch_checkpoint(
    *,
    base_model: Path,
    checkpoint: Path,
    project_root: Path,
    force: bool = False,
    validate_relation_weights: bool = False,
    allow_legacy_slot_gate: bool = False,
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

    relation_weight_report = None
    if validate_relation_weights:
        relation_weight_report = validate_relation_weight_keys(
            checkpoint_weight_keys(checkpoint)
        )
        legacy_only = relation_weight_report["missing_groups"] == [
            "relation_pyramid.image_gate_heads."
        ]
        if not relation_weight_report["valid"] and not (
            allow_legacy_slot_gate and legacy_only
        ):
            raise RuntimeError(
                "Checkpoint is missing trained UI relation weights: "
                + ", ".join(relation_weight_report["missing_groups"])
            )

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
    original_config = dict(config)
    if allow_legacy_slot_gate and relation_weight_report is not None:
        config["ui_relation_legacy_slot_gate_as_image_gate"] = bool(
            relation_weight_report["missing_groups"]
            == ["relation_pyramid.image_gate_heads."]
        )
    if validate_relation_weights:
        if not bool(config.get("enable_ui_relation", False)):
            raise RuntimeError(
                "Checkpoint config does not enable the UI relation generation path"
            )
        if config.get("relation_detail_layers") != [5, 15, 26]:
            raise RuntimeError(
                "Checkpoint config must use MoonViT detail layers [5, 15, 26], got "
                f"{config.get('relation_detail_layers')}"
            )
        if "relation_gate_threshold" not in config:
            raise RuntimeError(
                "Checkpoint config is missing relation_gate_threshold"
            )
        pbd_config_report = validate_pbd_config(config)
        if not pbd_config_report["valid"]:
            raise RuntimeError(
                "Checkpoint config is missing or has invalid PBD selector settings: "
                + json.dumps(pbd_config_report, ensure_ascii=False, sort_keys=True)
            )
    else:
        pbd_config_report = None
    config["auto_map"] = AUTO_MAP
    config_changed = config != original_config
    if config_changed:
        atomic_write_json(config_path, config)

    # Persist only the resulting canonical state.  Invocation-local fields
    # such as ``config_changed`` or copied/skipped lists would make an
    # otherwise idempotent second patch change the checkpoint's content hash,
    # which in turn would invalidate a durable evaluation identity on resume.
    durable_manifest = {
        "schema_version": 2,
        "base_model": str(base_model),
        "checkpoint": str(checkpoint),
        "project_root": str(project_root),
        "config_auto_map_canonical": config.get("auto_map") == AUTO_MAP,
        "config_sha256": sha256(config_path),
        "relation_weight_validation": relation_weight_report,
        "pbd_config_validation": pbd_config_report,
        "files": {
            name: {"source": str(source), "sha256": sha256(source)}
            for name, source in sorted(selected_sources.items())
        },
    }
    manifest_path = checkpoint / "locany_patch_manifest.json"
    atomic_write_json(manifest_path, durable_manifest)
    return {
        **durable_manifest,
        "manifest": str(manifest_path),
        "force": force,
        "copied": copied,
        "skipped": skipped,
        "stale_skipped": stale,
        "config_auto_map_updated": config_changed,
    }


def main() -> int:
    args = parse_args()
    report = patch_checkpoint(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        project_root=args.project_root,
        force=args.force,
        validate_relation_weights=args.validate_relation_weights,
        allow_legacy_slot_gate=args.allow_legacy_slot_gate,
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
