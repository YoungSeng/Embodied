#!/usr/bin/env python3
"""CPU-only inventory and runtime preflight for UI5 train rollouts.

This module intentionally imports neither torch nor transformers and never
opens checkpoint weight payloads.  It only validates metadata, file presence,
non-zero sizes, and safetensors/bin shard indices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
TASKS = ("occlusion", "cropping", "text_overflow", "text_ellipsis", "content_missing")
A800_BUNDLE = (
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/"
    "ui5_train_rollout_bundle_v1"
)
H20_BUNDLE_PARENT = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_data"
)
H20_BUNDLE = f"{H20_BUNDLE_PARENT}/ui5_train_rollout_bundle_v1"
M31_CHECKPOINT = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/"
    "Embodied/locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/"
    "checkpoint-12000"
)
CROP_CHECKPOINT = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/"
    "Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/"
    "checkpoint-12000"
)
PROCESSOR_CANDIDATES = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/hf_home/hub/"
    "models--nvidia--LocateAnything-3B/snapshots/"
    "c32291ca5e996f5a7a485845b4f57a233936bba0",
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/cache/huggingface/hub/"
    "models--nvidia--LocateAnything-3B/snapshots/"
    "c32291ca5e996f5a7a485845b4f57a233936bba0",
)
M31_REPO = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/"
    "Embodied-ui5-rollout8-m31"
)
CROP_REPO = (
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/"
    "Embodied-rollout8-h20x2-v3"
)
M31_ROLLOUT_COMMIT = "6367cc6660f7eb933048b81100915a05f9b49bf4"
V3_BASE_COMMIT = "ff6b3b7507e762012b23c8700f832b05e606dbd4"
A800_M31_SOURCE = (
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied/work_dirs/"
    "locany-ui5-m31-taskmoe-setdecoder-a800x4-sft-20260830-r2/checkpoint-12000"
)
A800_CROP_SOURCE = (
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/"
    "locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000"
)
A800_PROCESSOR_SOURCE = (
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/"
    "models--nvidia--LocateAnything-3B/snapshots/"
    "c32291ca5e996f5a7a485845b4f57a233936bba0"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(A800_BUNDLE))
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--m31-checkpoint", type=Path, default=Path(M31_CHECKPOINT))
    parser.add_argument("--crop-checkpoint", type=Path, default=Path(CROP_CHECKPOINT))
    parser.add_argument(
        "--processor-candidate", action="append", type=Path, default=None
    )
    parser.add_argument("--m31-repo", type=Path, default=Path(M31_REPO))
    parser.add_argument("--crop-repo", type=Path, default=Path(CROP_REPO))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="fail unless both checkpoints, a processor, and both worktrees are ready",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "config_ok": False,
        "weight_index": None,
        "weight_shards": [],
        "missing_weight_shards": [],
        "zero_byte_weight_shards": [],
        "complete": False,
    }
    if not path.is_dir():
        report["errors"] = ["checkpoint directory missing"]
        return report
    errors: list[str] = []
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            report["config_ok"] = isinstance(config, dict) and bool(config)
            report["config_model_type"] = config.get("model_type") if isinstance(config, dict) else None
            auto_map = config.get("auto_map", {}) if isinstance(config, dict) else {}
            remote_modules = sorted(
                {
                    str(value).split(".", 1)[0] + ".py"
                    for value in auto_map.values()
                    if isinstance(value, str) and "." in value
                }
            )
            report["remote_code_files"] = remote_modules
            missing_remote = [name for name in remote_modules if not (path / name).is_file()]
            report["missing_remote_code_files"] = missing_remote
            if missing_remote:
                errors.append("config auto_map references missing remote-code files")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid config.json: {exc}")
    else:
        errors.append("config.json missing")
    index_candidates = (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    )
    index_path = next((candidate for candidate in index_candidates if candidate.is_file()), None)
    shards: list[Path] = []
    if index_path is not None:
        report["weight_index"] = index_path.name
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append(f"{index_path.name} has no nonempty weight_map")
            else:
                shards = [path / name for name in sorted(set(map(str, weight_map.values())))]
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {index_path.name}: {exc}")
    else:
        candidates = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
        ]
        shards = [candidate for candidate in candidates if candidate.is_file()]
        if not shards:
            errors.append("no safetensors/bin weight file or shard index")
    report["weight_shards"] = [item.name for item in shards]
    report["missing_weight_shards"] = [item.name for item in shards if not item.is_file()]
    report["zero_byte_weight_shards"] = [
        item.name for item in shards if item.is_file() and item.stat().st_size <= 0
    ]
    if report["missing_weight_shards"]:
        errors.append("weight index references missing shards")
    if report["zero_byte_weight_shards"]:
        errors.append("weight shard is zero bytes")
    if not shards:
        errors.append("weight shard list is empty")
    report["total_weight_bytes"] = sum(
        item.stat().st_size for item in shards if item.is_file()
    )
    report["errors"] = errors
    report["complete"] = bool(report["config_ok"] and not errors)
    return report


def check_processor(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "complete": False}
    if not path.is_dir():
        report["errors"] = ["processor directory missing"]
        return report
    files = {item.name for item in path.iterdir() if item.is_file() and item.stat().st_size > 0}
    config_ok = bool({"preprocessor_config.json", "processor_config.json"} & files)
    tokenizer_ok = bool(
        {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"} & files
    )
    tokenizer_config_ok = "tokenizer_config.json" in files
    errors = []
    if not config_ok:
        errors.append("processor/preprocessor config missing")
    if not tokenizer_config_ok:
        errors.append("tokenizer_config.json missing")
    if not tokenizer_ok:
        errors.append("tokenizer vocabulary/model missing")
    report.update(
        {
            "files": sorted(files),
            "processor_config_ok": config_ok,
            "tokenizer_config_ok": tokenizer_config_ok,
            "tokenizer_payload_ok": tokenizer_ok,
            "errors": errors,
            "complete": not errors,
        }
    )
    return report


def git_revision(
    repo: Path,
    *,
    expected_head: str | None = None,
    required_ancestor: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(repo),
        "exists": repo.is_dir(),
        "inference_entrypoint": (repo / "scripts" / "inference_ui_defect_locany.py").is_file(),
        "head": None,
    }
    if not repo.is_dir():
        report["complete"] = False
        return report
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        report["head"] = result.stdout.strip()
        if expected_head is not None:
            report["expected_head"] = expected_head
            report["head_matches_expected"] = report["head"] == expected_head
        if required_ancestor is not None:
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "merge-base",
                    "--is-ancestor",
                    required_ancestor,
                    "HEAD",
                ],
                check=False,
                capture_output=True,
            )
            report["required_ancestor"] = required_ancestor
            report["required_ancestor_present"] = ancestor.returncode == 0
    except (OSError, subprocess.CalledProcessError) as exc:
        report["git_error"] = str(exc)
    report["complete"] = bool(
        report["head"]
        and report["inference_entrypoint"]
        and report.get("head_matches_expected", True)
        and report.get("required_ancestor_present", True)
    )
    return report


def check_output_root(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "writable": False, "fresh": False}
    path.mkdir(parents=True, exist_ok=True)
    disallowed = sorted(
        child.name
        for child in path.iterdir()
        if child.name not in {"diagnostics", "logs"}
    )
    probe = path / f".preflight-write-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("ui5 rollout output write check\n")
            handle.flush()
            os.fsync(handle.fileno())
        report["writable"] = True
    except OSError as exc:
        report["write_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        probe.unlink(missing_ok=True)
    report["unexpected_entries"] = disallowed
    report["fresh"] = not disallowed
    report["complete"] = bool(report["writable"] and report["fresh"])
    return report


def bundle_inventory(bundle: Path) -> tuple[dict[str, Any], list[list[Any]]]:
    required = {
        "bundle_manifest": bundle / "bundle_manifest.json",
        "unique_images": bundle / "manifest" / "unique_images.jsonl",
        "source_records": bundle / "manifest" / "source_records.jsonl",
        "task_samples": bundle / "manifest" / "task_samples.jsonl",
        "crop_samples": bundle / "manifest" / "crop_samples.jsonl",
        "annotation_exclusions": bundle / "manifest" / "annotation_exclusions.jsonl",
        "base_scan_plans": bundle / "base_scan_plans.json",
        "task_aware_manifest": bundle / "task_aware_manifest.jsonl",
        "detector_digest": bundle / "manifest" / "detector_digest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        return (
            {
                "path": str(bundle),
                "exists": bundle.is_dir(),
                "complete": False,
                "missing": missing,
            },
            [["bundle", "all", "missing_required_files", len(missing)]],
        )
    manifest = json.loads(required["bundle_manifest"].read_text(encoding="utf-8"))
    unique = read_jsonl(required["unique_images"])
    source = read_jsonl(required["source_records"])
    samples = read_jsonl(required["task_samples"])
    crops = read_jsonl(required["crop_samples"])
    annotation_exclusions = read_jsonl(required["annotation_exclusions"])
    task_aware = read_jsonl(required["task_aware_manifest"])
    plans = json.loads(required["base_scan_plans"].read_text(encoding="utf-8"))
    detector = json.loads(required["detector_digest"].read_text(encoding="utf-8"))
    missing_images = [
        row["image_relpath"]
        for row in unique
        if not (bundle / str(row["image_relpath"])).is_file()
    ]
    absolute_bundle_paths: list[str] = []
    for row in unique:
        if PurePosixPath(str(row.get("image_relpath", ""))).is_absolute():
            absolute_bundle_paths.append(str(row.get("image_relpath")))
    for row in source:
        for value in (
            row.get("source_file"),
            (row.get("original_training_record") or {}).get("image"),
            (row.get("portable_training_record") or {}).get("image"),
        ):
            if value and PurePosixPath(str(value)).is_absolute():
                absolute_bundle_paths.append(str(value))
    sample_ids = [str(row["record_id"]) for row in samples]
    duplicate_sample_ids = len(sample_ids) - len(set(sample_ids))
    polarity = Counter(
        (str(row["task"]), "positive" if row.get("gt_global") else "negative")
        for row in samples
    )
    source_polarity = Counter(
        (
            str(row["task"]),
            "positive" if row.get("gt_boxes_global_xyxy") else "negative",
        )
        for row in source
    )
    m31_keys = {(str(row["source_image_id"]), str(row["task"])) for row in samples}
    crop_keys = {(str(row["source_image_id"]), str(row["task"])) for row in task_aware}
    common = m31_keys & crop_keys
    coverage = sum(bool(row.get("pipeline_coverage_failure")) for row in samples)
    coverage_by_task = Counter(
        str(row["task"])
        for row in samples
        if row.get("pipeline_coverage_failure")
    )
    coord = sum(bool(row.get("coordinate_transform_anomaly")) for row in samples)
    annotation = sum(bool(row.get("annotation_anomaly")) for row in samples)
    digest_errors: list[str] = []
    forbidden_geometry = []
    forbidden_keys = {"final_tiles", "removed_gt_crossing_seams", "manual_gt_repair"}
    for image_id, plan in plans.items():
        if not isinstance(plan, dict) or "base_tiles" not in plan or plan.get("gt_used") is not False:
            forbidden_geometry.append(f"{image_id}: missing GT-free base_tiles")
        elif forbidden_keys & set(plan):
            forbidden_geometry.append(
                f"{image_id}: forbidden keys {sorted(forbidden_keys & set(plan))}"
            )
    for relative, expected in (manifest.get("files") or {}).items():
        path = bundle / relative
        if not path.is_file():
            digest_errors.append(f"missing digest target {relative}")
        elif expected.get("sha256") and sha256_file(path) != expected["sha256"]:
            digest_errors.append(f"digest mismatch {relative}")
    rows: list[list[Any]] = [
        ["bundle", "all", "original_training_records", len(source)],
        ["bundle", "all", "rollout_samples", len(samples)],
        ["bundle", "all", "unique_images", len(unique)],
        ["bundle", "all", "runtime_crop_records", len(crops)],
        ["bundle", "all", "common_image_id_task_intersection", len(common)],
        ["bundle", "all", "pipeline_coverage_failures", coverage],
        ["bundle", "all", "coordinate_transform_anomalies", coord],
        ["bundle", "all", "annotation_anomalies", annotation],
        [
            "bundle",
            "all",
            "registered_annotation_exclusions",
            len(annotation_exclusions),
        ],
    ]
    for task in TASKS:
        rows.extend(
            [
                ["task", task, "positive", polarity[(task, "positive")]],
                ["task", task, "negative", polarity[(task, "negative")]],
                [
                    "task",
                    task,
                    "original_record_positive",
                    source_polarity[(task, "positive")],
                ],
                [
                    "task",
                    task,
                    "original_record_negative",
                    source_polarity[(task, "negative")],
                ],
                ["task", task, "pipeline_coverage_failures", coverage_by_task[task]],
                [
                    "task",
                    task,
                    "total",
                    polarity[(task, "positive")] + polarity[(task, "negative")],
                ],
            ]
        )
    report = {
        "path": str(bundle),
        "exists": True,
        "manifest_complete": manifest.get("complete") is True,
        "original_training_records": len(source),
        "rollout_samples": len(samples),
        "unique_images": len(unique),
        "runtime_crop_records": len(crops),
        "base_scan_plan_images": len(plans),
        "crop_scan_plan_cache_complete": bool(
            plans and crops and len(samples) == len(task_aware) and m31_keys == crop_keys
        ),
        "m31_image_task_keys": len(m31_keys),
        "crop_image_task_keys": len(crop_keys),
        "common_image_task_intersection": len(common),
        "pipeline_coverage_failures": coverage,
        "pipeline_coverage_failures_by_task": dict(coverage_by_task),
        "coordinate_transform_anomalies": coord,
        "annotation_anomalies": annotation,
        "registered_annotation_exclusions": len(annotation_exclusions),
        "missing_images": missing_images[:100],
        "missing_image_count": len(missing_images),
        "absolute_bundle_paths": absolute_bundle_paths[:100],
        "absolute_bundle_path_count": len(absolute_bundle_paths),
        "duplicate_sample_ids": duplicate_sample_ids,
        "digest_errors": digest_errors,
        "forbidden_geometry_errors": forbidden_geometry[:100],
        "positive_negative_by_task": {
            task: {
                "rollout_samples": {
                    "positive": polarity[(task, "positive")],
                    "negative": polarity[(task, "negative")],
                },
                "original_training_records": {
                    "positive": source_polarity[(task, "positive")],
                    "negative": source_polarity[(task, "negative")],
                },
            }
            for task in TASKS
        },
        "detector": detector,
    }
    report["complete"] = bool(
        report["manifest_complete"]
        and int(manifest.get("original_training_records", -1)) == len(source)
        and int(manifest.get("rollout_samples", -1)) == len(samples)
        and int(manifest.get("unique_images", -1)) == len(unique)
        and int(manifest.get("crop_records_runtime_only", -1)) == len(crops)
        and not missing_images
        and not absolute_bundle_paths
        and not duplicate_sample_ids
        and not digest_errors
        and not forbidden_geometry
        and len(samples) == len(task_aware)
        and m31_keys == crop_keys
    )
    return report, rows


def copy_commands(
    m31: Mapping[str, Any], crop: Mapping[str, Any], processors: Sequence[Mapping[str, Any]], bundle: Path
) -> str:
    continuation = "\\"
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        "# Checkpoints are copied only when their H20 targets are absent/incomplete.",
    ]
    if not m31.get("complete"):
        lines.extend(
            [
                f'mkdir -p "{Path(M31_CHECKPOINT).parent.as_posix()}"',
                f"nastk cp -c=32 {continuation}",
                f'  "{A800_M31_SOURCE}" {continuation}',
                f'  "{Path(M31_CHECKPOINT).parent.as_posix()}/"',
                "",
            ]
        )
    if not crop.get("complete"):
        lines.extend(
            [
                f'mkdir -p "{Path(CROP_CHECKPOINT).parent.as_posix()}"',
                f"nastk cp -c=32 {continuation}",
                f'  "{A800_CROP_SOURCE}" {continuation}',
                f'  "{Path(CROP_CHECKPOINT).parent.as_posix()}/"',
                "",
            ]
        )
    if not any(report.get("complete") for report in processors):
        processor_parent = Path(PROCESSOR_CANDIDATES[0]).parent.as_posix()
        lines.extend(
            [
                f'mkdir -p "{processor_parent}"',
                f"nastk cp -c=32 {continuation}",
                f'  "{A800_PROCESSOR_SOURCE}" {continuation}',
                f'  "{processor_parent}/"',
                "",
            ]
        )
    if bundle.as_posix() != H20_BUNDLE or not (bundle / "bundle_manifest.json").is_file():
        lines.extend(
            [
                f'mkdir -p "{H20_BUNDLE_PARENT}"',
                f"nastk cp -c=32 {continuation}",
                f'  "{A800_BUNDLE}" {continuation}',
                f'  "{H20_BUNDLE_PARENT}/"',
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    bundle = args.bundle_root.expanduser().resolve(strict=False)
    diagnostics = (
        args.diagnostics_dir.expanduser().resolve(strict=False)
        if args.diagnostics_dir is not None
        else bundle / "diagnostics"
    )
    diagnostics.mkdir(parents=True, exist_ok=True)
    try:
        data, inventory_rows = bundle_inventory(bundle)
    except Exception as exc:
        data = {
            "path": str(bundle),
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        inventory_rows = [["bundle", "all", "inventory_error", data["error"]]]
    m31 = check_checkpoint(args.m31_checkpoint.expanduser().resolve(strict=False))
    crop = check_checkpoint(args.crop_checkpoint.expanduser().resolve(strict=False))
    processor_paths = args.processor_candidate or [Path(item) for item in PROCESSOR_CANDIDATES]
    processors = [check_processor(path.expanduser().resolve(strict=False)) for path in processor_paths]
    selected_processor = next(
        (report["path"] for report in processors if report.get("complete")), None
    )
    m31_repo = git_revision(
        args.m31_repo.expanduser().resolve(strict=False),
        expected_head=M31_ROLLOUT_COMMIT,
    )
    crop_repo = git_revision(
        args.crop_repo.expanduser().resolve(strict=False),
        required_ancestor=V3_BASE_COMMIT,
    )
    output_arg = getattr(args, "output_root", None)
    output = (
        check_output_root(output_arg.expanduser().resolve(strict=False))
        if output_arg is not None
        else None
    )
    runtime_ready = bool(
        m31.get("complete")
        and crop.get("complete")
        and selected_processor
        and m31_repo.get("complete")
        and crop_repo.get("complete")
        and (output is None or output.get("complete"))
    )
    ready = bool(data.get("complete") and (runtime_ready or not args.require_runtime))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "cpu_only": True,
        "torch_imported": "torch" in sys.modules,
        "transformers_imported": "transformers" in sys.modules,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "bundle": data,
        "checkpoints": {"m31": m31, "crop": crop},
        "processor_candidates": processors,
        "selected_processor": selected_processor,
        "repositories": {"m31": m31_repo, "crop": crop_repo},
        "output": output,
        "runtime_ready": runtime_ready,
        "require_runtime": args.require_runtime,
        "ready": ready,
    }
    path_mapping = {
        "schema_version": SCHEMA_VERSION,
        "bundle": {"a800": A800_BUNDLE, "h20": H20_BUNDLE},
        "checkpoints": {
            "m31": {"a800": A800_M31_SOURCE, "h20": M31_CHECKPOINT},
            "crop": {"a800": A800_CROP_SOURCE, "h20": CROP_CHECKPOINT},
        },
        "processor_candidates_h20": list(PROCESSOR_CANDIDATES),
        "repositories_h20": {"m31": M31_REPO, "crop": CROP_REPO},
        "rollout_output_h20": (
            "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/"
            "gui_rollouts/ui5-train-rollout8-h20x2-v3-20260904"
        ),
    }
    atomic_json(diagnostics / "preflight_summary.json", summary)
    atomic_json(diagnostics / "path_mapping.json", path_mapping)
    with (diagnostics / "data_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "task", "metric", "value"])
        writer.writerows(inventory_rows)
        for model, report in (("m31", m31), ("crop", crop)):
            writer.writerow(["checkpoint", model, "complete", int(bool(report.get("complete")))])
            writer.writerow(["checkpoint", model, "weight_bytes", report.get("total_weight_bytes", 0)])
        for index, report in enumerate(processors):
            writer.writerow(["processor", str(index), "complete", int(bool(report.get("complete")))])
    commands = copy_commands(m31, crop, processors, bundle)
    atomic_text(diagnostics / "nastk_copy_commands.sh", commands)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_runtime and not ready:
        print("[COPY_COMMANDS_IF_DATA_IS_MISSING]", flush=True)
        print(commands, end="", flush=True)
    return summary, 0 if ready else 1


def main(argv: Sequence[str] | None = None) -> int:
    _, return_code = run(parse_args(argv))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
