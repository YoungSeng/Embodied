#!/usr/bin/env python3
"""Freeze one or more UI5 rollout snapshots/selections for the next training run.

This is deliberately a CPU-only, pre-training operation.  It never mutates an
input and never updates an existing output directory.  A running curriculum
therefore keeps using the recipe it resolved at startup; newly arrived three-
hour snapshots can only enter a later run through a newly frozen bundle.

The three-hour snapshots emitted by snapshot_ui5_train_rollouts.py are
cumulative, so the latest completed atomic snapshot from one rollout job is
normally sufficient.  Repeated --input values are intended for distinct jobs,
shards, or an earlier frozen bundle; identical cumulative rows are harmlessly
deduplicated and any same-sample disagreement is rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 4
TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)
MODELS = ("m31", "crop")
ROLLOUT_IDS = (0, 1, 2, 3)
FORMAL_SEEDS = (20260903, 20260917, 20260931, 20260947)
DIFFICULTY_FIELDS = (
    "record_id",
    "sample_id",
    "source_image_id",
    "task",
    "image_relpath",
    "m31_correct_count",
    "crop_correct_count",
    "total_correct_count",
    "success_rate",
    "difficulty",
    "m31_complete4",
    "crop_complete4",
    "cross_model_complete8",
    "grpo_ready_m31",
    "grpo_ready_crop",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help=(
            "Completed snapshot directory or stable selection directory. "
            "Three-hour snapshots are cumulative: normally pass only the latest "
            "completed snapshot per job; repeat for distinct jobs/shards."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New immutable output directory; an existing path is rejected.",
    )
    return parser.parse_args(argv)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _loads(payload: str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON in {label}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = _loads(path.read_text(encoding="utf-8"), label=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = _loads(line, label=f"{path}:{line_no}")
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _canonical(value: Any) -> str:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite value is forbidden")
        if isinstance(item, Mapping):
            for nested in item.values():
                check(nested)
        elif isinstance(item, list):
            for nested in item:
                check(nested)

    check(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _relative_file(root: Path, raw: Any) -> tuple[str, Path]:
    value = str(raw or "")
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in value
    ):
        raise ValueError(f"unsafe snapshot manifest path: {value!r}")
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"snapshot manifest path escapes source: {value!r}")
    if not candidate.is_file():
        raise ValueError(f"snapshot manifest path is not a file: {candidate}")
    return relative.as_posix(), candidate


def _validate_snapshot(source: Path) -> dict[str, Any]:
    manifest_path = source / "manifest.json"
    success_path = source / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise ValueError(
            f"snapshot requires both manifest.json and _SUCCESS: {source}"
        )
    manifest_signature = _signature(manifest_path)
    success_signature = _signature(success_path)
    if not success_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"empty snapshot success marker: {success_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema at {source}")
    if manifest.get("atomic_publish") is not True or manifest.get("append_only") is not True:
        raise ValueError(f"snapshot is not declared atomic and append-only: {source}")
    if manifest.get("success_marker") != "_SUCCESS":
        raise ValueError(f"unexpected snapshot success marker contract: {source}")
    kind = manifest.get("snapshot_kind")
    if kind == "hourly":
        hour = manifest.get("scheduled_hour")
        if isinstance(hour, bool) or not isinstance(hour, int) or hour < 3 or hour % 3:
            raise ValueError(f"snapshot is not a valid three-hour boundary: {source}")
    elif kind != "final":
        raise ValueError(f"unsupported snapshot kind at {source}: {kind!r}")

    declared = manifest.get("files")
    if not isinstance(declared, list):
        raise ValueError(f"snapshot manifest has no file inventory: {source}")
    expected_signatures: dict[str, dict[str, Any]] = {}
    for item in declared:
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid snapshot file inventory entry: {source}")
        relative, path = _relative_file(source, item.get("path"))
        if relative in expected_signatures:
            raise ValueError(f"duplicate snapshot file inventory path: {relative}")
        signature = _signature(path)
        if item.get("bytes") != signature["bytes"] or item.get("sha256") != signature["sha256"]:
            raise ValueError(f"snapshot file inventory mismatch: {path}")
        if path.suffix == ".jsonl":
            count = item.get("jsonl_records")
            if isinstance(count, bool) or not isinstance(count, int) or count != _row_count(path):
                raise ValueError(f"snapshot JSONL row-count mismatch: {path}")
        elif item.get("jsonl_records") is not None:
            raise ValueError(f"non-JSONL snapshot file declares row count: {path}")
        expected_signatures[relative] = signature

    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "_SUCCESS"}
    }
    if actual != set(expected_signatures):
        missing = sorted(set(expected_signatures) - actual)
        extra = sorted(actual - set(expected_signatures))
        raise ValueError(
            f"snapshot file inventory is not exact: missing={missing}, extra={extra}"
        )
    if "complete8.jsonl" not in actual or "sample_difficulty.jsonl" not in actual:
        raise ValueError(f"snapshot lacks required selection exports: {source}")

    for relative, expected in expected_signatures.items():
        if _signature(source / Path(*PurePosixPath(relative).parts)) != expected:
            raise RuntimeError(f"snapshot changed during verification: {source / relative}")
    if _signature(manifest_path) != manifest_signature or _signature(success_path) != success_signature:
        raise RuntimeError(f"snapshot publication metadata changed during verification: {source}")
    return {
        "source_kind": "snapshot",
        "snapshot_kind": kind,
        "scheduled_hour": manifest.get("scheduled_hour"),
        "manifest_sha256": manifest_signature["sha256"],
        "success_marker_sha256": success_signature["sha256"],
    }


def _validate_selection(source: Path) -> dict[str, Any]:
    marker_state = ((source / "manifest.json").exists(), (source / "_SUCCESS").exists())
    if any(marker_state):
        raise ValueError(
            f"partial snapshot publication metadata is forbidden: {source}"
        )
    paths = (source / "complete8.jsonl", source / "sample_difficulty.jsonl")
    if not all(path.is_file() for path in paths):
        raise ValueError(f"selection requires complete8.jsonl and sample_difficulty.jsonl: {source}")
    before = {path.name: _signature(path) for path in paths}
    # Reading happens later; a second signature pass in _load_source closes the window.
    return {"source_kind": "selection", "stable_signatures": before}


def _validate_frozen(source: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = source / "manifest.json"
    success_path = source / "_SUCCESS"
    manifest_signature = _signature(manifest_path)
    success_signature = _signature(success_path)
    if not success_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"empty frozen selection success marker: {success_path}")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != "ui5_frozen_rollout_selection"
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("success_marker") != "_SUCCESS"
        or manifest.get("training_input_policy")
        != "resolve_once_at_run_start_no_hot_reload"
        or manifest.get("technical_policy")
        != "complete8_and_error_free_routes_only"
    ):
        raise ValueError(f"invalid frozen selection manifest contract: {source}")
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"frozen selection manifest lacks sources: {source}")
    expected_source_set = hashlib.sha256(_canonical(sources).encode("utf-8")).hexdigest()
    if manifest.get("source_set_sha256") != expected_source_set:
        raise ValueError(f"frozen selection source-set digest mismatch: {source}")

    declared = manifest.get("files")
    if not isinstance(declared, list):
        raise ValueError(f"frozen selection manifest lacks file inventory: {source}")
    expected_signatures: dict[str, dict[str, Any]] = {}
    for item in declared:
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid frozen selection file inventory entry: {source}")
        relative, path = _relative_file(source, item.get("path"))
        if relative in expected_signatures:
            raise ValueError(f"duplicate frozen selection file path: {relative}")
        signature = _signature(path)
        if item.get("bytes") != signature["bytes"] or item.get("sha256") != signature["sha256"]:
            raise ValueError(f"frozen selection file inventory mismatch: {path}")
        expected_count = _row_count(path) if path.suffix == ".jsonl" else None
        if item.get("jsonl_records") != expected_count:
            raise ValueError(f"frozen selection row-count inventory mismatch: {path}")
        expected_signatures[relative] = signature
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "_SUCCESS"}
    }
    if actual != set(expected_signatures):
        raise ValueError(f"frozen selection file inventory is not exact: {source}")
    if "complete8.jsonl" not in actual or "sample_difficulty.jsonl" not in actual:
        raise ValueError(f"frozen selection lacks required exports: {source}")
    for relative, expected in expected_signatures.items():
        path = source / Path(*PurePosixPath(relative).parts)
        if _signature(path) != expected:
            raise RuntimeError(f"frozen selection changed during verification: {path}")
    if _signature(manifest_path) != manifest_signature or _signature(success_path) != success_signature:
        raise RuntimeError(f"frozen selection publication metadata changed: {source}")
    return {
        "source_kind": "frozen_selection",
        "manifest_sha256": manifest_signature["sha256"],
        "success_marker_sha256": success_signature["sha256"],
        "inherited_source_set_sha256": expected_source_set,
    }


def _integer(row: Mapping[str, Any], key: str, *, sample_id: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"sample {sample_id} has invalid {key}: {value!r}")
    return value


def _is_empty_issues(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(MODELS):
        return False
    return all(isinstance(value[model], Mapping) and not value[model] for model in MODELS)


def _validate_complete8_row(row: Mapping[str, Any], *, source: Path) -> tuple[str, str]:
    record_id = row.get("record_id")
    sample_id = row.get("sample_id")
    if not isinstance(record_id, str) or not record_id or not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"complete8 row lacks string record_id/sample_id: {source}")
    task = row.get("task")
    if task not in TASKS:
        raise ValueError(f"sample {sample_id} has invalid task: {task!r}")
    for key in ("source_image_id", "image_relpath", "prompt"):
        if not isinstance(row.get(key), str) or not row.get(key):
            raise ValueError(f"sample {sample_id} lacks {key}")
    image_relpath = PurePosixPath(str(row["image_relpath"]))
    if image_relpath.is_absolute() or ".." in image_relpath.parts or "\\" in str(row["image_relpath"]):
        raise ValueError(f"sample {sample_id} has unsafe image_relpath")
    if not isinstance(row.get("gt_global"), list):
        raise ValueError(f"sample {sample_id} has invalid gt_global")
    for key in ("m31_complete4", "crop_complete4", "cross_model_complete8", "technical_error_free"):
        if row.get(key) is not True:
            raise ValueError(f"sample {sample_id} is not technically complete: {key}")
    for key, expected in (
        ("m31_completed_rollout_count", 4),
        ("crop_completed_rollout_count", 4),
        ("completed_rollout_count", 8),
        ("runtime_error_count", 0),
        ("parse_error_count", 0),
    ):
        if _integer(row, key, sample_id=sample_id) != expected:
            raise ValueError(f"sample {sample_id} has invalid {key}")
    for key in ("grpo_parse_clean_m31", "grpo_parse_clean_crop"):
        if row.get(key) is not True:
            raise ValueError(f"sample {sample_id} is not parse-clean: {key}")
    if row.get("runtime_errors") != [] or row.get("exclusion_reason") is not None:
        raise ValueError(f"sample {sample_id} contains a technical exclusion")
    if not _is_empty_issues(row.get("technical_issues")):
        raise ValueError(f"sample {sample_id} contains technical issues")

    rollouts = row.get("rollouts")
    if not isinstance(rollouts, Mapping) or set(rollouts) != set(MODELS):
        raise ValueError(f"sample {sample_id} lacks the two rollout groups")
    correct_counts: dict[str, int] = {}
    for model in MODELS:
        group = rollouts[model]
        if not isinstance(group, list) or len(group) != 4:
            raise ValueError(f"sample {sample_id} {model} group is not rollout4")
        seen: set[int] = set()
        correct = 0
        for route in group:
            if not isinstance(route, Mapping):
                raise ValueError(f"sample {sample_id} {model} route is invalid")
            rollout_id = route.get("rollout_id")
            if isinstance(rollout_id, bool) or not isinstance(rollout_id, int):
                raise ValueError(f"sample {sample_id} {model} has invalid rollout_id")
            if rollout_id not in ROLLOUT_IDS:
                raise ValueError(f"sample {sample_id} {model} has unknown rollout_id")
            if rollout_id in seen:
                raise ValueError(f"sample {sample_id} {model} has duplicate rollout_id")
            seen.add(rollout_id)
            if route.get("model_id") != model or route.get("seed") != FORMAL_SEEDS[rollout_id]:
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} identity mismatch")
            if route.get("status") != "completed" or route.get("runtime_error") is not None:
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} runtime failure")
            if (
                route.get("parse_status") == "parse_error"
                or not isinstance(route.get("parse_status"), str)
                or not route.get("parse_status")
            ):
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} parse failure")
            if route.get("contains_crop_parse_error") is not False or route.get("oom_final_failure") is not False:
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} technical failure")
            exact = route.get("exact_correct")
            if not isinstance(exact, bool) or route.get("reward") is not exact:
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} lacks boolean reward")
            if route.get("image_confusion") not in {"TP", "TN", "FP", "FN"}:
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} lacks confusion")
            if route.get("gt_global") != row.get("gt_global"):
                raise ValueError(f"sample {sample_id} {model}/{rollout_id} GT mismatch")
            correct += int(exact)
        if seen != set(ROLLOUT_IDS):
            raise ValueError(f"sample {sample_id} {model} rollout IDs are incomplete")
        correct_counts[model] = correct

    m31 = _integer(row, "m31_correct_count", sample_id=sample_id)
    crop = _integer(row, "crop_correct_count", sample_id=sample_id)
    total = _integer(row, "total_correct_count", sample_id=sample_id)
    if m31 != correct_counts["m31"] or crop != correct_counts["crop"] or total != m31 + crop:
        raise ValueError(f"sample {sample_id} correctness counts do not match rollouts")
    expected_difficulty = "easy" if total == 8 else "hard" if total == 0 else "medium"
    if row.get("difficulty") != expected_difficulty:
        raise ValueError(f"sample {sample_id} has inconsistent difficulty")
    success_rate = row.get("success_rate")
    if isinstance(success_rate, bool) or not isinstance(success_rate, (int, float)) or not math.isclose(
        float(success_rate), total / 8.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"sample {sample_id} has inconsistent success_rate")
    _canonical(row)
    return sample_id, record_id


def _difficulty_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in DIFFICULTY_FIELDS}


def _indexed(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} row lacks string sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _load_source(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    has_manifest = (source / "manifest.json").exists()
    has_success = (source / "_SUCCESS").exists()
    if has_manifest != has_success:
        raise ValueError(f"partial publication metadata is forbidden: {source}")
    if has_manifest:
        published_manifest = _read_json(source / "manifest.json")
        metadata = (
            _validate_frozen(source, published_manifest)
            if published_manifest.get("artifact_type")
            == "ui5_frozen_rollout_selection"
            else _validate_snapshot(source)
        )
    else:
        metadata = _validate_selection(source)
    complete_path = source / "complete8.jsonl"
    projected_path = source / "sample_difficulty.jsonl"
    complete_signature = _signature(complete_path)
    projected_signature = _signature(projected_path)
    complete_rows = _read_jsonl(complete_path)
    projected_rows = _read_jsonl(projected_path)
    complete_by_id: dict[str, Mapping[str, Any]] = {}
    record_ids: dict[str, str] = {}
    for row in complete_rows:
        sample_id, record_id = _validate_complete8_row(row, source=source)
        if sample_id in complete_by_id:
            raise ValueError(f"duplicate complete8 sample_id in {source}: {sample_id}")
        if record_id in record_ids:
            raise ValueError(f"duplicate complete8 record_id in {source}: {record_id}")
        complete_by_id[sample_id] = row
        record_ids[record_id] = sample_id
    projected_by_id = _indexed(projected_rows, label=f"sample_difficulty in {source}")
    if set(projected_by_id) != set(complete_by_id):
        raise ValueError(f"complete8/sample_difficulty ID sets differ in {source}")
    for sample_id, row in complete_by_id.items():
        expected = _difficulty_projection(row)
        if _canonical(projected_by_id[sample_id]) != _canonical(expected):
            raise ValueError(f"sample_difficulty is not an exact projection for {sample_id}")

    if _signature(complete_path) != complete_signature or _signature(projected_path) != projected_signature:
        raise RuntimeError(f"selection files changed while being frozen: {source}")
    if metadata["source_kind"] == "selection":
        before = metadata.pop("stable_signatures")
        if before != {
            "complete8.jsonl": complete_signature,
            "sample_difficulty.jsonl": projected_signature,
        }:
            raise RuntimeError(f"selection changed before it could be frozen: {source}")
    metadata.update(
        {
            "path": str(source),
            "complete8_sha256": complete_signature["sha256"],
            "sample_difficulty_sha256": projected_signature["sha256"],
            "rows": len(complete_rows),
        }
    )
    return complete_rows, metadata


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _atomic_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DIFFICULTY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def freeze(inputs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    sources = [path.expanduser().resolve(strict=True) for path in inputs]
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate input directories are forbidden")
    for source in sources:
        if not source.is_dir():
            raise ValueError(f"input is not a directory: {source}")

    output = output_dir.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    for source in sources:
        if source == output or source in output.parents:
            raise ValueError(f"output must not be nested inside an input: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"staging path already exists: {temporary}")

    merged: dict[str, dict[str, Any]] = {}
    record_owner: dict[str, str] = {}
    provenance: dict[str, list[int]] = {}
    source_metadata: list[dict[str, Any]] = []
    input_rows = 0
    for source_index, source in enumerate(sources):
        rows, metadata = _load_source(source)
        source_metadata.append(metadata)
        input_rows += len(rows)
        for row in rows:
            sample_id = str(row["sample_id"])
            record_id = str(row["record_id"])
            owner = record_owner.get(record_id)
            if owner is not None and owner != sample_id:
                raise ValueError(
                    f"record_id conflict across inputs: {record_id} belongs to {owner} and {sample_id}"
                )
            record_owner[record_id] = sample_id
            if sample_id in merged and _canonical(merged[sample_id]) != _canonical(row):
                raise ValueError(f"sample_id conflict across inputs: {sample_id}")
            merged.setdefault(sample_id, dict(row))
            provenance.setdefault(sample_id, []).append(source_index)
    if not merged:
        raise ValueError("no technically complete rollout8 samples were found")

    ordered = sorted(
        merged.values(),
        key=lambda row: (TASKS.index(str(row["task"])), str(row["sample_id"]), str(row["record_id"])),
    )
    projected = [_difficulty_projection(row) for row in ordered]
    created_at = datetime.now(timezone.utc).isoformat()
    source_set_sha256 = hashlib.sha256(
        _canonical(source_metadata).encode("utf-8")
    ).hexdigest()
    temporary.mkdir()
    try:
        _atomic_jsonl(temporary / "complete8.jsonl", ordered)
        _atomic_jsonl(temporary / "sample_difficulty.jsonl", projected)
        _write_csv(temporary / "sample_difficulty.csv", projected)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "source_count": len(sources),
            "input_rows": input_rows,
            "unique_complete8_samples": len(ordered),
            "deduplicated_rows": input_rows - len(ordered),
            "source_set_sha256": source_set_sha256,
            "task_counts": {
                task: sum(row["task"] == task for row in ordered) for task in TASKS
            },
            "difficulty_counts": {
                difficulty: sum(row["difficulty"] == difficulty for row in ordered)
                for difficulty in ("easy", "medium", "hard")
            },
        }
        _atomic_json(temporary / "summary.json", summary)
        files = []
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json":
                continue
            files.append(
                {
                    "path": path.name,
                    **_signature(path),
                    "jsonl_records": _row_count(path) if path.suffix == ".jsonl" else None,
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "ui5_frozen_rollout_selection",
            "status": "complete",
            "created_at": created_at,
            "immutable": True,
            "success_marker": "_SUCCESS",
            "training_input_policy": "resolve_once_at_run_start_no_hot_reload",
            "technical_policy": "complete8_and_error_free_routes_only",
            "conflict_policy": "identical_sample_rows_deduplicate_otherwise_fail",
            "source_set_sha256": source_set_sha256,
            "sources": source_metadata,
            "sample_sources": {
                sample_id: indexes for sample_id, indexes in sorted(provenance.items())
            },
            "files": files,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        success = temporary / "_SUCCESS"
        with success.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(created_at + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze(args.inputs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
