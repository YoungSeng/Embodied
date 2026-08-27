#!/usr/bin/env python3
"""Validation helpers for immutable, GT-free UI5 evaluation detector caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from locany_ui5_common import TASK_JSONL


CACHE_MARKER_SCHEMA_VERSION = 2
GEOMETRY_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid cache JSONL at {path}:{line_no}") from exc
    return rows


def digest_ids(values: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def marker_path(cache_dir: Path, scan_name: str) -> Path:
    return cache_dir / scan_name / "eval_detector_cache_ready.json"


def _resolve_recorded_file(cache_dir: Path, record: Mapping[str, Any]) -> Path:
    raw = Path(str(record.get("path", "")))
    return raw if raw.is_absolute() else cache_dir / raw


def _validate_file_record(
    cache_dir: Path, record: Mapping[str, Any], *, label: str
) -> Path:
    path = _resolve_recorded_file(cache_dir, record)
    if not path.is_file():
        raise FileNotFoundError(f"detector cache {label} is missing: {path}")
    expected = str(record.get("sha256", ""))
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise RuntimeError(
            f"detector cache {label} digest mismatch: expected={expected}, actual={actual}, path={path}"
        )
    expected_lines = record.get("jsonl_rows")
    if expected_lines is not None and count_jsonl(path) != int(expected_lines):
        raise RuntimeError(
            f"detector cache {label} JSONL row count changed: {path}"
        )
    return path


def validate_eval_detector_cache(
    cache_dir: Path,
    *,
    scan_name: str,
    expected_unique_images: int = 0,
    require_ready: bool = True,
    input_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every dataset/detector/geometry digest still matches."""

    cache_dir = cache_dir.expanduser().resolve(strict=False)
    ready_path = marker_path(cache_dir, scan_name)
    if not ready_path.is_file():
        if require_ready:
            raise FileNotFoundError(
                f"readonly detector cache marker is missing: {ready_path}; "
                "build the cache before training/evaluation"
            )
        return {}
    marker = json.loads(ready_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != CACHE_MARKER_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported detector cache marker schema: {marker.get('schema_version')}"
        )
    if marker.get("scan_name") != scan_name or marker.get("ready") is not True:
        raise RuntimeError("detector cache marker is not ready for the requested scan")
    if marker.get("created_after_all_checks") is not True:
        raise RuntimeError("detector cache marker was not written after all checks")
    if marker.get("gt_used") is not False:
        raise RuntimeError("evaluation detector cache must declare gt_used=false")

    dataset = marker.get("dataset") or {}
    unique_count = int(dataset.get("content_unique_images", -1))
    if expected_unique_images and unique_count != int(expected_unique_images):
        raise RuntimeError(
            f"detector cache unique image count mismatch: {unique_count} != {expected_unique_images}"
        )
    task_records = dataset.get("task_files", [])
    if {str(record.get("task")) for record in task_records} != set(TASK_JSONL):
        raise RuntimeError("readonly cache must bind exactly the five UI5 task JSONL files")
    for task_record in task_records:
        task_path = _validate_file_record(
            cache_dir, task_record, label=f"task JSONL {task_record.get('task')}"
        )
        if input_dir is not None:
            task = str(task_record.get("task"))
            expected_path = input_dir.expanduser().resolve(strict=True) / TASK_JSONL[task]
            if task_path.resolve(strict=True) != expected_path.resolve(strict=True):
                raise RuntimeError(
                    f"readonly cache dataset path mismatch for {task}: "
                    f"cache={task_path}, evaluation={expected_path}"
                )
    unique_manifest = _validate_file_record(
        cache_dir, dataset["unique_manifest"], label="unique manifest"
    )
    _validate_file_record(cache_dir, dataset["task_manifest"], label="task manifest")
    unique_rows = read_jsonl(unique_manifest)
    if digest_ids([str(row["image_id"]) for row in unique_rows]) != dataset.get("image_id_digest"):
        raise RuntimeError("readonly cache image_id digest mismatch")
    if digest_ids([str(row["content_id"]) for row in unique_rows]) != dataset.get("content_id_digest"):
        raise RuntimeError("readonly cache content_id digest mismatch")

    detector = marker.get("detector") or {}
    config_path = _validate_file_record(cache_dir, detector["config_file"], label="detector config")
    current_detector_config = json.loads(config_path.read_text(encoding="utf-8"))
    if detector.get("config_digest") != json_digest(current_detector_config):
        raise RuntimeError("detector configuration digest mismatch")
    merged_path = _validate_file_record(
        cache_dir, detector["merged_detections"], label="merged detections"
    )
    if count_jsonl(merged_path) != unique_count:
        raise RuntimeError("merged detections count does not equal content-unique image count")
    for stage in ("text", "icon"):
        summary = detector.get(f"{stage}_stage_summary")
        if not summary:
            raise RuntimeError(f"detector cache marker has no {stage} runtime/shard summary")
        _validate_file_record(cache_dir, summary["file"], label=f"{stage} stage summary")
        if int(summary.get("images", -1)) != unique_count:
            raise RuntimeError(f"{stage} stage image count does not match cache dataset")
        shards = summary.get("shards", [])
        done_markers = summary.get("done_markers", [])
        if not shards or len(shards) != len(done_markers):
            raise RuntimeError(f"detector cache {stage} shard/done summary is incomplete")
        shard_rows = 0
        for index, record in enumerate(shards):
            _validate_file_record(cache_dir, record, label=f"{stage} shard {index}")
            shard_rows += int(record.get("jsonl_rows", 0))
        for index, record in enumerate(done_markers):
            _validate_file_record(cache_dir, record, label=f"{stage} done marker {index}")
        if shard_rows != unique_count:
            raise RuntimeError(f"detector cache {stage} shard rows do not match dataset")
        if summary.get("shard_manifest_digest") != json_digest(shards):
            raise RuntimeError(f"detector cache {stage} shard manifest digest mismatch")

    geometry = marker.get("geometry") or {}
    if geometry.get("schema_version") != GEOMETRY_SCHEMA_VERSION:
        raise RuntimeError("horizontal scan geometry schema does not match this code")
    if geometry.get("config_digest") != json_digest(geometry.get("config")):
        raise RuntimeError("horizontal scan geometry configuration digest mismatch")
    for key in ("scan_manifest", "summary", "statistics", "gallery"):
        _validate_file_record(cache_dir, geometry[key], label=f"geometry {key}")
    scan_manifest = _resolve_recorded_file(cache_dir, geometry["scan_manifest"])
    if count_jsonl(scan_manifest) != unique_count:
        raise RuntimeError("horizontal scan manifest count does not match cache dataset")
    if geometry.get("gate_passes") is not True:
        raise RuntimeError("horizontal scan geometry gates did not pass")
    return marker
