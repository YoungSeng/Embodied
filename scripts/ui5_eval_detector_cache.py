#!/usr/bin/env python3
"""Validation helpers for immutable, GT-free UI5 evaluation detector caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from locany_ui5_common import TASK_JSONL
from ui5_lossless_tiling import (
    build_raw_detector_edge_geometry,
    detector_task_context_bands,
    strict_vertical_partition_metrics,
)


CACHE_MARKER_SCHEMA_VERSION = 5
GEOMETRY_SCHEMA_VERSION = 5


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
    required_cache_scope: str | None = None,
    require_strict_nonoverlap: bool = False,
    require_raw_detector_edge_alignment: bool = False,
    require_detector_unique_containment: bool = False,
    expected_task_files: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every dataset/detector/geometry digest still matches."""

    cache_dir = cache_dir.expanduser().resolve(strict=False)
    if required_cache_scope in {"validation", "full_test", "full_train"} and int(expected_unique_images) <= 0:
        raise ValueError(
            f"{required_cache_scope} cache validation requires an explicit positive expected_unique_images"
        )
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
    if marker.get("strict_vertical_partition") is not True:
        raise RuntimeError("schema-v5 marker must declare strict_vertical_partition=true")
    if marker.get("raw_detector_edge_aligned") is not True:
        raise RuntimeError("schema-v5 marker must declare raw_detector_edge_aligned=true")
    if marker.get("detector_bbox_unique_containment") is not True:
        raise RuntimeError(
            "schema-v5 marker must declare detector_bbox_unique_containment=true"
        )
    cache_scope = str(marker.get("cache_scope", ""))
    if cache_scope not in {"preview", "validation", "full_test", "full_train"}:
        raise RuntimeError("detector cache marker has no valid cache_scope")
    if required_cache_scope and cache_scope != required_cache_scope:
        raise RuntimeError(
            f"detector cache scope mismatch: cache={cache_scope}, required={required_cache_scope}"
        )
    max_images_per_task = int(marker.get("max_images_per_task", -1))
    if cache_scope == "preview" and max_images_per_task <= 0:
        raise RuntimeError("preview detector cache must have max_images_per_task > 0")
    if cache_scope in {"validation", "full_test", "full_train"} and max_images_per_task != 0:
        raise RuntimeError(f"{cache_scope} detector cache requires max_images_per_task=0")

    dataset = marker.get("dataset") or {}
    unique_count = int(dataset.get("content_unique_images", -1))
    if int(marker.get("expected_unique_images", -1)) != unique_count:
        raise RuntimeError("detector cache expected_unique_images does not match its dataset")
    if expected_unique_images and unique_count != int(expected_unique_images):
        raise RuntimeError(
            f"detector cache unique image count mismatch: {unique_count} != {expected_unique_images}"
        )
    task_records = dataset.get("task_files", [])
    if not isinstance(task_records, list) or not all(
        isinstance(record, Mapping) for record in task_records
    ):
        raise RuntimeError("readonly cache task_files must be a list of file records")
    task_names = [str(record.get("task")) for record in task_records]
    expected_tasks = expected_task_files if expected_task_files is not None else TASK_JSONL
    if len(task_records) != len(expected_tasks) or set(task_names) != set(expected_tasks):
        raise RuntimeError("readonly cache must bind exactly the five UI5 task JSONL files")
    for task_record in task_records:
        task = str(task_record.get("task"))
        recorded_rows = task_record.get("jsonl_rows")
        if recorded_rows is not None:
            try:
                recorded_rows = int(recorded_rows)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"detector cache task JSONL {task} has invalid jsonl_rows"
                ) from exc
        task_path = _validate_file_record(
            cache_dir, task_record, label=f"task JSONL {task}"
        )
        actual_rows = count_jsonl(task_path)
        if expected_task_files is None and cache_scope == "full_test" and actual_rows != unique_count:
            raise RuntimeError(
                f"full_test detector cache task JSONL row count mismatch for {task}: "
                f"{actual_rows} != {unique_count}"
            )
        if input_dir is not None or expected_task_files is not None:
            expected_path = Path(expected_task_files[task]) if expected_task_files is not None else input_dir.expanduser().resolve(strict=True) / TASK_JSONL[task]
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
    geometry_files = {
        key: _validate_file_record(cache_dir, geometry[key], label=f"geometry {key}")
        for key in ("scan_manifest", "summary", "statistics", "gallery")
    }
    gallery_files = geometry.get("gallery_files") or []
    if not gallery_files:
        raise RuntimeError("horizontal scan cache has no digest-bound gallery files")
    for index, record in enumerate(gallery_files):
        _validate_file_record(cache_dir, record, label=f"geometry gallery file {index}")
    if geometry.get("gallery_digest") != json_digest(gallery_files):
        raise RuntimeError("horizontal scan gallery digest mismatch")
    if geometry.get("v4_v5_coordinate_comparison"):
        _validate_file_record(
            cache_dir,
            geometry["v4_v5_coordinate_comparison"],
            label="geometry v4/v5 coordinate comparison",
        )
    scan_manifest = _resolve_recorded_file(cache_dir, geometry["scan_manifest"])
    if count_jsonl(scan_manifest) != unique_count:
        raise RuntimeError("horizontal scan manifest count does not match cache dataset")
    if geometry.get("gate_passes") is not True:
        raise RuntimeError("horizontal scan geometry gates did not pass")
    config = geometry.get("config") or {}
    if (
        geometry.get("strict_vertical_partition") is not True
        or geometry.get("raw_detector_edge_aligned") is not True
        or geometry.get("detector_bbox_unique_containment") is not True
        or config.get("strict_vertical_partition") is not True
        or int(config.get("context_pixels", -1)) != 0
        or config.get("seam_edge_reference") != "raw_detector_bbox"
        or config.get("seam_candidates") != "safe-raw-detector-edges-only"
        or config.get("seam_selection")
        != "raw_detector_edge_dynamic_programming"
        or float(config.get("target_guard_ratio", -1)) != 0.0
        or int(config.get("target_guard_pixels_min", -1)) != 0
        or int(config.get("target_guard_pixels_max", -1)) != 0
        or scan_name != "horizontal_scan_v5_raw_detector_edge_aligned"
    ):
        raise RuntimeError(
            "schema-v5 cache is not strict and raw-detector-bbox-edge aligned"
        )
    if require_strict_nonoverlap and geometry.get("strict_vertical_partition") is not True:
        raise RuntimeError("strict non-overlap cache was required")
    if (
        require_raw_detector_edge_alignment
        and geometry.get("raw_detector_edge_aligned") is not True
    ):
        raise RuntimeError("raw detector bbox-edge alignment was required")
    if (
        require_detector_unique_containment
        and geometry.get("detector_bbox_unique_containment") is not True
    ):
        raise RuntimeError("detector bbox unique containment was required")
    summary = json.loads(geometry_files["summary"].read_text(encoding="utf-8"))
    if (
        summary.get("cache_scope") != cache_scope
        or summary.get("scan_name") != scan_name
        or summary.get("gt_used") is not False
    ):
        raise RuntimeError("horizontal scan summary identity/scope declaration is invalid")
    gate = summary.get("geometry_gate") or {}
    if gate.get("passes") is not True or not all(
        value is True for value in (gate.get("conditions") or {}).values()
    ):
        raise RuntimeError("horizontal scan summary does not contain a passing hard gate")
    scan_rows = read_jsonl(scan_manifest)
    raw_edge_records: list[dict[str, Any]] = []
    for row in scan_rows:
        if row.get("geometry_config_digest") != geometry.get("config_digest"):
            raise RuntimeError(
                f"scan row geometry config digest mismatch: {row.get('image_id')}"
            )
        metrics = strict_vertical_partition_metrics(
            int(row["width"]), int(row["height"]), row["tiles"]
        )
        if not metrics["strict_vertical_partition"]:
            raise RuntimeError(f"scan row is not a strict partition: {row.get('image_id')}")
        for key in (
            "adjacent_overlap_pixels_total",
            "adjacent_gap_pixels_total",
            "duplicate_pixel_area",
        ):
            if int(metrics[key]) != 0 or int(row.get(key, -1)) != 0:
                raise RuntimeError(f"scan row violates {key}: {row.get('image_id')}")
        for key in ("sum_tile_area", "union_tile_area", "original_area"):
            if int(row.get(key, -1)) != int(metrics[key]):
                raise RuntimeError(
                    f"scan row recorded {key} does not match recomputation: {row.get('image_id')}"
                )
        if not (
            metrics["sum_tile_area"]
            == metrics["union_tile_area"]
            == metrics["original_area"]
        ):
            raise RuntimeError(f"scan row area identity failed: {row.get('image_id')}")
        if (
            float(metrics["processed_pixel_ratio"]) != 1.0
            or float(row.get("processed_pixel_ratio", -1)) != 1.0
            or float(row.get("lossless_pixel_coverage_ratio", -1)) != 1.0
        ):
            raise RuntimeError(f"scan row processed_pixel_ratio is not 1: {row.get('image_id')}")
        zero_fields = (
            "seam_crossed_detector_bbox_count",
            "detector_boundary_cut_count",
            "uncontained_detector_bbox_count",
            "non_raw_edge_seam_count",
            "full_tile_in_multi_plan_count",
            "duplicate_tile_count",
            "nested_tile_count",
            "balanced_fallback_seam_count",
        )
        for key in zero_fields:
            if int(row.get(key, -1)) != 0:
                raise RuntimeError(f"scan row violates {key}: {row.get('image_id')}")
        detector_count = int(row.get("detector_box_count", -1))
        if (
            detector_count < 0
            or int(row.get("detector_bbox_contained_count", -1)) != detector_count
            or int(row.get("detector_bbox_unique_containment_count", -1)) != detector_count
            or float(row.get("detector_bbox_containment_rate", -1)) != 1.0
            or float(row.get("detector_bbox_unique_containment_rate", -1)) != 1.0
        ):
            raise RuntimeError(
                f"scan row does not uniquely contain every detector bbox: {row.get('image_id')}"
            )
        detector_items = [
            {**item, "source": source}
            for source, key in (("text", "text_detections"), ("icon", "icon_detections"))
            for item in row.get(key, [])
        ]
        raw_geometry = build_raw_detector_edge_geometry(
            int(row["width"]), int(row["height"]), detector_items
        )
        if (
            float(row.get("target_guard_ratio", -1)) != 0.0
            or int(row.get("target_guard_pixels_min", -1)) != 0
            or int(row.get("target_guard_pixels_max", -1)) != 0
            or int(row.get("target_guard_pixels_effective", -1)) != 0
            or row.get("seam_edge_reference") != "raw_detector_bbox"
        ):
            raise RuntimeError(f"raw-edge mode has nonzero guard: {row.get('image_id')}")
        for row_key, geometry_key in (
            ("raw_detector_edge_candidates", "raw_edge_candidates"),
            ("safe_raw_detector_edge_candidates", "safe_raw_edge_candidates"),
            ("unsafe_raw_detector_edge_candidates", "unsafe_raw_edge_candidates"),
        ):
            if row.get(row_key) != raw_geometry[geometry_key]:
                raise RuntimeError(
                    f"raw detector edge candidates changed ({row_key}): {row.get('image_id')}"
                )
        if int(row.get("raw_detector_edge_candidate_count", -1)) != len(
            raw_geometry["raw_edge_candidates"]
        ) or int(row.get("safe_raw_detector_edge_candidate_count", -1)) != len(
            raw_geometry["safe_raw_edge_candidates"]
        ):
            raise RuntimeError(f"raw detector edge count changed: {row.get('image_id')}")
        seams = [int(value) for value in row.get("horizontal_seams", [])]
        if config.get("task_context_policy"):
            if config["task_context_policy"] != "ui14_detector_neighbors_v1" or row.get("tasks") != config.get("task_keys"):
                raise RuntimeError("Task context policy/cache routing mismatch")
            bands = detector_task_context_bands(raw_geometry["records"], int(row["width"]), row["tasks"][0])
            allowed = [edge for edge in raw_geometry["safe_raw_edge_candidates"]
                       if not any(top < edge < bottom for top, bottom in bands)]
            if row.get("protected_vertical_bands") != bands or row.get("context_safe_raw_detector_edge_candidates") != allowed or any(seam not in allowed for seam in seams):
                raise RuntimeError("Crop splits detector-defined task context")
        if any(seam not in raw_geometry["safe_raw_edge_candidates"] for seam in seams):
            raise RuntimeError(f"scan row contains a non-safe-raw-edge seam: {row.get('image_id')}")
        distances = [
            min(
                (abs(seam - edge) for edge in raw_geometry["raw_edge_candidates"]),
                default=int(row["height"]),
            )
            for seam in seams
        ]
        if (
            row.get("every_seam_is_raw_detector_edge") is not True
            or distances
            != row.get("seam_nearest_raw_detector_edge_distance_pixels")
            or max(distances, default=0) != 0
        ):
            raise RuntimeError(f"seam/raw-edge alignment failed: {row.get('image_id')}")
        provenance = row.get("seam_raw_edge_provenance", [])
        if len(provenance) != len(seams):
            raise RuntimeError(f"seam provenance count mismatch: {row.get('image_id')}")
        for seam, item in zip(seams, provenance):
            expected = raw_geometry["edge_provenance"][seam]
            if any(item.get(key) != expected.get(key) for key in expected):
                raise RuntimeError(f"seam provenance changed: {row.get('image_id')}")
        raw_unique = sum(
            sum(
                int(tile[0]) <= int(box[0])
                and int(tile[1]) <= int(box[1])
                and int(tile[2]) >= int(box[2])
                and int(tile[3]) >= int(box[3])
                for tile in row["tiles"]
            )
            == 1
            for box in raw_geometry["raw_boxes"]
        )
        if raw_unique != len(raw_geometry["raw_boxes"]):
            raise RuntimeError(
                f"raw detector bbox unique containment failed: {row.get('image_id')}"
            )
        raw_edge_records.append(
            {
                "image_id": row["image_id"],
                "raw_edges": row["raw_detector_edge_candidates"],
                "safe_raw_edges": row["safe_raw_detector_edge_candidates"],
                "unsafe_raw_edges": row["unsafe_raw_detector_edge_candidates"],
            }
        )
        if row.get("gt_used") is not False:
            raise RuntimeError(f"scan row must declare gt_used=false: {row.get('image_id')}")
    raw_detector_edge_digest = json_digest(raw_edge_records)
    if (
        geometry.get("raw_detector_edge_digest") != raw_detector_edge_digest
        or summary.get("raw_detector_edge_digest") != raw_detector_edge_digest
    ):
        raise RuntimeError("raw detector edge digest mismatch")
    geometry_state_digest = json_digest(
        {
            "config": config,
            "merged_detection_sha256": sha256_file(merged_path),
            "raw_detector_edge_digest": raw_detector_edge_digest,
        }
    )
    if (
        geometry.get("geometry_state_digest") != geometry_state_digest
        or summary.get("geometry_state_digest") != geometry_state_digest
    ):
        raise RuntimeError("geometry-state digest mismatch")
    return marker
