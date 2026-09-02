#!/usr/bin/env python3
"""Build the portable, image-deduplicated UI5 train-rollout bundle.

The crop geometry in the emitted bundle is deliberately reconstructed only
from the GT-free ``base_scan_plans.json``.  Training-only ``final_tiles``,
removed seams, and manual-repair geometry are never read by rollout workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = 1
TASKS = (
    "occlusion",
    "cropping",
    "text_overflow",
    "text_ellipsis",
    "content_missing",
)
DEFAULT_FULL_DATA = Path(
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3"
)
DEFAULT_AUDIT_ROOT = Path(
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825"
)
DEFAULT_CROP_ROOT = DEFAULT_AUDIT_ROOT / (
    "crop_audit_v4_gt_repair/crop_only_horizontal_v5_train_repair_f04503b"
)
DEFAULT_OUTPUT = Path(
    "/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/"
    "Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/"
    "ui5_train_rollout_bundle_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", type=Path, default=DEFAULT_FULL_DATA)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--crop-root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, payload: str, length: int = 20) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


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


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def normalize_task(value: str) -> str:
    task = value.removeprefix("ui_")
    if task not in TASKS:
        raise ValueError(f"unknown UI5 task: {value}")
    return task


def conversation_value(record: Mapping[str, Any], role: str) -> str:
    aliases = {"human", "user"} if role == "human" else {"gpt", "assistant"}
    for item in record.get("conversations", []):
        if isinstance(item, Mapping) and str(item.get("from")) in aliases:
            return str(item.get("value", ""))
    return ""


def answer_boxes_1000(answer: str) -> list[list[int]]:
    import re

    pattern = re.compile(
        r"<box>\s*<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*"
        r"<\s*(-?\d+)\s*>\s*<\s*(-?\d+)\s*>\s*</box>",
        re.IGNORECASE,
    )
    boxes: list[list[int]] = []
    for match in pattern.finditer(answer):
        raw = [min(1000, max(0, int(value))) for value in match.groups()]
        x1, x2 = sorted((raw[0], raw[2]))
        y1, y2 = sorted((raw[1], raw[3]))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return [list(box) for box in dict.fromkeys(tuple(box) for box in boxes)]


def norm_to_pixels(box: Sequence[int], width: int, height: int) -> list[int]:
    return [
        round(int(box[0]) / 1000 * width),
        round(int(box[1]) / 1000 * height),
        round(int(box[2]) / 1000 * width),
        round(int(box[3]) / 1000 * height),
    ]


def contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0])
        and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2])
        and int(outer[3]) >= int(inner[3])
    )


def intersects(left: Sequence[int], right: Sequence[int]) -> bool:
    return not (
        int(left[2]) <= int(right[0])
        or int(right[2]) <= int(left[0])
        or int(left[3]) <= int(right[1])
        or int(right[3]) <= int(left[1])
    )


def union_area(rectangles: Sequence[Sequence[int]]) -> int:
    if not rectangles:
        return 0
    xs = sorted({int(v) for box in rectangles for v in (box[0], box[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        spans = sorted(
            (int(box[1]), int(box[3]))
            for box in rectangles
            if int(box[0]) < right and int(box[2]) > left
        )
        if not spans:
            continue
        start, end = spans[0]
        covered = 0
        for next_start, next_end in spans[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        total += (right - left) * (covered + end - start)
    return total


def validate_base_tiles(
    image_id: str, width: int, height: int, tiles: Sequence[Sequence[int]]
) -> list[list[int]]:
    normalized = [[int(value) for value in tile] for tile in tiles]
    if not normalized:
        raise ValueError(f"base scan plan has no tiles: {image_id}")
    for tile in normalized:
        if len(tile) != 4 or not (
            0 <= tile[0] < tile[2] <= width and 0 <= tile[1] < tile[3] <= height
        ):
            raise ValueError(f"invalid base tile for {image_id}: {tile}")
    if union_area(normalized) != width * height:
        raise ValueError(f"base tiles do not cover image {image_id}")
    return normalized


def crop_transform(gt: Sequence[int], crop: Sequence[int]) -> dict[str, Any]:
    crop_width = int(crop[2]) - int(crop[0])
    crop_height = int(crop[3]) - int(crop[1])
    local = [
        int(gt[0]) - int(crop[0]),
        int(gt[1]) - int(crop[1]),
        int(gt[2]) - int(crop[0]),
        int(gt[3]) - int(crop[1]),
    ]
    norm = [
        round(local[0] / crop_width * 1000),
        round(local[1] / crop_height * 1000),
        round(local[2] / crop_width * 1000),
        round(local[3] / crop_height * 1000),
    ]
    reconstructed = [
        round(norm[0] / 1000 * crop_width) + int(crop[0]),
        round(norm[1] / 1000 * crop_height) + int(crop[1]),
        round(norm[2] / 1000 * crop_width) + int(crop[0]),
        round(norm[3] / 1000 * crop_height) + int(crop[1]),
    ]
    return {
        "global_bbox_xyxy": list(map(int, gt)),
        "local_bbox_xyxy": local,
        "local_bbox_1000": norm,
        "offset_xy": [int(crop[0]), int(crop[1])],
        "scale_xy": [crop_width / 1000.0, crop_height / 1000.0],
        "roundtrip_global_bbox_xyxy": reconstructed,
        "roundtrip_max_error_px": max(
            abs(int(a) - int(b)) for a, b in zip(gt, reconstructed)
        ),
    }


def source_file_for(
    provenance: Mapping[str, Any], full_data: Path
) -> tuple[Path, int, str]:
    original = str(provenance.get("source_file", ""))
    line_no = int(provenance.get("line_no", 0))
    if line_no <= 0:
        raise ValueError(f"invalid source line number: {provenance}")
    candidate = Path(original)
    if not candidate.is_file():
        candidate = full_data / candidate.name
    if not candidate.is_file():
        raise FileNotFoundError(
            f"source record file missing: original={original}, fallback={candidate}"
        )
    try:
        relative = candidate.resolve().relative_to(full_data.resolve()).as_posix()
    except ValueError:
        relative = candidate.name
    return candidate.resolve(), line_no, relative


def load_source_record(path: Path, line_no: int, cache: dict[Path, list[str]]) -> dict[str, Any]:
    lines = cache.setdefault(path, path.read_text(encoding="utf-8").splitlines())
    if line_no > len(lines) or not lines[line_no - 1].strip():
        raise ValueError(f"source line does not exist: {path}:{line_no}")
    value = json.loads(lines[line_no - 1])
    if not isinstance(value, dict):
        raise ValueError(f"source line is not an object: {path}:{line_no}")
    return value


def resolve_image_source(row: Mapping[str, Any]) -> Path:
    candidates = [row.get("image_path"), *(row.get("canonical_paths") or [])]
    for value in candidates:
        if value and Path(str(value)).is_file():
            return Path(str(value)).resolve()
    raise FileNotFoundError(
        f"no readable source image for image_id={row.get('image_id')}: {candidates}"
    )


def detector_state(audit_root: Path, crop_root: Path) -> dict[str, Any]:
    summary_path = crop_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    declared = (summary.get("input_state") or {}).get("detections_digest")
    detector_path = audit_root / "detections" / "merged" / "detections.jsonl"
    actual = None
    algorithm = "sha256"
    if detector_path.is_file():
        if isinstance(declared, str) and declared.startswith("blake2b128:"):
            digest = hashlib.blake2b(digest_size=16)
            with detector_path.open("rb") as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
            actual = "blake2b128:" + digest.hexdigest()
            algorithm = "blake2b128"
        else:
            actual = sha256_file(detector_path)
    if declared and actual and declared != actual:
        raise ValueError(
            f"detector digest mismatch: summary={declared}, actual={actual}"
        )
    return {
        "algorithm": algorithm,
        "declared_digest": declared,
        "actual_digest": actual,
        "source_file": "audit/detections/merged/detections.jsonl",
        "verified": bool(declared and actual and declared == actual),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    full_data = args.full_data.expanduser().resolve(strict=True)
    audit_root = args.audit_root.expanduser().resolve(strict=True)
    crop_root = args.crop_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve(strict=False)
    required = {
        "unique_images": audit_root / "manifest" / "unique_images.jsonl",
        "task_samples": audit_root / "manifest" / "task_samples.jsonl",
        "base_scan_plans": crop_root / "base_scan_plans.json",
        "task_aware_manifest": crop_root / "task_aware_manifest.jsonl",
        "crop_summary": crop_root / "summary.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    training_files = [full_data / f"ui_{task}_train.jsonl" for task in TASKS]
    missing.extend(
        str(path) for path in training_files if not path.is_file() or path.stat().st_size <= 0
    )
    if missing:
        raise FileNotFoundError("missing bundle inputs: " + ", ".join(missing))

    complete_path = output / "bundle_manifest.json"
    if complete_path.is_file():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True and existing.get("schema_version") == SCHEMA_VERSION:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing
        raise RuntimeError(f"existing bundle is not complete: {output}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"refusing to mix with a partial/nonempty bundle directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "manifest").mkdir(parents=True, exist_ok=True)

    unique_input = read_jsonl(required["unique_images"])
    sample_input = read_jsonl(required["task_samples"])
    observed_tasks = {normalize_task(str(row["task"])) for row in sample_input}
    if observed_tasks != set(TASKS):
        raise ValueError(
            f"task_samples must contain all five UI5 tasks: observed={sorted(observed_tasks)}"
        )
    # Reading validates the source artifact exists, but rollout geometry never
    # consults its training-only final_tiles or repair fields.
    audit_task_rows = read_jsonl(required["task_aware_manifest"])
    audit_keys = {
        (str(row["image_id"]), normalize_task(str(row["task"])))
        for row in audit_task_rows
    }
    raw_plans = json.loads(required["base_scan_plans"].read_text(encoding="utf-8"))
    if not isinstance(raw_plans, dict):
        raise ValueError("base_scan_plans.json must be an object indexed by image_id")

    unique_rows: list[dict[str, Any]] = []
    image_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(sorted(unique_input, key=lambda item: str(item["image_id"])), 1):
        image_id = str(row["image_id"])
        source = resolve_image_source(row)
        suffix = source.suffix.lower() or ".img"
        destination = output / "images" / f"{image_id}{suffix}"
        if destination.is_file():
            if destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"existing copied image has wrong size: {destination}")
        else:
            shutil.copy2(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"dimension mismatch for {image_id}")
        portable = {
            "image_id": image_id,
            "content_id": row.get("content_id"),
            "image_relpath": destination.relative_to(output).as_posix(),
            "basename": row.get("basename") or source.name,
            "width": width,
            "height": height,
            "tasks": [normalize_task(str(task)) for task in row.get("tasks", [])],
            "sha256": sha256_file(destination),
        }
        unique_rows.append(portable)
        image_by_id[image_id] = portable
        if index % 1000 == 0 or index == len(unique_input):
            print(f"[images] {index}/{len(unique_input)}", flush=True)

    atomic_jsonl(output / "manifest" / "unique_images.jsonl", unique_rows)

    source_cache: dict[Path, list[str]] = {}
    source_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    portable_task_aware: list[dict[str, Any]] = []
    portable_plans: dict[str, dict[str, Any]] = {}
    polarity = Counter()
    original_record_count = 0
    coverage_failures = 0
    coordinate_anomalies = 0
    for sample_index, sample in enumerate(
        sorted(sample_input, key=lambda item: (str(item["image_id"]), str(item["task"]))),
        1,
    ):
        image_id = str(sample["image_id"])
        task = normalize_task(str(sample["task"]))
        if image_id not in image_by_id:
            raise ValueError(f"task sample references unknown image: {image_id}")
        if (image_id, task) not in audit_keys:
            raise ValueError(f"crop task-aware manifest misses {(image_id, task)}")
        image = image_by_id[image_id]
        width, height = int(image["width"]), int(image["height"])
        gt_global = [[int(v) for v in box] for box in sample.get("gt_boxes", [])]
        gt_1000 = [[int(round(float(v))) for v in box] for box in sample.get("gt_boxes_1000", [])]
        if len(gt_global) != len(gt_1000):
            raise ValueError(f"GT pixel/norm count mismatch: {sample.get('sample_id')}")
        sample_id = str(sample["sample_id"])
        provenance_rows = list(sample.get("source_records") or [])
        if not provenance_rows:
            provenance_rows = [
                {"source_file": sample.get("source_file"), "line_no": sample.get("line_no")}
            ]
        source_ids: list[str] = []
        prompts: list[str] = []
        source_gt_union: set[tuple[int, int, int, int]] = set()
        representative_record: dict[str, Any] | None = None
        portable_provenance: list[dict[str, Any]] = []
        for provenance in provenance_rows:
            source_path, line_no, source_rel = source_file_for(provenance, full_data)
            original = load_source_record(source_path, line_no, source_cache)
            prompt = conversation_value(original, "human")
            answer = conversation_value(original, "gpt")
            original_norm = answer_boxes_1000(answer)
            original_pixels = [norm_to_pixels(box, width, height) for box in original_norm]
            source_gt_union.update(tuple(box) for box in original_pixels)
            source_record_id = stable_id(
                "record", f"{source_rel}\0{line_no}\0{image_id}\0{task}", 24
            )
            portable_record = json.loads(json.dumps(original, ensure_ascii=False))
            portable_record["image"] = image["image_relpath"]
            if "images" in portable_record:
                portable_record["images"] = [image["image_relpath"]]
            source_rows.append(
                {
                    "source_record_id": source_record_id,
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "task": task,
                    "source_file": source_rel,
                    "line_no": line_no,
                    "prompt": prompt,
                    "answer": answer,
                    "gt_boxes_1000": original_norm,
                    "gt_boxes_global_xyxy": original_pixels,
                    "original_training_record": portable_record,
                    "original_training_record_sha256": json_digest(original),
                    "portable_training_record": portable_record,
                }
            )
            source_ids.append(source_record_id)
            prompts.append(prompt)
            portable_provenance.append(
                {
                    "source_record_id": source_record_id,
                    "source_file": source_rel,
                    "line_no": line_no,
                }
            )
            if representative_record is None:
                representative_record = portable_record
            original_record_count += 1
        prompt_set = {value for value in prompts if value}
        annotation_anomaly = bool(
            sample.get("same_task_polarity_conflict")
            or len(prompt_set) != 1
            or source_gt_union != {tuple(box) for box in gt_global}
        )
        prompt = min(prompt_set) if prompt_set else ""
        if not prompt:
            annotation_anomaly = True

        if task == "content_missing":
            base_tiles = [[0, 0, width, height]]
            plan_digest = json_digest(base_tiles)
        else:
            raw_plan = raw_plans.get(image_id)
            if not isinstance(raw_plan, Mapping):
                raise ValueError(f"base scan plan missing image_id={image_id}")
            if "base_tiles" in raw_plan:
                candidate_tiles = raw_plan["base_tiles"]
            elif "tiles" in raw_plan:
                candidate_tiles = raw_plan["tiles"]
            else:
                raise ValueError(f"base scan plan has no GT-free tiles: {image_id}")
            base_tiles = validate_base_tiles(image_id, width, height, candidate_tiles)
            plan_digest = str(raw_plan.get("geometry_digest") or json_digest(base_tiles))
        if image_id in portable_plans:
            if portable_plans[image_id]["base_tiles"] != base_tiles and task != "content_missing":
                raise ValueError(f"inconsistent base tiles for image_id={image_id}")
        elif task != "content_missing":
            portable_plans[image_id] = {
                "image_id": image_id,
                "width": width,
                "height": height,
                "base_tiles": base_tiles,
                "geometry_digest": plan_digest,
                "gt_used": False,
                "source": "crop_root/base_scan_plans.json",
            }

        contained_any = [any(contains(tile, gt) for tile in base_tiles) for gt in gt_global]
        crossing = [index for index, ok in enumerate(contained_any) if not ok]
        coverage_failure = bool(crossing)
        if coverage_failure:
            coverage_failures += 1
        sample_crop_ids: list[str] = []
        sample_coordinate_anomaly = False
        for crop_index, crop in enumerate(base_tiles):
            crop_id = stable_id("crop", f"{sample_id}\0{crop_index}\0{crop}", 24)
            transforms = [
                crop_transform(gt, crop)
                for gt in gt_global
                if contains(crop, gt)
            ]
            partial_indices = [
                index
                for index, gt in enumerate(gt_global)
                if intersects(crop, gt) and not contains(crop, gt)
            ]
            coordinate_anomaly = any(
                int(transform["roundtrip_max_error_px"]) > 1 for transform in transforms
            )
            sample_coordinate_anomaly = sample_coordinate_anomaly or coordinate_anomaly
            crop_rows.append(
                {
                    "record_id": sample_id,
                    "sample_id": sample_id,
                    "source_image_id": image_id,
                    "task": task,
                    "prompt": prompt,
                    "image_relpath": image["image_relpath"],
                    "crop_id": crop_id,
                    "crop_index": crop_index,
                    "crop_xyxy": crop,
                    "crop_size": [crop[2] - crop[0], crop[3] - crop[1]],
                    "gt_local": [row["local_bbox_xyxy"] for row in transforms],
                    "gt_local_1000": [row["local_bbox_1000"] for row in transforms],
                    "gt_global": [row["global_bbox_xyxy"] for row in transforms],
                    "sample_gt_global": gt_global,
                    "coordinate_transforms": transforms,
                    "partial_gt_indices": partial_indices,
                    "pipeline_coverage_failure": coverage_failure,
                    "coordinate_transform_anomaly": coordinate_anomaly,
                    "geometry_source": "base_scan_plans.base_tiles",
                    "gt_used_for_geometry": False,
                }
            )
            sample_crop_ids.append(crop_id)
        if sample_coordinate_anomaly:
            coordinate_anomalies += 1

        task_row = {
            "record_id": sample_id,
            "sample_id": sample_id,
            "source_image_id": image_id,
            "image_id": image_id,
            "task": task,
            "split": "train",
            "image_relpath": image["image_relpath"],
            "width": width,
            "height": height,
            "prompt": prompt,
            "gt_global": gt_global,
            "gt_global_1000": gt_1000,
            "positive": bool(gt_global),
            "gt_count": len(gt_global),
            "source_records": portable_provenance,
            "source_record_ids": source_ids,
            "original_training_record": representative_record,
            "annotation_anomaly": annotation_anomaly,
            "coordinate_transform_anomaly": sample_coordinate_anomaly,
            "pipeline_coverage_failure": coverage_failure,
            "coverage_failure_type": (
                "PIPELINE_COVERAGE_FAILURE" if coverage_failure else None
            ),
            "base_seam_crossing_gt_indices": crossing,
            "crop_ids": sample_crop_ids,
            "crop_count": len(sample_crop_ids),
            "crop_geometry_source": "base_scan_plans.base_tiles",
            "grpo_eligible": not (
                annotation_anomaly or sample_coordinate_anomaly or coverage_failure
            ),
        }
        task_rows.append(task_row)
        portable_task_aware.append(
            {
                "record_id": sample_id,
                "sample_id": sample_id,
                "source_image_id": image_id,
                "task": task,
                "image_relpath": image["image_relpath"],
                "width": width,
                "height": height,
                "prompt": prompt,
                "gt_global": gt_global,
                "base_tiles": base_tiles,
                "crop_ids": sample_crop_ids,
                "pipeline_coverage_failure": coverage_failure,
                "base_seam_crossing_gt_indices": crossing,
                "content_missing_global_view": task == "content_missing",
                "geometry_source": "base_scan_plans.base_tiles",
                "gt_used_for_geometry": False,
            }
        )
        polarity[(task, "positive" if gt_global else "negative")] += 1
        if sample_index % 5000 == 0 or sample_index == len(sample_input):
            print(f"[samples] {sample_index}/{len(sample_input)}", flush=True)

    source_rows.sort(key=lambda row: row["source_record_id"])
    if len({str(row["source_record_id"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source_record_id collision/duplicate detected")
    task_rows.sort(key=lambda row: (row["task"], row["record_id"]))
    crop_rows.sort(key=lambda row: (row["task"], row["record_id"], row["crop_index"]))
    portable_task_aware.sort(key=lambda row: (row["task"], row["record_id"]))
    atomic_jsonl(output / "manifest" / "source_records.jsonl", source_rows)
    atomic_jsonl(output / "manifest" / "task_samples.jsonl", task_rows)
    atomic_jsonl(output / "manifest" / "crop_samples.jsonl", crop_rows)
    atomic_jsonl(output / "task_aware_manifest.jsonl", portable_task_aware)
    atomic_json(output / "base_scan_plans.json", portable_plans)
    detector = detector_state(audit_root, crop_root)
    atomic_json(output / "manifest" / "detector_digest.json", detector)

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "created_at": utc_now(),
        "path_policy": "all runnable bundle paths are relative to bundle root",
        "geometry_policy": "crop rollout reads only base_scan_plans.base_tiles",
        "original_training_records": original_record_count,
        "rollout_samples": len(task_rows),
        "unique_images": len(unique_rows),
        "crop_records_runtime_only": len(crop_rows),
        "materialized_crop_images": 0,
        "pipeline_coverage_failures": coverage_failures,
        "coordinate_transform_anomalies": coordinate_anomalies,
        "positive_negative_by_task": {
            task: {
                "positive": polarity[(task, "positive")],
                "negative": polarity[(task, "negative")],
            }
            for task in TASKS
        },
        "detector": detector,
        "excluded_payloads": [
            "materialized_crop_images",
            "validation_or_test_detector_cache",
            "ocr_or_icon_weights",
            "training_token_cache",
            "final_tiles",
            "removed_gt_crossing_seams",
            "manual_gt_repair_geometry",
        ],
        "files": {},
    }
    for relative in (
        "manifest/unique_images.jsonl",
        "manifest/source_records.jsonl",
        "manifest/task_samples.jsonl",
        "manifest/crop_samples.jsonl",
        "manifest/detector_digest.json",
        "base_scan_plans.json",
        "task_aware_manifest.jsonl",
    ):
        path = output / relative
        inventory["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_json(complete_path, inventory)
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return inventory


def main(argv: Sequence[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except Exception as exc:
        print(f"[bundle:error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
