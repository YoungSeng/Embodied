#!/usr/bin/env python3
"""Build full-width UI5 crop-only training strips from immutable detections.

The base partition is exactly the GT-free schema-v5 raw-detector-edge scan used
at evaluation.  Training GT may only remove seams that cross a GT box.  This
merges adjacent strips into another strict, lossless vertical partition; GT is
never rendered into an image and never becomes model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from analyze_ui5_source_overlap import content_fingerprint
from run_ui5_crop_audit import (
    REGION_TASKS,
    TASK_LABELS,
    ProgressReporter,
    atomic_save_png,
    atomic_write_json,
    atomic_write_jsonl,
    build_answer,
    normalize_gt_in_crop,
    open_raw_image,
    read_jsonl,
    rect_contains,
    rect_intersects,
)
from run_ui5_gt_repair import EXCLUDED_SAMPLE_ID, EXCLUDED_TASK
from ui5_lossless_tiling import generate_detector_scan_plan, strict_vertical_partition_metrics


SCHEMA_VERSION = 2
DEFAULT_NAME = "crop_only_horizontal_v5_train_repair"
SCRIPT_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = (
    SCRIPT_DIR / "build_ui5_croponly_training_manifest.py",
    SCRIPT_DIR / "ui5_lossless_tiling.py",
    SCRIPT_DIR / "run_ui5_crop_audit.py",
    SCRIPT_DIR / "run_ui5_gt_repair.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=DEFAULT_NAME)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--max-crops", type=int, default=10)
    parser.add_argument("--target-height", type=int, default=960)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _digest_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _implementation_state() -> dict[str, Any]:
    files = {
        path.name: content_fingerprint(path.resolve(strict=True))
        for path in IMPLEMENTATION_FILES
    }
    return {"files": files, "digest": _digest_json(files)}


def _detection_boxes(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source, key in (("text", "text_detections"), ("icon", "icon_detections")):
        for index, detection in enumerate(row.get(key, [])):
            bbox = [int(round(float(value))) for value in detection["bbox"]]
            output.append({"bbox": bbox, "source": source, "id": f"{source}_{index:06d}"})
    return output


def _remove_gt_crossing_seams(
    base_tiles: Sequence[Sequence[int]],
    gt_boxes: Sequence[Sequence[int]],
    *,
    width: int,
    height: int,
) -> tuple[list[list[int]], list[int]]:
    base_seams = [int(tile[3]) for tile in base_tiles[:-1]]
    removed = sorted(
        seam
        for seam in base_seams
        if any(int(gt[1]) < seam < int(gt[3]) for gt in gt_boxes)
    )
    retained = [seam for seam in base_seams if seam not in set(removed)]
    edges = [0, *retained, int(height)]
    tiles = [[0, top, int(width), bottom] for top, bottom in zip(edges, edges[1:])]
    metrics = strict_vertical_partition_metrics(width, height, tiles)
    if not metrics["strict_vertical_partition"]:
        raise AssertionError(f"GT seam repair broke the strict partition: {tiles}")
    if metrics["processed_pixel_ratio"] != 1.0:
        raise AssertionError(f"GT seam repair duplicated or dropped pixels: {metrics}")
    return tiles, removed


def _make_conversations(
    sample: Mapping[str, Any], crop: Sequence[int]
) -> tuple[list[dict[str, str]], list[int], int]:
    contained = [
        index
        for index, gt in enumerate(sample["gt_boxes"])
        if rect_contains(crop, gt)
    ]
    partial = [
        index
        for index, gt in enumerate(sample["gt_boxes"])
        if rect_intersects(crop, gt) and not rect_contains(crop, gt)
    ]
    if partial:
        raise AssertionError(
            f"partial GT remained after seam repair: sample={sample['sample_id']} "
            f"crop={list(crop)} partial={partial}"
        )
    transforms = [normalize_gt_in_crop(sample["gt_boxes"][index], crop) for index in contained]
    max_error = max((int(row["roundtrip_max_error_px"]) for row in transforms), default=0)
    if max_error > 1:
        raise AssertionError(
            f"crop label round-trip exceeds 1 px: sample={sample['sample_id']} "
            f"crop={list(crop)} error={max_error}"
        )
    label = TASK_LABELS[str(sample["task"])]
    conversations = [
        {
            "from": "human",
            "value": f"Locate all the instances that match the following description: {label}.",
        },
        {
            "from": "gpt",
            "value": build_answer(label, [row["norm1000"] for row in transforms]),
        },
    ]
    return conversations, contained, max_error


def _materialize(source: Path, destination: Path, crop: Sequence[int]) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        with Image.open(destination) as existing:
            expected = (int(crop[2]) - int(crop[0]), int(crop[3]) - int(crop[1]))
            if existing.size == expected:
                return
    with open_raw_image(source) as image:
        cropped = image.crop(tuple(map(int, crop)))
        try:
            atomic_save_png(cropped, destination)
        finally:
            cropped.close()


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = args.audit_dir.expanduser().resolve(strict=True)
    parent = audit_dir.parent
    output_root = audit_dir / args.output_name
    manifest_path = output_root / "task_aware_manifest.jsonl"
    summary_path = output_root / "summary.json"
    done_path = output_root / "complete.json"
    detections_path = (
        args.detections.expanduser().resolve(strict=True)
        if args.detections is not None
        else (parent / "detections" / "merged" / "detections.jsonl").resolve(strict=True)
    )
    task_samples_path = (parent / "manifest" / "task_samples.jsonl").resolve(strict=True)
    repair_actions_path = (audit_dir / "gt_repair_actions.jsonl").resolve(strict=True)
    if not 1 <= int(args.max_crops) <= 10:
        raise ValueError("--max-crops must be in [1,10]")
    if int(args.target_height) <= 0:
        raise ValueError("--target-height must be positive")

    implementation = _implementation_state()
    input_state = {
        "schema_version": SCHEMA_VERSION,
        "implementation_files": implementation["files"],
        "implementation_digest": implementation["digest"],
        "detections_digest": content_fingerprint(detections_path),
        "task_samples_digest": content_fingerprint(task_samples_path),
        "repair_actions_digest": content_fingerprint(repair_actions_path),
        "max_crops": int(args.max_crops),
        "target_height": int(args.target_height),
        "geometry": "schema_v5_raw_detector_edge_then_train_gt_seam_removal",
    }
    if args.resume and done_path.is_file() and manifest_path.is_file() and summary_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("input_state_digest") == _digest_json(input_state):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(done.get("manifest_rows", -1)) == len(read_jsonl(manifest_path)):
                print(f"[crop-only] --resume reused {manifest_path}", flush=True)
                return summary

    detections = {str(row["image_id"]): row for row in read_jsonl(detections_path)}
    samples = [
        row
        for row in read_jsonl(task_samples_path)
        if not (
            str(row["sample_id"]) == EXCLUDED_SAMPLE_ID
            and str(row["task"]) == EXCLUDED_TASK
        )
    ]
    repair_keys = {
        (str(row["sample_id"]), int(row["gt_index"]))
        for row in read_jsonl(repair_actions_path)
    }
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_image[str(sample["image_id"])].append(sample)
    if set(by_image) - set(detections):
        raise ValueError(
            f"merged detections miss {len(set(by_image) - set(detections))} training images"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(
        stage="crop-only-horizontal-train",
        total=len(by_image),
        output_dir=output_root,
        interval_seconds=float(args.progress_interval_seconds),
        unit="images",
    )
    reporter.update(0, detail="v5 GT-free strips + train-only seam repair", force=True)
    rows: list[dict[str, Any]] = []
    mapped_repair_keys: set[tuple[str, int]] = set()
    physical_paths: set[str] = set()
    base_plan_by_image: dict[str, dict[str, Any]] = {}
    for image_index, (image_id, image_samples) in enumerate(sorted(by_image.items()), 1):
        detection = detections[image_id]
        representative = image_samples[0]
        width, height = int(representative["width"]), int(representative["height"])
        source = Path(str(representative["canonical_path"])).resolve(strict=True)
        boxes = _detection_boxes(detection)
        base_plan = generate_detector_scan_plan(
            width,
            height,
            boxes,
            max_tiles=int(args.max_crops),
            target_tile_height=int(args.target_height),
            context_pixels=0,
            strict_vertical_partition=True,
            target_guard_ratio=0.0,
            target_guard_min_pixels=0,
            target_guard_max_pixels=0,
            seam_edge_reference="raw-detector-bbox",
            seam_candidates="safe-raw-detector-edges-only",
        )
        base_plan_by_image[image_id] = {
            "tiles": base_plan["tiles"],
            "horizontal_seams": base_plan["horizontal_seams"],
            "geometry_digest": _digest_json(base_plan),
        }
        for sample in sorted(image_samples, key=lambda row: str(row["task"])):
            task = str(sample["task"])
            if task == "ui_content_missing":
                # Recipe builder reuses the original image and original normalized labels.
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "image_id": image_id,
                        "task": task,
                        "split": sample.get("split"),
                        "source_image": str(source),
                        "width": width,
                        "height": height,
                        "positive": bool(sample["gt_boxes"]),
                        "gt_boxes": sample["gt_boxes"],
                        "gt_boxes_1000": sample["gt_boxes_1000"],
                        "base_tiles": [[0, 0, width, height]],
                        "final_tiles": [[0, 0, width, height]],
                        "removed_gt_crossing_seams": [],
                        "training_records": [],
                        "content_missing_global_view": True,
                    }
                )
                continue
            if task not in REGION_TASKS:
                raise ValueError(f"unknown UI5 task: {task}")
            tiles, removed_seams = _remove_gt_crossing_seams(
                base_plan["tiles"],
                sample["gt_boxes"],
                width=width,
                height=height,
            )
            training_records: list[dict[str, Any]] = []
            for tile_index, tile in enumerate(tiles):
                conversations, contained, max_error = _make_conversations(sample, tile)
                repair_indices = sorted(
                    index
                    for index in contained
                    if (str(sample["sample_id"]), index) in repair_keys
                )
                mapped_repair_keys.update(
                    (str(sample["sample_id"]), index) for index in repair_indices
                )
                source_kind = (
                    "manual_gt_repair"
                    if repair_indices
                    else ("train_gt_seam_repair" if removed_seams else "raw_detector_strip")
                )
                crop_path = image_dir / (
                    f"{image_id}__y{int(tile[1]):06d}_{int(tile[3]):06d}.png"
                )
                _materialize(source, crop_path, tile)
                physical_paths.add(str(crop_path.resolve()))
                training_records.append(
                    {
                        "image": str(crop_path.resolve()),
                        "conversations": conversations,
                        "_ui5_sample_id": sample["sample_id"],
                        "_ui5_image_id": image_id,
                        "_ui5_source_image": str(source),
                        "_ui5_task": task,
                        "_ui5_split": sample.get("split"),
                        "_ui5_record_kind": "crop",
                        "_ui5_crop_source": source_kind,
                        "_ui5_crop_bbox": list(tile),
                        "_ui5_crop_index": tile_index,
                        "_ui5_base_tile_count": len(base_plan["tiles"]),
                        "_ui5_final_tile_count": len(tiles),
                        "_ui5_removed_gt_crossing_seams": removed_seams,
                        "_ui5_contained_gt_indices": contained,
                        "_ui5_manual_repair_gt_indices": repair_indices,
                        "_ui5_training_eligible": True,
                        "_ui5_roundtrip_max_error_px": max_error,
                        "_ui5_partial_gt_indices": [],
                        "_ui5_positive": bool(contained),
                        "_ui5_horizontal_full_width": True,
                    }
                )
            if not all(
                any(rect_contains(tile, gt) for tile in tiles)
                for gt in sample["gt_boxes"]
            ):
                raise AssertionError(f"valid GT is absent from crop-only records: {sample['sample_id']}")
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "image_id": image_id,
                    "task": task,
                    "split": sample.get("split"),
                    "source_image": str(source),
                    "width": width,
                    "height": height,
                    "positive": bool(sample["gt_boxes"]),
                    "gt_boxes": sample["gt_boxes"],
                    "gt_boxes_1000": sample["gt_boxes_1000"],
                    "base_tiles": base_plan["tiles"],
                    "final_tiles": tiles,
                    "removed_gt_crossing_seams": removed_seams,
                    "strict_vertical_partition": True,
                    "training_records": training_records,
                    "content_missing_global_view": False,
                }
            )
        reporter.update(image_index, detail=f"image_id={image_id}")

    missing_repair = repair_keys - mapped_repair_keys
    if missing_repair:
        raise RuntimeError(
            f"{len(missing_repair)} valid repair GTs have no crop-only positive record; "
            f"first={sorted(missing_repair)[:20]}"
        )
    records = [record for row in rows for record in row["training_records"]]
    task_counts = Counter(str(record["_ui5_task"]) for record in records)
    polarity = Counter(
        (str(record["_ui5_task"]), "positive" if record["_ui5_positive"] else "negative")
        for record in records
    )
    source_crop_counts = Counter(str(record["_ui5_source_image"]) for record in records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "crop_only",
        "geometry": input_state["geometry"],
        "gt_used_only_for_training_seam_repair_and_labels": True,
        "gt_rendered_into_images": False,
        "input_state": input_state,
        "input_state_digest": _digest_json(input_state),
        "task_manifest_rows": len(rows),
        "region_crop_records": len(records),
        "content_missing_global_records": sum(
            str(row["task"]) == "ui_content_missing" for row in rows
        ),
        "local_task_full_image_records": 0,
        "physical_crop_files": len(physical_paths),
        "records_by_task": dict(sorted(task_counts.items())),
        "positive_negative_by_task": {
            task: {
                "positive": polarity[(task, "positive")],
                "negative": polarity[(task, "negative")],
            }
            for task in sorted(task_counts)
        },
        "partial_negative_count": sum(
            bool(record.get("_ui5_partial_gt_indices")) and not record["_ui5_positive"]
            for record in records
        ),
        "repair_gt_expected": len(repair_keys),
        "repair_gt_mapped": len(mapped_repair_keys),
        "all_repair_gt_mapped": repair_keys <= mapped_repair_keys,
        "all_legal_strips_retained": True,
        "crop_count_per_source": {
            "mean": sum(source_crop_counts.values()) / max(1, len(source_crop_counts)),
            "max": max(source_crop_counts.values(), default=0),
        },
    }
    atomic_write_jsonl(manifest_path, rows)
    atomic_write_json(output_root / "base_scan_plans.json", base_plan_by_image)
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        done_path,
        {
            "schema_version": SCHEMA_VERSION,
            "input_state_digest": summary["input_state_digest"],
            "manifest_digest": content_fingerprint(manifest_path),
            "manifest_rows": len(rows),
            "summary_digest": content_fingerprint(summary_path),
            "complete": True,
        },
    )
    reporter.update(
        len(by_image),
        status="completed",
        detail=f"records={len(records)} repair_gt={len(mapped_repair_keys)}/{len(repair_keys)}",
        force=True,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(build(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
