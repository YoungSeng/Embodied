#!/usr/bin/env python3
"""Prepare GT-free PP-OCR/icon detector crops for UI5 evaluation.

The two GPU detectors run once per unique test image and write resumable shard
JSONL.  Geometry is CPU-only: it turns the merged detections into a strict,
full-width horizontal partition.  Adjacent crops share a half-open boundary
but no pixel row, and every boundary is detector-safe.  No annotation or GT
field is read by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from locany_ui5_common import TASK_JSONL, TASKS
from run_ui5_crop_audit import (
    AuditPaths,
    ProgressReporter,
    atomic_write_json,
    atomic_write_jsonl,
    detector_config,
    digest_ids,
    ensure_detector_config,
    merge_detections,
    read_jsonl,
    resolve_python_executable,
    run_detection_stage,
    run_detector_worker,
)
from ui5_lossless_tiling import generate_detector_scan_plan, strict_vertical_partition_metrics
from ui5_eval_detector_cache import (
    CACHE_MARKER_SCHEMA_VERSION,
    GEOMETRY_SCHEMA_VERSION,
    json_digest as cache_json_digest,
    marker_path as cache_marker_path,
    sha256_file,
)


# Detector selection/shard identity remains v2 so a geometry-only schema bump
# never invalidates or rewrites the expensive raw text/icon cache.
DETECTOR_MANIFEST_FORMAT_VERSION = 2
SCAN_FORMAT_VERSION = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "text", "icon", "merge", "crop", "_worker"),
        default="all",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parser-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2), default=1)
    parser.add_argument("--allow-two-processes-per-gpu", action="store_true")
    parser.add_argument("--text-python", default=os.environ.get("TEXT_PYTHON"))
    parser.add_argument("--icon-python", default=os.environ.get("ICON_PYTHON"))
    parser.add_argument("--image-loader-threads", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=750)
    parser.add_argument("--max-images-per-task", type=int, default=0)
    parser.add_argument("--max-unique-images", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--skip-figma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument("--progress-every-images", type=int, default=25)
    parser.add_argument("--text-model-dir", type=Path, default=None)
    parser.add_argument("--icon-model", type=Path, default=None)
    parser.add_argument("--text-long-side", type=int, default=1920)
    parser.add_argument("--text-box-threshold", type=float, default=0.3)
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--icon-long-side", type=int, default=1920)
    parser.add_argument("--icon-confidence", type=float, default=0.05)
    parser.add_argument("--scan-name", default="horizontal_scan_v3_no_overlap")
    parser.add_argument(
        "--cache-scope", choices=("auto", "preview", "full_test"), default="auto"
    )
    parser.add_argument("--expected-full-test-unique-images", type=int, default=17281)
    parser.add_argument("--scan-max-crops", type=int, default=10)
    parser.add_argument("--scan-target-height", type=int, default=960)
    parser.add_argument("--scan-overlap-ratio", type=float, default=0.12)
    parser.add_argument("--scan-vertical-link-ratio", type=float, default=0.025)
    parser.add_argument("--scan-context-ratio", type=float, default=0.20)
    parser.add_argument("--scan-min-context-image-ratio", type=float, default=0.015)
    parser.add_argument("--scan-dense-band-ratio", type=float, default=0.80)
    parser.add_argument("--scan-detector-margin-ratio", type=float, default=0.003)
    parser.add_argument("--scan-seam-search-ratio", type=float, default=0.25)
    parser.add_argument("--scan-context-pixels", type=int, default=0)
    parser.add_argument(
        "--strict-vertical-partition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scan-minimum-core-height-ratio", type=float, default=0.35)
    parser.add_argument("--visualization-samples", type=int, default=60)
    parser.add_argument("--save-preview-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detector-stage", choices=("text", "icon"), help=argparse.SUPPRESS)
    # Accepted only because the shared audit worker still emits them for its own entrypoint.
    parser.add_argument("--source-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--locany-data-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.detector_only_mode = True
    args.detector_worker_script = Path(__file__).resolve()
    return args


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_paths(sample: Mapping[str, Any], base: Path) -> list[Path]:
    images = sample.get("images", sample.get("image"))
    if images is None:
        return []
    if isinstance(images, (str, Mapping)):
        images = [images]
    if not isinstance(images, list):
        return []
    paths: list[Path] = []
    for image in images:
        raw = image if isinstance(image, str) else image.get("path") if isinstance(image, Mapping) else None
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = base / path
        paths.append(path.resolve(strict=False))
    return paths


def _task_paths(path: Path, *, limit: int, skip_figma: bool) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            for image_path in _extract_paths(sample, path.parent):
                key = str(image_path)
                if key in seen or (skip_figma and ":" in image_path.name):
                    continue
                if not image_path.is_file():
                    raise FileNotFoundError(f"test image referenced by {path}:{line_no} is missing: {image_path}")
                seen.add(key)
                selected.append(image_path)
                if limit and len(selected) >= limit:
                    return selected
    return selected


def prepare_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_dir is None:
        raise ValueError("--input-dir is required for --stage prepare/all")
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    task_files = {task: input_dir / TASK_JSONL[task] for task in TASKS}
    missing = [path for path in task_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing test JSONL: " + ", ".join(map(str, missing)))

    selected_by_task = {
        task: _task_paths(
            task_files[task],
            limit=args.max_images_per_task,
            skip_figma=args.skip_figma,
        )
        for task in TASKS
    }
    total_selected = sum(len(paths) for paths in selected_by_task.values())
    reporter = ProgressReporter(
        stage="eval-prepare",
        total=total_selected,
        output_dir=args.output_dir,
        interval_seconds=float(getattr(args, "progress_interval_seconds", 10.0)),
        unit="task-images",
    )
    reporter.update(0, detail="读取测试图片内容指纹并按内容去重", force=True)
    by_content: dict[str, dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    path_info: dict[str, tuple[str, int, int]] = {}
    completed = 0
    for task in TASKS:
        for task_index, image_path in enumerate(selected_by_task[task]):
            info = path_info.get(str(image_path))
            if info is None:
                content_id = _file_digest(image_path)
                with Image.open(image_path) as opened:
                    oriented = ImageOps.exif_transpose(opened)
                    width, height = oriented.size
                info = (content_id, width, height)
                path_info[str(image_path)] = info
            content_id, width, height = info
            image_id = f"eval_{content_id[:20]}"
            row = by_content.setdefault(
                content_id,
                {
                    "image_id": image_id,
                    "content_id": content_id,
                    "image_path": str(image_path),
                    "image_paths": [],
                    "width": width,
                    "height": height,
                    "tasks": [],
                },
            )
            if (row["width"], row["height"]) != (width, height):
                raise ValueError(f"same-content dimensions changed for {image_path}")
            if str(image_path) not in row["image_paths"]:
                row["image_paths"].append(str(image_path))
            if task not in row["tasks"]:
                row["tasks"].append(task)
            task_rows.append(
                {
                    "task": task,
                    "task_index": task_index,
                    "image_id": image_id,
                    "content_id": content_id,
                    "image_path": str(image_path),
                }
            )
            completed += 1
            reporter.update(completed)

    unique = sorted(by_content.values(), key=lambda row: row["image_id"])
    if not unique:
        raise ValueError("no readable test images were selected")
    paths = AuditPaths(args.output_dir)
    paths.manifest.mkdir(parents=True, exist_ok=True)
    paths.shards.mkdir(parents=True, exist_ok=True)
    selection = {
        "format_version": DETECTOR_MANIFEST_FORMAT_VERSION,
        "input_dir": str(input_dir),
        "task_files": {task: str(path) for task, path in task_files.items()},
        "task_file_digests": {task: sha256_file(path) for task, path in task_files.items()},
        "max_images_per_task": args.max_images_per_task,
        "skip_figma": args.skip_figma,
        "unique_images": len(unique),
        "image_id_digest": digest_ids(row["image_id"] for row in unique),
        "content_id_digest": digest_ids(row["content_id"] for row in unique),
    }
    identity_path = paths.manifest / "selection_config.json"
    if identity_path.is_file() and args.resume:
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("evaluation detector cache selection changed; use a new --output-dir")
    atomic_write_jsonl(paths.unique_images, unique)
    atomic_write_jsonl(paths.task_samples, task_rows)
    for stale in paths.shards.glob("shard_*.jsonl"):
        stale.unlink()
    for start in range(0, len(unique), args.shard_size):
        atomic_write_jsonl(
            paths.shards / f"shard_{start // args.shard_size:05d}.jsonl",
            unique[start : start + args.shard_size],
        )
    atomic_write_json(identity_path, selection)
    ensure_detector_config(paths.detector_config, detector_config(args))
    print(
        f"[eval prepare] task records={len(task_rows)}, content-unique images={len(unique)}, "
        f"shards={math.ceil(len(unique) / args.shard_size)}, GT=disabled",
        flush=True,
    )
    reporter.update(
        total_selected,
        status="completed",
        detail=f"{len(unique)} 张内容唯一图片，{math.ceil(len(unique) / args.shard_size)} 个 shard",
        force=True,
    )
    return unique


def _density(box_count: int) -> str:
    # Match the training crop-audit effectiveness report exactly.
    if box_count <= 50:
        return "sparse"
    if box_count <= 150:
        return "medium"
    return "dense"


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tile_counts = [int(row["tile_count"]) for row in rows]
    partition_metrics = [
        strict_vertical_partition_metrics(
            int(row["width"]), int(row["height"]), row["tiles"]
        )
        for row in rows
    ]
    processed = [float(metrics["processed_pixel_ratio"]) for metrics in partition_metrics]
    gains = [float(row["mean_vertical_linear_gain"]) for row in rows]
    contain_rates = [float(row.get("detector_bbox_containment_rate", 1.0)) for row in rows]
    detector_total = sum(int(row.get("detector_box_count", 0)) for row in rows)
    detector_contained = sum(
        int(row.get("detector_bbox_contained_count", 0)) for row in rows
    )
    detector_unique_contained = sum(
        int(row.get("detector_bbox_unique_containment_count", 0)) for row in rows
    )
    seam_sources = Counter(
        source for row in rows for source in row.get("seam_source", [])
    )
    return {
        "images": len(rows),
        "tile_count_mean": mean(tile_counts) if tile_counts else 0.0,
        "tile_count_p50": median(tile_counts) if tile_counts else 0.0,
        "tile_count_p90": _percentile(tile_counts, 0.90),
        "tile_count_max": max(tile_counts, default=0),
        "tile_count_distribution": dict(sorted(Counter(tile_counts).items())),
        "processed_pixel_ratio_with_overlap_mean": mean(processed) if processed else 0.0,
        "processed_pixel_ratio_with_overlap_p50": median(processed) if processed else 0.0,
        "processed_pixel_ratio_with_overlap_p90": _percentile(processed, 0.90),
        "processed_pixel_ratio_mean": mean(processed) if processed else 0.0,
        "processed_pixel_ratio_p50": median(processed) if processed else 0.0,
        "processed_pixel_ratio_p90": _percentile(processed, 0.90),
        "mean_vertical_linear_gain": mean(gains) if gains else 0.0,
        "single_full_image_count": sum(int(row["tile_count"]) == 1 for row in rows),
        "detector_empty_count": sum(int(row["detector_box_count"]) == 0 for row in rows),
        "detector_boundary_cut_count": sum(int(row["detector_boundary_cut_count"]) for row in rows),
        "detector_bbox_contained_count": detector_contained,
        "detector_bbox_unique_containment_count": detector_unique_contained,
        "detector_bbox_count": detector_total,
        "detector_bbox_containment_rate": (
            detector_contained / detector_total if detector_total else 1.0
        ),
        "detector_bbox_containment_rate_min": min(contain_rates, default=1.0),
        "uncontained_detector_bbox_count": sum(
            int(row.get("uncontained_detector_bbox_count", 0)) for row in rows
        ),
        "seam_crossed_detector_bbox_count": sum(
            int(row.get("seam_crossed_detector_bbox_count", 0)) for row in rows
        ),
        "horizontal_seam_count": sum(
            int(row.get("horizontal_seam_count", 0)) for row in rows
        ),
        "seam_source_counts": dict(sorted(seam_sources.items())),
        "safe_seam_count": sum(int(row.get("safe_seam_count", 0)) for row in rows),
        "balanced_fallback_seam_count": sum(
            int(row.get("balanced_fallback_seam_count", 0)) for row in rows
        ),
        "full_tile_in_multi_plan_count": sum(
            int(row.get("full_tile_in_multi_plan_count", 0)) for row in rows
        ),
        "duplicate_tile_count": sum(int(row.get("duplicate_tile_count", 0)) for row in rows),
        "nested_tile_count": sum(int(row.get("nested_tile_count", 0)) for row in rows),
        "near_full_tile_count": sum(int(row.get("near_full_tile_count", 0)) for row in rows),
        "min_crop_height_ratio": min(
            (float(row.get("min_crop_height_ratio", 1.0)) for row in rows), default=0.0
        ),
        "max_crop_height_ratio": max(
            (float(row.get("max_crop_height_ratio", 1.0)) for row in rows), default=0.0
        ),
        "adjacent_overlap_ratio_mean": mean(
            [float(row.get("adjacent_overlap_ratio_mean", 0.0)) for row in rows]
        ) if rows else 0.0,
        "strict_vertical_partition_failure_count": sum(
            metrics["strict_vertical_partition"] is not True for metrics in partition_metrics
        ),
        "adjacent_overlap_pixels_total": sum(
            int(metrics["adjacent_overlap_pixels_total"]) for metrics in partition_metrics
        ),
        "adjacent_gap_pixels_total": sum(
            int(metrics["adjacent_gap_pixels_total"]) for metrics in partition_metrics
        ),
        "sum_tile_area": sum(int(metrics["sum_tile_area"]) for metrics in partition_metrics),
        "union_tile_area": sum(int(metrics["union_tile_area"]) for metrics in partition_metrics),
        "original_area": sum(int(metrics["original_area"]) for metrics in partition_metrics),
        "duplicate_pixel_area": sum(int(metrics["duplicate_pixel_area"]) for metrics in partition_metrics),
        "area_identity_failure_count": sum(
            not (
                int(metrics["sum_tile_area"])
                == int(metrics["union_tile_area"])
                == int(metrics["original_area"])
            )
            for metrics in partition_metrics
        ),
        "processed_pixel_ratio_not_one_count": sum(
            float(metrics["processed_pixel_ratio"]) != 1.0 for metrics in partition_metrics
        ),
        "tile_count_reduced_image_count": sum(
            int(row.get("actual_tile_count", row["tile_count"]))
            < int(row.get("desired_tile_count", row["tile_count"]))
            for row in rows
        ),
        "lossless_coverage_failure_count": sum(
            float(row["lossless_pixel_coverage_ratio"]) != 1.0 for row in rows
        ),
    }


def _draw_preview(
    image: Image.Image,
    record: Mapping[str, Any],
    output: Path,
    crop_dir: Path | None,
) -> None:
    max_panel_width = 850
    max_crop_width = 1700
    scale = min(1.0, max_panel_width / image.width, 4200 / image.height)
    original_display = image.resize((round(image.width * scale), round(image.height * scale))) if scale < 1 else image.copy()
    display = original_display.copy()
    draw = ImageDraw.Draw(display)
    font = ImageFont.load_default()
    def box(raw: Sequence[int]) -> tuple[int, int, int, int]:
        return tuple(round(int(value) * scale) for value in raw)  # type: ignore[return-value]
    for item in record["text_detections"]:
        draw.rectangle(box(item["bbox"]), outline=(0, 210, 80), width=max(2, round(3 * scale)))
    for item in record["icon_detections"]:
        draw.rectangle(box(item["bbox"]), outline=(255, 145, 0), width=max(2, round(3 * scale)))
    for seam in record.get("horizontal_seams", []):
        y = round(int(seam) * scale)
        dash = max(5, round(14 * scale))
        for x in range(0, display.width, dash * 2):
            draw.line((x, y, min(display.width, x + dash), y), fill=(235, 35, 35), width=max(2, round(3 * scale)))
    for index, tile in enumerate(record["tiles"]):
        draw.rectangle(box(tile), outline=(30, 110, 255), width=max(2, round(4 * scale)))
        draw.text((6, box(tile)[1] + 4), f"crop {index}", fill=(30, 110, 255), font=font)

    crops: list[Image.Image] = []
    for index, tile in enumerate(record["tiles"]):
        crop = image.crop(tuple(tile))
        thumb_scale = min(1.0, max_crop_width / crop.width, 1800 / crop.height)
        thumb = crop.resize((round(crop.width * thumb_scale), round(crop.height * thumb_scale))) if thumb_scale < 1 else crop.copy()
        crops.append(thumb)
        if crop_dir is not None:
            crop_path = crop_dir / f"{record['image_id']}__crop{index:02d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
        crop.close()
    banner_height = 42
    panel_gap = 12
    top_width = original_display.width + panel_gap + display.width
    canvas_height = banner_height + 20 + display.height + sum(item.height + 36 for item in crops)
    canvas = Image.new("RGB", (max(top_width, *(item.width for item in crops)), canvas_height), "white")
    canvas_draw = ImageDraw.Draw(canvas)
    title = (
        f"{record['image_id']} | {record['density']} | text={len(record['text_detections'])} "
        f"icon={len(record['icon_detections'])} | crops={len(record['tiles'])} | lossless=100%"
    )
    canvas_draw.text((8, 10), title, fill="black", font=font)
    y = banner_height
    canvas_draw.text((8, y + 4), "Original", fill="black", font=font)
    canvas.paste(original_display, (0, y + 20))
    canvas_draw.text((original_display.width + panel_gap + 8, y + 4), "Text/Icon + seams + crops", fill="black", font=font)
    canvas.paste(display, (original_display.width + panel_gap, y + 20))
    y += display.height + 20
    for index, thumb in enumerate(crops):
        tile = record["tiles"][index]
        crop_height = int(tile[3]) - int(tile[1])
        canvas_draw.text(
            (8, y + 5),
            f"crop {index}: y=[{tile[1]},{tile[3]}], h={crop_height}, "
            f"height_ratio={crop_height / max(1, image.height):.4f}, "
            f"vertical_gain={image.height / max(1, crop_height):.3f}x",
            fill="black",
            font=font,
        )
        y += 36
        canvas.paste(thumb, (0, y))
        y += thumb.height
        thumb.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    canvas.save(temporary, format="JPEG", quality=92)
    os.replace(temporary, output)
    display.close()
    original_display.close()
    canvas.close()


def _select_visualizations(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count <= 0:
        return []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["density"])].append(row)
    selected: list[Mapping[str, Any]] = []
    remaining = min(count, 60, len(rows))
    for density_index, density in enumerate(("sparse", "medium", "dense")):
        candidates = sorted(
            grouped[density], key=lambda row: (int(row["detector_box_count"]), str(row["image_id"]))
        )
        layers_left = 3 - density_index
        quota = min(20, len(candidates), math.ceil(remaining / layers_left))
        if quota:
            indices = (
                [len(candidates) // 2]
                if quota == 1
                else [round(index * (len(candidates) - 1) / (quota - 1)) for index in range(quota)]
            )
            selected.extend(candidates[index] for index in indices)
            remaining -= quota
    return selected


def _write_v2_v3_coordinate_comparison(
    output_dir: Path,
    crop_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare a preserved v2 scan with v3 without mutating either manifest."""

    old_manifest = output_dir / "horizontal_scan_v2" / "detector_scan_crops.jsonl"
    comparison_path = crop_root / "v2_v3_coordinate_compare.csv"
    if not old_manifest.is_file():
        comparison_path.unlink(missing_ok=True)
        return {"available": False, "compared_images": 0, "path": None}
    old_by_id = {str(row["image_id"]): row for row in read_jsonl(old_manifest)}
    new_by_id = {str(row["image_id"]): row for row in rows}
    if set(old_by_id) != set(new_by_id):
        raise RuntimeError(
            "horizontal_scan_v2/v3 image_id sets differ; refusing a partial coordinate comparison"
        )
    fields = (
        "image_id",
        "width",
        "height",
        "v2_tile_count",
        "v3_tile_count",
        "v2_tiles",
        "v3_tiles",
        "v2_adjacent_overlap_pixels_total",
        "v3_adjacent_overlap_pixels_total",
        "v2_processed_pixel_ratio",
        "v3_processed_pixel_ratio",
        "v2_seam_crossed_detector_bbox_count",
        "v3_seam_crossed_detector_bbox_count",
        "v2_detector_boundary_cut_count",
        "v3_detector_boundary_cut_count",
    )
    temporary = comparison_path.with_name(f".{comparison_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id in sorted(new_by_id):
            old = old_by_id[image_id]
            new = new_by_id[image_id]
            old_metrics = strict_vertical_partition_metrics(
                int(old["width"]), int(old["height"]), old["tiles"]
            )
            new_metrics = strict_vertical_partition_metrics(
                int(new["width"]), int(new["height"]), new["tiles"]
            )
            writer.writerow(
                {
                    "image_id": image_id,
                    "width": new["width"],
                    "height": new["height"],
                    "v2_tile_count": len(old["tiles"]),
                    "v3_tile_count": len(new["tiles"]),
                    "v2_tiles": json.dumps(old["tiles"], separators=(",", ":")),
                    "v3_tiles": json.dumps(new["tiles"], separators=(",", ":")),
                    "v2_adjacent_overlap_pixels_total": old_metrics[
                        "adjacent_overlap_pixels_total"
                    ],
                    "v3_adjacent_overlap_pixels_total": new_metrics[
                        "adjacent_overlap_pixels_total"
                    ],
                    "v2_processed_pixel_ratio": old_metrics["processed_pixel_ratio"],
                    "v3_processed_pixel_ratio": new_metrics["processed_pixel_ratio"],
                    "v2_seam_crossed_detector_bbox_count": old.get(
                        "seam_crossed_detector_bbox_count", ""
                    ),
                    "v3_seam_crossed_detector_bbox_count": new.get(
                        "seam_crossed_detector_bbox_count", ""
                    ),
                    "v2_detector_boundary_cut_count": old.get(
                        "detector_boundary_cut_count", ""
                    ),
                    "v3_detector_boundary_cut_count": new.get(
                        "detector_boundary_cut_count", ""
                    ),
                }
            )
    os.replace(temporary, comparison_path)
    return {
        "available": True,
        "compared_images": len(rows),
        "path": str(comparison_path),
    }


def _cache_file_record(cache_dir: Path, path: Path, *, jsonl_rows: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        recorded_path = str(resolved.relative_to(cache_dir.resolve(strict=True)))
    except ValueError:
        recorded_path = str(resolved)
    record: dict[str, Any] = {"path": recorded_path, "sha256": sha256_file(resolved)}
    if jsonl_rows is not None:
        record["jsonl_rows"] = int(jsonl_rows)
    return record


def _geometry_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    conditions = {
        "strict_vertical_partition_true": int(overall["strict_vertical_partition_failure_count"]) == 0,
        "adjacent_overlap_pixels_total_zero": int(overall["adjacent_overlap_pixels_total"]) == 0,
        "adjacent_gap_pixels_total_zero": int(overall["adjacent_gap_pixels_total"]) == 0,
        "duplicate_pixel_area_zero": int(overall["duplicate_pixel_area"]) == 0,
        "sum_tile_area_equals_union_tile_area": int(overall["sum_tile_area"]) == int(overall["union_tile_area"]),
        "union_tile_area_equals_original_area": int(overall["union_tile_area"]) == int(overall["original_area"]),
        "processed_pixel_ratio_equals_1": int(overall["processed_pixel_ratio_not_one_count"]) == 0,
        "lossless_pixel_coverage_equals_1": int(overall["lossless_coverage_failure_count"]) == 0,
        "detector_bbox_containment_equals_1": (
            float(overall["detector_bbox_containment_rate"]) == 1.0
            and int(overall["uncontained_detector_bbox_count"]) == 0
        ),
        "every_detector_bbox_belongs_to_exactly_one_crop": (
            int(overall["detector_bbox_unique_containment_count"])
            == int(overall["detector_bbox_count"])
        ),
        "full_tile_in_multi_plan_count_zero": int(overall["full_tile_in_multi_plan_count"]) == 0,
        "duplicate_tile_count_zero": int(overall["duplicate_tile_count"]) == 0,
        "nested_tile_count_zero": int(overall["nested_tile_count"]) == 0,
        "seam_crossed_detector_bbox_count_zero": int(overall["seam_crossed_detector_bbox_count"]) == 0,
        "detector_boundary_cut_count_zero": int(overall["detector_boundary_cut_count"]) == 0,
        "balanced_fallback_seam_count_zero": int(overall["balanced_fallback_seam_count"]) == 0,
        "gt_not_used": summary.get("gt_used") is False,
    }
    return {"conditions": conditions, "passes": all(conditions.values())}


def _resolve_cache_scope(
    args: argparse.Namespace, selection: Mapping[str, Any], unique_count: int
) -> tuple[str, int, int]:
    max_images_per_task = int(selection.get("max_images_per_task", 0))
    inferred = "preview" if max_images_per_task > 0 else "full_test"
    requested = str(getattr(args, "cache_scope", "auto"))
    cache_scope = inferred if requested == "auto" else requested
    if cache_scope != inferred:
        raise RuntimeError(
            f"cache scope contradicts prepared manifest: requested={cache_scope}, "
            f"max_images_per_task={max_images_per_task}"
        )
    expected = (
        unique_count
        if cache_scope == "preview"
        else int(getattr(args, "expected_full_test_unique_images", 17281))
    )
    if unique_count != expected:
        raise RuntimeError(
            f"{cache_scope} cache unique image count mismatch: {unique_count} != {expected}"
        )
    return cache_scope, max_images_per_task, expected


def _write_cache_ready_marker(
    args: argparse.Namespace,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    crop_root: Path,
    geometry_config: Mapping[str, Any],
) -> dict[str, Any]:
    paths = AuditPaths(args.output_dir)
    selection_path = paths.manifest / "selection_config.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cache_scope, max_images_per_task, expected_unique_images = _resolve_cache_scope(
        args, selection, len(rows)
    )
    task_files = []
    for task in TASKS:
        path = Path(selection["task_files"][task])
        task_files.append({"task": task, **_cache_file_record(args.output_dir, path)})

    detector_cfg = json.loads(paths.detector_config.read_text(encoding="utf-8"))
    detector_section: dict[str, Any] = {
        "parser_commit": detector_cfg["parser_commit"],
        "config": detector_cfg,
        "config_digest": cache_json_digest(detector_cfg),
        "config_file": _cache_file_record(args.output_dir, paths.detector_config),
        "merged_detections": _cache_file_record(
            args.output_dir, paths.merged, jsonl_rows=len(rows)
        ),
    }
    for stage in ("text", "icon"):
        stage_path = paths.stage_dir(stage) / "stage_summary.json"
        if not stage_path.is_file():
            raise FileNotFoundError(
                f"cannot publish readonly cache without {stage} runtime summary: {stage_path}"
            )
        stage_summary = json.loads(stage_path.read_text(encoding="utf-8"))
        shard_records = [
            _cache_file_record(
                args.output_dir, shard_path, jsonl_rows=len(read_jsonl(shard_path))
            )
            for shard_path in sorted(paths.stage_dir(stage).glob("shard_*.jsonl"))
        ]
        done_records = [
            _cache_file_record(args.output_dir, done_path)
            for done_path in sorted(paths.stage_dir(stage).glob("shard_*.done.json"))
        ]
        if (
            not shard_records
            or len(shard_records) != len(done_records)
            or sum(int(record["jsonl_rows"]) for record in shard_records) != len(rows)
            or int(stage_summary.get("images", -1)) != len(rows)
            or not stage_summary.get("runtime")
        ):
            raise RuntimeError(
                f"cannot publish readonly cache: {stage} shard/done summary is incomplete"
            )
        detector_section[f"{stage}_stage_summary"] = {
            "file": _cache_file_record(args.output_dir, stage_path),
            "images": int(stage_summary.get("images", -1)),
            "workers": int(stage_summary.get("workers", 0)),
            "runtime": stage_summary.get("runtime", {}),
            "completed_shards": len(list(paths.stage_dir(stage).glob("shard_*.done.json"))),
            "shards": shard_records,
            "done_markers": done_records,
            "shard_manifest_digest": cache_json_digest(shard_records),
        }

    gate = summary["geometry_gate"]
    if gate.get("passes") is not True:
        raise RuntimeError(
            "horizontal scan reports were written, but cache ready marker is withheld; "
            f"failed gates={[name for name, passed in gate['conditions'].items() if not passed]}"
        )
    if str(summary.get("cache_scope")) != cache_scope:
        raise RuntimeError(
            "summary cache_scope changed before ready marker publication: "
            f"summary={summary.get('cache_scope')}, resolved={cache_scope}"
        )
    marker = {
        "schema_version": CACHE_MARKER_SCHEMA_VERSION,
        "ready": True,
        "scan_name": args.scan_name,
        "cache_scope": cache_scope,
        "max_images_per_task": max_images_per_task,
        "expected_unique_images": expected_unique_images,
        "strict_vertical_partition": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_after_all_checks": True,
        "gt_used": False,
        "dataset": {
            "task_files": task_files,
            "content_unique_images": len(rows),
            "image_id_digest": digest_ids(row["image_id"] for row in rows),
            "content_id_digest": digest_ids(row["content_id"] for row in rows),
            "unique_manifest": _cache_file_record(
                args.output_dir, paths.unique_images, jsonl_rows=len(rows)
            ),
            "task_manifest": _cache_file_record(
                args.output_dir,
                paths.task_samples,
                jsonl_rows=len(read_jsonl(paths.task_samples)),
            ),
        },
        "detector": detector_section,
        "geometry": {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "config": geometry_config,
            "config_digest": cache_json_digest(geometry_config),
            "gate_passes": True,
            "strict_vertical_partition": True,
            "scan_manifest": _cache_file_record(
                args.output_dir,
                crop_root / "detector_scan_crops.jsonl",
                jsonl_rows=len(rows),
            ),
            "summary": _cache_file_record(args.output_dir, crop_root / "summary.json"),
            "statistics": _cache_file_record(args.output_dir, crop_root / "statistics.csv"),
            "gallery": _cache_file_record(args.output_dir, crop_root / "gallery" / "index.html"),
        },
    }
    comparison_path = crop_root / "v2_v3_coordinate_compare.csv"
    if comparison_path.is_file():
        marker["geometry"]["v2_v3_coordinate_comparison"] = _cache_file_record(
            args.output_dir, comparison_path
        )
    atomic_write_json(cache_marker_path(args.output_dir, args.scan_name), marker)
    return marker


def build_scan_crops(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = AuditPaths(args.output_dir)
    geometry_config = {
        "max_crops": args.scan_max_crops,
        "target_tile_height": args.scan_target_height,
        "legacy_ignored_parameters": {
            "overlap_ratio": args.scan_overlap_ratio,
            "vertical_link_ratio": args.scan_vertical_link_ratio,
            "context_ratio": args.scan_context_ratio,
            "min_context_image_ratio": args.scan_min_context_image_ratio,
            "dense_band_ratio": args.scan_dense_band_ratio,
            "seam_search_ratio": args.scan_seam_search_ratio,
        },
        "detector_margin_ratio": args.scan_detector_margin_ratio,
        "detector_margin_pixels": [2, 12],
        "context_pixels": args.scan_context_pixels,
        "strict_vertical_partition": args.strict_vertical_partition,
        "seam_selection": "global_detector_free_gap_dynamic_programming",
        "minimum_core_height_ratio": args.scan_minimum_core_height_ratio,
        "horizontal_extent": "full_image_width",
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "gt_used": False,
    }
    crop_root = args.output_dir / args.scan_name
    state_path = crop_root / "scan_state.json"
    scan_manifest_path = crop_root / "detector_scan_crops.jsonl"
    current_state = {
        "format_version": SCAN_FORMAT_VERSION,
        "scan_name": args.scan_name,
        "merged_detection_digest": _file_digest(paths.merged),
        "unique_manifest_digest": _file_digest(paths.unique_images),
        "task_manifest_digest": _file_digest(paths.task_samples),
        "geometry_config": geometry_config,
        "visualization_samples": args.visualization_samples,
        "save_preview_crops": args.save_preview_crops,
    }
    required_outputs = [
        scan_manifest_path,
        crop_root / "summary.json",
        crop_root / "statistics.csv",
        crop_root / "gallery" / "index.html",
    ]
    if (args.output_dir / "horizontal_scan_v2" / "detector_scan_crops.jsonl").is_file():
        required_outputs.append(crop_root / "v2_v3_coordinate_compare.csv")
    if getattr(args, "resume", False) and state_path.is_file() and all(
        path.is_file() for path in required_outputs
    ):
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        if previous_state == current_state:
            cached = read_jsonl(scan_manifest_path)
            ready_path = cache_marker_path(args.output_dir, args.scan_name)
            # A ready marker is always the final atomic publication.  Rebuild
            # it from the digest-bound reports even on a geometry resume, so a
            # stale/partial marker can never survive a successful invocation.
            ready_path.unlink(missing_ok=True)
            cached_summary = json.loads(
                (crop_root / "summary.json").read_text(encoding="utf-8")
            )
            _write_cache_ready_marker(
                args,
                rows=cached,
                summary=cached_summary,
                crop_root=crop_root,
                geometry_config=geometry_config,
            )
            print(
                f"[eval crop] --resume validated {len(cached)} cached scan plans; "
                "skip geometry, preview rendering and crop PNG writes",
                flush=True,
            )
            ProgressReporter(
                stage="eval-crop",
                total=len(cached),
                output_dir=args.output_dir,
                interval_seconds=float(getattr(args, "progress_interval_seconds", 10.0)),
                initial_completed=len(cached),
                unit="images",
            ).update(
                len(cached),
                status="completed",
                detail="--resume 参数与 merged detection digest 一致，全部复用",
                force=True,
            )
            return cached

    cache_marker_path(args.output_dir, args.scan_name).unlink(missing_ok=True)
    merged = read_jsonl(paths.merged)
    unique_rows = read_jsonl(paths.unique_images)
    unique = {row["image_id"]: row for row in unique_rows}
    if len(merged) != len(unique):
        raise ValueError(f"merged/manifest count mismatch: {len(merged)} != {len(unique)}")
    rows: list[dict[str, Any]] = []
    reporter = ProgressReporter(
        stage="eval-crop",
        total=len(merged),
        output_dir=args.output_dir,
        interval_seconds=float(getattr(args, "progress_interval_seconds", 10.0)),
        unit="images",
    )
    reporter.update(0, detail="纯 CPU 严格无重叠水平分区；GT disabled", force=True)
    for detected_index, detected in enumerate(merged, 1):
        manifest = unique[detected["image_id"]]
        detector_items = [
            {**item, "source": source}
            for source, key in (("text", "text_detections"), ("icon", "icon_detections"))
            for item in detected[key]
        ]
        plan = generate_detector_scan_plan(
            int(detected["width"]),
            int(detected["height"]),
            detector_items,
            max_tiles=args.scan_max_crops,
            target_tile_height=args.scan_target_height,
            overlap_ratio=args.scan_overlap_ratio,
            vertical_link_ratio=args.scan_vertical_link_ratio,
            context_ratio=args.scan_context_ratio,
            min_context_image_ratio=args.scan_min_context_image_ratio,
            dense_band_ratio=args.scan_dense_band_ratio,
            detector_margin_ratio=args.scan_detector_margin_ratio,
            seam_search_ratio=args.scan_seam_search_ratio,
            context_pixels=args.scan_context_pixels,
            minimum_core_height_ratio=args.scan_minimum_core_height_ratio,
            strict_vertical_partition=args.strict_vertical_partition,
        )
        row = {
            "image_id": detected["image_id"],
            "content_id": detected["content_id"],
            "image_path": manifest["image_path"],
            "image_paths": manifest["image_paths"],
            "tasks": manifest["tasks"],
            "width": detected["width"],
            "height": detected["height"],
            "text_detections": detected["text_detections"],
            "icon_detections": detected["icon_detections"],
            "density": _density(len(detector_items)),
            "geometry_config_digest": _json_digest(geometry_config),
            **plan,
        }
        rows.append(row)
        reporter.update(
            detected_index,
            detail=(
                f"{row['density']}，{row['detector_box_count']} boxes -> "
                f"{row['tile_count']} horizontal scans"
            ),
        )
    rows.sort(key=lambda row: row["image_id"])

    atomic_write_jsonl(scan_manifest_path, rows)
    coordinate_comparison = _write_v2_v3_coordinate_comparison(
        args.output_dir, crop_root, rows
    )
    by_density = {
        density: _metric_summary([row for row in rows if row["density"] == density])
        for density in ("sparse", "medium", "dense")
    }
    by_task: dict[str, Any] = {}
    for task in TASKS:
        task_rows = [row for row in rows if task in row["tasks"]]
        if task == "content_missing":
            # The inference worker intentionally overrides these task-agnostic
            # proposals with one complete global view.
            effective_rows = [
                {
                    **row,
                    "tile_count": 1,
                    "desired_tile_count": 1,
                    "actual_tile_count": 1,
                    "processed_pixel_ratio_with_overlap": 1.0,
                    "processed_pixel_ratio": 1.0,
                    "mean_vertical_linear_gain": 1.0,
                    "detector_boundary_cut_count": 0,
                    "lossless_pixel_coverage_ratio": 1.0,
                    "horizontal_seam_count": 0,
                    "seam_source": [],
                    "full_tile_in_multi_plan_count": 0,
                    "duplicate_tile_count": 0,
                    "nested_tile_count": 0,
                    "near_full_tile_count": 1,
                    "min_crop_height_ratio": 1.0,
                    "max_crop_height_ratio": 1.0,
                    "adjacent_overlap_ratio_mean": 0.0,
                }
                for row in task_rows
            ]
        else:
            effective_rows = task_rows
        by_task[task] = {
            **_metric_summary(effective_rows),
            "effective_mode": "full_image_global_view"
            if task == "content_missing"
            else "detector_scan",
        }
    selected = _select_visualizations(rows, args.visualization_samples)
    gallery_density_counts = Counter(str(row["density"]) for row in selected)
    selection = json.loads((paths.manifest / "selection_config.json").read_text(encoding="utf-8"))
    selection_limit = int(selection.get("max_images_per_task", 0))
    inferred_scope = "preview" if selection_limit > 0 else "full_test"
    cache_scope = inferred_scope if args.cache_scope == "auto" else args.cache_scope
    summary = {
        "format_version": SCAN_FORMAT_VERSION,
        "scan_name": args.scan_name,
        "mode": "detector_scan",
        "description": "GT-free strict non-overlapping full-width horizontal seam partition",
        "cache_scope": cache_scope,
        "max_images_per_task": selection_limit,
        "unique_images": len(rows),
        "image_id_digest": digest_ids(row["image_id"] for row in rows),
        "geometry_config": geometry_config,
        "overall": _metric_summary(rows),
        "by_density": by_density,
        "by_task": by_task,
        "task_agnostic_geometry_statistics": _metric_summary(rows),
        "content_missing_effective_statistics": by_task.get("content_missing", {}),
        "gallery_selection": {
            "requested": args.visualization_samples,
            "selected": len(selected),
            "pool_images": len(rows),
            "by_density": {
                density: {
                    "available": sum(row["density"] == density for row in rows),
                    "selected": gallery_density_counts.get(density, 0),
                }
                for density in ("sparse", "medium", "dense")
            },
        },
        "v2_v3_coordinate_comparison": coordinate_comparison,
        "raw_detector_files_unchanged": True,
        "gt_used": False,
    }
    summary["geometry_gate"] = _geometry_gate(summary)
    atomic_write_json(crop_root / "summary.json", summary)
    csv_path = crop_root / "statistics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = [
        "image_id", "image_path", "width", "height", "density", "detector_box_count",
        "connected_band_count", "tile_count", "lossless_pixel_coverage_ratio",
        "horizontal_seam_count", "horizontal_seams", "seam_source_counts",
        "strict_vertical_partition", "adjacent_overlap_pixels",
        "adjacent_overlap_pixels_total", "adjacent_gap_pixels",
        "adjacent_gap_pixels_total", "sum_tile_area", "union_tile_area",
        "original_area", "duplicate_pixel_area", "processed_pixel_ratio",
        "safe_seam_count", "desired_tile_count", "actual_tile_count",
        "tile_count_reduction_reason", "balanced_fallback_seam_count",
        "detector_bbox_containment_rate", "detector_bbox_unique_containment_count",
        "uncontained_detector_bbox_count",
        "seam_crossed_detector_bbox_count", "full_tile_in_multi_plan_count",
        "duplicate_tile_count", "nested_tile_count", "minimum_core_height",
        "maximum_core_height", "core_height_ratio", "min_crop_height_ratio",
        "max_crop_height_ratio", "adjacent_overlap_ratio_mean",
        "processed_pixel_ratio_with_overlap", "mean_vertical_linear_gain",
        "near_full_tile_count", "detector_boundary_cut_count", "fallback_reason",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    os.replace(temporary, csv_path)

    visualization_dir = crop_root / "visualizations"
    crop_dir = crop_root / "preview_crops" if args.save_preview_crops else None
    links: list[tuple[str, str, str]] = []
    for row in selected:
        image_path = Path(row["image_path"])
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        output = visualization_dir / f"{row['image_id']}__detector_scan.jpg"
        try:
            _draw_preview(image, row, output, crop_dir)
        finally:
            image.close()
        links.append((row["image_id"], row["density"], output.name))
    gallery = crop_root / "gallery" / "index.html"
    gallery.parent.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(
        f'<article data-density="{html.escape(density)}"><h3>{html.escape(image_id)} · {html.escape(density)}</h3>'
        f'<a href="../visualizations/{html.escape(name)}"><img loading="lazy" src="../visualizations/{html.escape(name)}"></a></article>'
        for image_id, density, name in links
    )
    gallery.write_text(
        "<!doctype html><meta charset='utf-8'><title>UI5 detector scan previews</title>"
        "<style>body{font-family:sans-serif;margin:20px}nav button{margin:4px}article{margin:20px 0}img{max-width:100%;border:1px solid #ccc}</style>"
        "<h1>UI5 测试集 detector-scan crops（GT 未使用）</h1>"
        "<nav><button onclick=\"f('all')\">全部</button><button onclick=\"f('sparse')\">sparse</button>"
        "<button onclick=\"f('medium')\">medium</button><button onclick=\"f('dense')\">dense</button></nav>"
        + cards
        + "<script>function f(x){document.querySelectorAll('article').forEach(e=>e.style.display=(x==='all'||e.dataset.density===x)?'block':'none')}</script>",
        encoding="utf-8",
    )
    # Geometry state is written after all reports; the readonly marker is the
    # final atomic publication and is withheld on any failed check.
    atomic_write_json(state_path, current_state)
    _write_cache_ready_marker(
        args,
        rows=rows,
        summary=summary,
        crop_root=crop_root,
        geometry_config=geometry_config,
    )
    print(
        f"[eval crop] images={len(rows)}, previews={len(selected)}, "
        f"bbox containment={summary['overall']['detector_bbox_containment_rate']:.6f}, "
        f"processed mean={summary['overall']['processed_pixel_ratio_with_overlap_mean']:.4f}, "
        f"summary={crop_root / 'summary.json'}",
        flush=True,
    )
    reporter.update(
        len(rows),
        status="completed",
        detail=f"预览 {len(selected)} 张，bbox containment=100%，readonly marker=ready",
        force=True,
    )
    return rows


def validate_args(args: argparse.Namespace) -> None:
    if args.max_images_per_task < 0:
        raise ValueError("--max-images-per-task cannot be negative")
    if not 1 <= args.scan_max_crops <= 10:
        raise ValueError("--scan-max-crops must be in [1, 10]")
    if args.scan_target_height <= 0:
        raise ValueError("--scan-target-height must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(args.scan_name)):
        raise ValueError("--scan-name must be one safe directory name")
    if not 0 < args.scan_overlap_ratio < 1:
        raise ValueError("--scan-overlap-ratio must be in (0, 1)")
    for name in (
        "scan_vertical_link_ratio",
        "scan_context_ratio",
        "scan_min_context_image_ratio",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if not 0 < args.scan_dense_band_ratio <= 1:
        raise ValueError("--scan-dense-band-ratio must be in (0, 1]")
    if not 0 <= args.scan_detector_margin_ratio <= 0.05:
        raise ValueError("--scan-detector-margin-ratio must be in [0, 0.05]")
    if not 0 <= args.scan_seam_search_ratio <= 0.5:
        raise ValueError("--scan-seam-search-ratio must be in [0, 0.5]")
    if args.scan_context_pixels != 0:
        raise ValueError("--scan-context-pixels must be 0 for strict vertical partition")
    if args.strict_vertical_partition is not True:
        raise ValueError("--strict-vertical-partition is mandatory for schema-v3 scans")
    if not 0 < args.scan_minimum_core_height_ratio <= 1:
        raise ValueError("--scan-minimum-core-height-ratio must be in (0, 1]")
    if args.visualization_samples < 0:
        raise ValueError("--visualization-samples cannot be negative")
    if args.expected_full_test_unique_images <= 0:
        raise ValueError("--expected-full-test-unique-images must be positive")
    if args.shard_size <= 0 or args.image_loader_threads <= 0:
        raise ValueError("--shard-size and --image-loader-threads must be positive")
    if args.workers_per_gpu == 2 and not args.allow_two_processes_per_gpu:
        raise ValueError("2 processes/GPU requires --allow-two-processes-per-gpu")
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    args.parser_root = args.parser_root.expanduser().resolve(strict=True)
    args.icon_model = args.icon_model.expanduser().resolve(strict=False) if args.icon_model else None
    args.text_model_dir = args.text_model_dir.expanduser().resolve(strict=False) if args.text_model_dir else None
    if args.stage in {"all", "text"} and not args.text_python:
        raise ValueError(
            f"--stage {args.stage} requires explicit --text-python; do not run PaddleOCR in the main/icon environment"
        )
    if args.stage in {"all", "icon"} and not args.icon_python:
        raise ValueError(
            f"--stage {args.stage} requires explicit --icon-python; do not run the icon detector in the Paddle environment"
        )
    if args.text_python:
        args.text_python = resolve_python_executable(args.text_python, "--text-python")
    if args.icon_python:
        args.icon_python = resolve_python_executable(args.icon_python, "--icon-python")
    if args.stage == "all" and os.path.normcase(str(args.text_python)) == os.path.normcase(str(args.icon_python)):
        raise ValueError(
            "--stage all requires separate text/icon Python environments; the two resolved paths are identical"
        )
    if args.stage == "all" and Path(str(args.icon_python)).resolve() != Path(sys.executable).resolve():
        raise ValueError(
            "--stage all must be launched by the LocateAnything/icon Python: "
            f"launcher={sys.executable}, icon_python={args.icon_python}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.stage == "_worker":
        run_detector_worker(args)
        return 0
    stages = ("prepare", "text", "icon", "merge", "crop") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"[eval detector pipeline] stage={stage}", flush=True)
        if stage == "prepare":
            prepare_manifest(args)
        elif stage in {"text", "icon"}:
            run_detection_stage(args, stage)
        elif stage == "merge":
            merge_detections(args)
        elif stage == "crop":
            build_scan_crops(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
