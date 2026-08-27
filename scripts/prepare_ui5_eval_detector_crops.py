#!/usr/bin/env python3
"""Prepare GT-free PP-OCR/icon detector crops for UI5 evaluation.

The two GPU detectors run once per unique test image and write resumable shard
JSONL.  Geometry is CPU-only: it turns the merged detections into full-width,
overlapping horizontal scan crops whose union covers the complete image and
whose boundaries never cross a detector box.  No annotation or GT field is
read by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
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
    run_detection_stage,
    run_detector_worker,
)
from ui5_lossless_tiling import generate_detector_scan_plan


FORMAT_VERSION = 1


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
    parser.add_argument("--scan-max-crops", type=int, default=10)
    parser.add_argument("--scan-target-height", type=int, default=960)
    parser.add_argument("--scan-overlap-ratio", type=float, default=0.12)
    parser.add_argument("--scan-vertical-link-ratio", type=float, default=0.025)
    parser.add_argument("--scan-context-ratio", type=float, default=0.20)
    parser.add_argument("--scan-min-context-image-ratio", type=float, default=0.015)
    parser.add_argument("--scan-dense-band-ratio", type=float, default=0.80)
    parser.add_argument("--visualization-samples", type=int, default=20)
    parser.add_argument("--save-preview-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detector-stage", choices=("text", "icon"), help=argparse.SUPPRESS)
    # Accepted only because the shared audit worker still emits them for its own entrypoint.
    parser.add_argument("--source-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--locany-data-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.text_python = args.text_python or sys.executable
    args.icon_python = args.icon_python or sys.executable
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
        "format_version": FORMAT_VERSION,
        "input_dir": str(input_dir),
        "task_files": {task: str(path) for task, path in task_files.items()},
        "max_images_per_task": args.max_images_per_task,
        "skip_figma": args.skip_figma,
        "unique_images": len(unique),
        "image_id_digest": digest_ids(row["image_id"] for row in unique),
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
    processed = [float(row["processed_pixel_ratio_with_overlap"]) for row in rows]
    gains = [float(row["mean_vertical_linear_gain"]) for row in rows]
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
        "mean_vertical_linear_gain": mean(gains) if gains else 0.0,
        "single_full_image_count": sum(int(row["tile_count"]) == 1 for row in rows),
        "detector_empty_count": sum(int(row["detector_box_count"]) == 0 for row in rows),
        "detector_boundary_cut_count": sum(int(row["detector_boundary_cut_count"]) for row in rows),
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
    max_width = 1500
    scale = min(1.0, max_width / image.width, 5000 / image.height)
    display = image.resize((round(image.width * scale), round(image.height * scale))) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(display)
    font = ImageFont.load_default()
    def box(raw: Sequence[int]) -> tuple[int, int, int, int]:
        return tuple(round(int(value) * scale) for value in raw)  # type: ignore[return-value]
    for item in record["text_detections"]:
        draw.rectangle(box(item["bbox"]), outline=(0, 210, 80), width=max(2, round(3 * scale)))
    for item in record["icon_detections"]:
        draw.rectangle(box(item["bbox"]), outline=(255, 145, 0), width=max(2, round(3 * scale)))
    for index, tile in enumerate(record["tiles"]):
        draw.rectangle(box(tile), outline=(30, 110, 255), width=max(2, round(4 * scale)))
        draw.text((6, box(tile)[1] + 4), f"crop {index}", fill=(30, 110, 255), font=font)

    crops: list[Image.Image] = []
    for index, tile in enumerate(record["tiles"]):
        crop = image.crop(tuple(tile))
        thumb_scale = min(1.0, max_width / crop.width, 1800 / crop.height)
        thumb = crop.resize((round(crop.width * thumb_scale), round(crop.height * thumb_scale))) if thumb_scale < 1 else crop.copy()
        crops.append(thumb)
        if crop_dir is not None:
            crop_path = crop_dir / f"{record['image_id']}__crop{index:02d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
        crop.close()
    banner_height = 42
    canvas_height = banner_height + display.height + sum(item.height + 28 for item in crops)
    canvas = Image.new("RGB", (max(display.width, *(item.width for item in crops)), canvas_height), "white")
    canvas_draw = ImageDraw.Draw(canvas)
    title = (
        f"{record['image_id']} | {record['density']} | text={len(record['text_detections'])} "
        f"icon={len(record['icon_detections'])} | crops={len(record['tiles'])} | lossless=100%"
    )
    canvas_draw.text((8, 10), title, fill="black", font=font)
    y = banner_height
    canvas.paste(display, (0, y))
    y += display.height
    for index, thumb in enumerate(crops):
        canvas_draw.text((8, y + 5), f"crop {index}: {record['tiles'][index]}", fill="black", font=font)
        y += 28
        canvas.paste(thumb, (0, y))
        y += thumb.height
        thumb.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    canvas.save(temporary, format="JPEG", quality=92)
    os.replace(temporary, output)
    display.close()
    canvas.close()


def _select_visualizations(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count <= 0:
        return []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["density"])].append(row)
    selected: list[Mapping[str, Any]] = []
    while len(selected) < min(count, len(rows)):
        changed = False
        for density in ("sparse", "medium", "dense"):
            candidates = grouped[density]
            if candidates:
                selected.append(candidates.pop(len(candidates) // 2))
                changed = True
                if len(selected) >= min(count, len(rows)):
                    break
        if not changed:
            break
    return selected


def build_scan_crops(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = AuditPaths(args.output_dir)
    geometry_config = {
        "max_crops": args.scan_max_crops,
        "target_tile_height": args.scan_target_height,
        "overlap_ratio": args.scan_overlap_ratio,
        "vertical_link_ratio": args.scan_vertical_link_ratio,
        "context_ratio": args.scan_context_ratio,
        "min_context_image_ratio": args.scan_min_context_image_ratio,
        "dense_band_ratio": args.scan_dense_band_ratio,
        "horizontal_extent": "full_image_width",
        "gt_used": False,
    }
    crop_root = args.output_dir / "scan_crops"
    state_path = crop_root / "scan_state.json"
    scan_manifest_path = crop_root / "detector_scan_crops.jsonl"
    current_state = {
        "format_version": FORMAT_VERSION,
        "merged_detection_digest": _file_digest(paths.merged),
        "unique_manifest_digest": _file_digest(paths.unique_images),
        "task_manifest_digest": _file_digest(paths.task_samples),
        "geometry_config": geometry_config,
        "visualization_samples": args.visualization_samples,
        "save_preview_crops": args.save_preview_crops,
    }
    required_outputs = (
        scan_manifest_path,
        crop_root / "summary.json",
        crop_root / "statistics.csv",
        crop_root / "gallery" / "index.html",
    )
    if getattr(args, "resume", False) and state_path.is_file() and all(
        path.is_file() for path in required_outputs
    ):
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        if previous_state == current_state:
            cached = read_jsonl(scan_manifest_path)
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
    reporter.update(0, detail="纯 CPU 横向连通扫描几何；GT disabled", force=True)
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
                    "processed_pixel_ratio_with_overlap": 1.0,
                    "mean_vertical_linear_gain": 1.0,
                    "detector_boundary_cut_count": 0,
                    "lossless_pixel_coverage_ratio": 1.0,
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
    summary = {
        "format_version": FORMAT_VERSION,
        "mode": "detector_scan",
        "description": "GT-free full-width horizontal connected-band scan",
        "unique_images": len(rows),
        "image_id_digest": digest_ids(row["image_id"] for row in rows),
        "geometry_config": geometry_config,
        "overall": _metric_summary(rows),
        "by_density": by_density,
        "by_task": by_task,
        "raw_detector_files_unchanged": True,
        "gt_used": False,
    }
    atomic_write_json(crop_root / "summary.json", summary)
    csv_path = crop_root / "statistics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = [
        "image_id", "image_path", "width", "height", "density", "detector_box_count",
        "connected_band_count", "tile_count", "lossless_pixel_coverage_ratio",
        "processed_pixel_ratio_with_overlap", "mean_vertical_linear_gain",
        "detector_boundary_cut_count", "fallback_reason",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    os.replace(temporary, csv_path)

    selected = _select_visualizations(rows, args.visualization_samples)
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
    # Completion state is deliberately last: an interrupted report/preview
    # refresh can never make the next --resume accept partial outputs.
    atomic_write_json(state_path, current_state)
    print(
        f"[eval crop] images={len(rows)}, previews={len(selected)}, "
        f"boundary cuts={summary['overall']['detector_boundary_cut_count']}, "
        f"summary={crop_root / 'summary.json'}",
        flush=True,
    )
    reporter.update(
        len(rows),
        status="completed",
        detail=f"预览 {len(selected)} 张，detector boundary cuts=0",
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
    if args.visualization_samples < 0:
        raise ValueError("--visualization-samples cannot be negative")
    if args.shard_size <= 0 or args.image_loader_threads <= 0:
        raise ValueError("--shard-size and --image-loader-threads must be positive")
    if args.workers_per_gpu == 2 and not args.allow_two_processes_per_gpu:
        raise ValueError("2 processes/GPU requires --allow-two-processes-per-gpu")
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    args.parser_root = args.parser_root.expanduser().resolve(strict=True)
    args.icon_model = args.icon_model.expanduser().resolve(strict=False) if args.icon_model else None
    args.text_model_dir = args.text_model_dir.expanduser().resolve(strict=False) if args.text_model_dir else None
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
