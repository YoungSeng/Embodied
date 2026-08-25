#!/usr/bin/env python3
"""Task-aware, resumable UI5 detector/crop audit.

The GPU stages write immutable task-agnostic detections.  Every crop parameter
comparison after ``merge`` is CPU-only and associates proposals with each
``image x task`` annotation independently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from analyze_ui5_source_overlap import (
    TASK_NAMES,
    analyze as analyze_overlap,
    assistant_answer,
    content_fingerprint,
    resolve_training_image,
)
from prepare_ui_defect_locany import TASKS


BASELINE_COMMIT = "c06f1479a11b0175579994b880466b57bba50a87"
PARSER_COMMIT = "06eaebf8eb4ea01e61b690f2ff972bf614915918"
BOX_PATTERN = re.compile(
    r"<box>\s*<(-?\d+(?:\.\d+)?)>\s*<(-?\d+(?:\.\d+)?)>\s*"
    r"<(-?\d+(?:\.\d+)?)>\s*<(-?\d+(?:\.\d+)?)>\s*</box>"
)
CONFIGS = {
    "A": {"horizontal_link_ratio": 0.015, "vertical_link_ratio": 0.015, "context_ratio": 0.10},
    "B": {"horizontal_link_ratio": 0.015, "vertical_link_ratio": 0.015, "context_ratio": 0.20},
    "C": {"horizontal_link_ratio": 0.025, "vertical_link_ratio": 0.025, "context_ratio": 0.20},
}
TASK_LABELS = {task["name"]: task["en"] for task in TASKS}
PIPELINE_STAGES = ("prepare", "text", "icon", "merge", "crop-audit")


@dataclass(frozen=True)
class AuditPaths:
    output: Path

    @property
    def manifest(self) -> Path:
        return self.output / "manifest"

    @property
    def unique_images(self) -> Path:
        return self.manifest / "unique_images.jsonl"

    @property
    def task_samples(self) -> Path:
        return self.manifest / "task_samples.jsonl"

    @property
    def shards(self) -> Path:
        return self.manifest / "shards"

    @property
    def detections(self) -> Path:
        return self.output / "detections"

    def stage_dir(self, stage: str) -> Path:
        return self.detections / stage

    @property
    def merged(self) -> Path:
        return self.detections / "merged" / "detections.jsonl"

    @property
    def detector_config(self) -> Path:
        return self.detections / "detector_config.json"

    @property
    def crop_audit(self) -> Path:
        return self.output / "crop_audit"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "text", "icon", "merge", "crop-audit", "_worker"),
        default="all",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--locany-data-dir", type=Path, required=True)
    parser.add_argument("--parser-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2), default=1)
    parser.add_argument("--allow-two-processes-per-gpu", action="store_true")
    parser.add_argument("--image-loader-threads", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=750)
    parser.add_argument(
        "--max-unique-images",
        type=int,
        default=0,
        help="0 means all; use 2000 with a separate output directory for process/GPU benchmarking",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=10.0,
        help="Consolidated terminal/run_status.json update interval",
    )
    parser.add_argument(
        "--progress-every-images",
        type=int,
        default=25,
        help="Detector worker progress checkpoint frequency",
    )
    parser.add_argument(
        "--text-model-dir",
        type=Path,
        default=None,
        help="Optional local PP-OCRv5 directory; omitted means PaddleOCR auto-download/cache",
    )
    parser.add_argument("--icon-model", type=Path, default=None)
    parser.add_argument("--text-long-side", type=int, default=1920)
    parser.add_argument("--text-box-threshold", type=float, default=0.3)
    parser.add_argument("--icon-long-side", type=int, default=1920)
    parser.add_argument("--icon-confidence", type=float, default=0.05)
    parser.add_argument("--max-crops", type=int, default=10)
    parser.add_argument("--boundary-margin-ratio", type=float, default=0.01)
    parser.add_argument("--whole-image-trim-ratio", type=float, default=0.01)
    parser.add_argument("--whole-image-detection-margin-ratio", type=float, default=0.005)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detector-stage", choices=("text", "icon"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    value = int(round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Write human-readable progress and an atomic machine-readable snapshot."""

    def __init__(
        self,
        *,
        stage: str,
        total: int,
        output_dir: Path,
        interval_seconds: float,
        initial_completed: int = 0,
        unit: str = "images",
    ) -> None:
        self.stage = stage
        self.total = max(0, int(total))
        self.output_dir = output_dir
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.initial_completed = max(0, int(initial_completed))
        self.unit = unit
        self.started = time.monotonic()
        self.last_print = 0.0

    def update(
        self,
        completed: int,
        *,
        status: str = "running",
        detail: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        completed = max(0, min(int(completed), self.total)) if self.total else max(0, int(completed))
        elapsed = max(0.0, time.monotonic() - self.started)
        newly_completed = max(0, completed - self.initial_completed)
        rate = newly_completed / elapsed if elapsed > 0 and newly_completed else 0.0
        remaining = max(0, self.total - completed)
        eta_seconds = remaining / rate if rate > 0 else None
        percent = completed / self.total if self.total else (1.0 if status == "completed" else 0.0)
        stage_index = PIPELINE_STAGES.index(self.stage) + 1 if self.stage in PIPELINE_STAGES else None
        payload = {
            "stage": self.stage,
            "stage_index": stage_index,
            "stage_total": len(PIPELINE_STAGES),
            "status": status,
            "detail": detail,
            "completed": completed,
            "total": self.total,
            "unit": self.unit,
            "percent": round(percent, 6),
            "elapsed_seconds": round(elapsed, 3),
            "rate_per_second": round(rate, 6),
            "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.output_dir / "run_status.json", payload)
        now = time.monotonic()
        if force or now - self.last_print >= self.interval_seconds:
            stage_label = (
                f"{self.stage} {stage_index}/{len(PIPELINE_STAGES)}"
                if stage_index is not None
                else self.stage
            )
            suffix = f" | {detail}" if detail else ""
            print(
                f"[进度 {stage_label}] {completed}/{self.total} {self.unit} "
                f"({percent:.1%}) | 已耗时 {format_duration(elapsed)} | "
                f"速度 {rate:.2f} {self.unit}/s | ETA {format_duration(eta_seconds)}{suffix}",
                flush=True,
            )
            self.last_print = now
        return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def digest_ids(ids: Iterable[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for image_id in sorted(ids):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_id(prefix: str, value: str, length: int = 20) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}_{digest[:length]}"


def normalized_1000_to_pixels(
    box: Sequence[float], width: int, height: int
) -> list[int]:
    values = [
        round(float(box[0]) * width / 1000),
        round(float(box[1]) * height / 1000),
        round(float(box[2]) * width / 1000),
        round(float(box[3]) * height / 1000),
    ]
    x1, x2 = sorted((max(0, min(width, values[0])), max(0, min(width, values[2]))))
    y1, y2 = sorted((max(0, min(height, values[1])), max(0, min(height, values[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"zero-area GT after conversion: {box} at {width}x{height}")
    return [x1, y1, x2, y2]


def parse_gt_boxes(answer: str, width: int, height: int) -> tuple[list[list[int]], list[list[float]]]:
    norm = [[float(value) for value in match] for match in BOX_PATTERN.findall(answer)]
    pixels = [normalized_1000_to_pixels(box, width, height) for box in norm]
    # Exact duplicates carry no additional localization information.
    dedup: dict[tuple[int, int, int, int], list[float]] = {}
    for pixel_box, norm_box in zip(pixels, norm):
        dedup.setdefault(tuple(pixel_box), norm_box)
    return [list(box) for box in dedup], [dedup[box] for box in dedup]


def open_raw_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def verify_revisions(project_root: Path, parser_root: Path) -> dict[str, str]:
    project_head = git_output(project_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"required baseline {BASELINE_COMMIT} is not an ancestor of {project_head}")
    parser_head = git_output(parser_root, "rev-parse", "HEAD")
    if parser_head != PARSER_COMMIT:
        raise RuntimeError(f"parser must be {PARSER_COMMIT}, found {parser_head}")
    return {"baseline": BASELINE_COMMIT, "project_head": project_head, "parser": parser_head}


def detector_config(args: argparse.Namespace) -> dict[str, Any]:
    text_model = args.text_model_dir.resolve(strict=False) if args.text_model_dir else None
    icon_model = args.icon_model or args.parser_root / "weights" / "icon_detect_v3" / "model.pt"
    return {
        "parser_commit": PARSER_COMMIT,
        "text": {
            "model_dir": str(text_model) if text_model else None,
            "auto_download": text_model is None,
            "model_name": "PP-OCRv5_server_det",
            "engine": "paddle_static",
            "long_side": args.text_long_side,
            "limit_type": "max",
            "pixel_threshold": 0.3,
            "box_threshold": args.text_box_threshold,
            "unclip_ratio": 1.5,
            "min_area": 0,
            "min_width": 0,
            "min_height": 0,
        },
        "icon": {
            "model": str(icon_model.resolve(strict=False)),
            "long_side": args.icon_long_side,
            "confidence": args.icon_confidence,
            "iou_threshold": 0.7,
            "max_detections": 1000,
            "min_area": 0,
            "min_width": 0,
            "min_height": 0,
        },
    }


def ensure_detector_config(path: Path, config: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                f"detector configuration is immutable once written: {path}; use a new output directory"
            )
    else:
        atomic_write_json(path, config)


def training_files(locany_data_dir: Path) -> list[Path]:
    files = [locany_data_dir / f"{task}_train.jsonl" for task in TASK_NAMES]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing training JSONL: " + ", ".join(map(str, missing)))
    return files


def source_files(source_dir: Path) -> list[Path]:
    files = [source_dir / task["file"] for task in TASKS]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source JSONL: " + ", ".join(map(str, missing)))
    return files


def print_preflight(
    args: argparse.Namespace,
    *,
    unique_count: int | None = None,
    readable_count: int | None = None,
    detector_stage: str | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    revisions = verify_revisions(project_root, args.parser_root.resolve(strict=True))
    sources = source_files(args.source_dir)
    trains = training_files(args.locany_data_dir)
    config = detector_config(args)
    if (
        detector_stage in {"text", "all"}
        and config["text"]["model_dir"] is not None
        and not Path(config["text"]["model_dir"]).is_dir()
    ):
        raise FileNotFoundError(f"missing PP-OCRv5 model directory: {config['text']['model_dir']}")
    if detector_stage in {"icon", "all"} and not Path(config["icon"]["model"]).is_file():
        raise FileNotFoundError(f"missing icon detector weights: {config['icon']['model']}")
    lines = [
        "=== UI5 crop audit preflight ===",
        f"baseline_commit : {revisions['baseline'][:12]} (ancestor of {revisions['project_head'][:12]})",
        "CPT             : disabled (no CPT data or training entrypoint)",
        "input_files     : " + ", ".join(str(path) for path in (*sources, *trains)),
        f"images_readable : {readable_count if readable_count is not None else 'from prepared manifest'}",
        f"unique_images   : {unique_count if unique_count is not None else 'from prepared manifest'}",
        f"parser_commit   : {revisions['parser']}",
        f"text_model      : {config['text']['model_dir'] or 'PaddleOCR automatic download/cache'}",
        f"icon_model      : {config['icon']['model']}",
        f"output_dir      : {args.output_dir.resolve(strict=False)}",
        f"GPUs            : {args.gpus}",
        f"workers/GPU     : {args.workers_per_gpu}",
        f"parameters      : {json.dumps({'detector': config, 'crop': CONFIGS, 'boundary_margin_ratio': args.boundary_margin_ratio, 'max_crops': args.max_crops}, ensure_ascii=False)}",
    ]
    print("\n".join(lines), flush=True)


def build_task_aware_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 500 <= args.shard_size <= 1000:
        raise ValueError("--shard-size must be in [500, 1000]")
    paths = AuditPaths(args.output_dir)
    # Fail on missing JSONL/parser/icon weights before spending time hashing images.
    print_preflight(args, detector_stage="all")
    overlap_reporter = ProgressReporter(
        stage="prepare",
        total=1,
        output_dir=args.output_dir,
        interval_seconds=args.progress_interval_seconds,
        unit="substeps",
    )
    overlap_reporter.update(
        0,
        detail="子步骤 1/3：统计源数据与训练数据重叠",
        force=True,
    )
    overlap_phase_reporters: dict[str, ProgressReporter] = {}

    def overlap_progress(phase: str, completed: int, total: int) -> None:
        if phase not in overlap_phase_reporters:
            overlap_phase_reporters[phase] = ProgressReporter(
                stage="prepare",
                total=total,
                output_dir=args.output_dir,
                interval_seconds=args.progress_interval_seconds,
                unit="records",
            )
        overlap_phase_reporters[phase].update(
            completed,
            detail=f"子步骤 1/3：重叠统计 {phase}",
            force=completed in {0, total},
        )

    overlap = analyze_overlap(
        args.source_dir,
        args.locany_data_dir,
        paths.manifest / "overlap",
        progress_callback=overlap_progress,
    )
    overlap_reporter.update(
        1,
        status="completed",
        detail="子步骤 1/3：重叠统计完成",
        force=True,
    )
    fingerprints: dict[str, str] = {}
    aliases_by_content: dict[str, set[str]] = defaultdict(set)
    raw_samples: list[dict[str, Any]] = []
    readable = 0
    task_records: list[tuple[Mapping[str, str], Path, int, dict[str, Any]]] = []
    for task in TASKS:
        annotation = args.locany_data_dir / f"{task['name']}_train.jsonl"
        task_records.extend(
            (task, annotation, line_no, record)
            for line_no, record in enumerate(read_jsonl(annotation), 1)
        )
    manifest_reporter = ProgressReporter(
        stage="prepare",
        total=len(task_records),
        output_dir=args.output_dir,
        interval_seconds=args.progress_interval_seconds,
        unit="records",
    )
    manifest_reporter.update(
        0,
        detail="子步骤 2/3：解析图片、内容指纹与 task-aware GT",
        force=True,
    )
    for record_index, (task, annotation, line_no, record) in enumerate(task_records, 1):
            raw_image = str(record["image"])
            image_path = resolve_training_image(raw_image, args.source_dir, args.locany_data_dir)
            canonical = str(image_path)
            if canonical not in fingerprints:
                fingerprints[canonical] = content_fingerprint(image_path)
                readable += 1
            content_id = fingerprints[canonical]
            aliases_by_content[content_id].add(canonical)
            image_id = stable_id("img", content_id)
            with open_raw_image(image_path) as image:
                width, height = image.size
            gt_pixels, gt_1000 = parse_gt_boxes(assistant_answer(record), width, height)
            raw_samples.append(
                {
                    "task": task["name"],
                    "source_file": str(annotation.resolve()),
                    "line_no": line_no,
                    "image_path": raw_image,
                    "canonical_path": canonical,
                    "content_id": content_id,
                    "image_id": image_id,
                    "width": width,
                    "height": height,
                    "positive": bool(gt_pixels),
                    "gt_count": len(gt_pixels),
                    "gt_boxes": gt_pixels,
                    "gt_boxes_1000": gt_1000,
                    "split": "train",
                }
            )
            if (
                record_index % args.progress_every_images == 0
                or record_index == len(task_records)
            ):
                manifest_reporter.update(
                    record_index,
                    detail="子步骤 2/3：解析图片、内容指纹与 task-aware GT",
                )

    # Exactly one manifest sample per image x task.  Same-task duplicates retain
    # source provenance and their distinct GT is combined only within that task.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in raw_samples:
        grouped[(sample["image_id"], sample["task"])].append(sample)
    task_samples: list[dict[str, Any]] = []
    for (image_id, task), members in sorted(grouped.items()):
        statuses = {bool(member["positive"]) for member in members}
        gt_by_box: dict[tuple[int, int, int, int], list[float]] = {}
        for member in members:
            for pixel_box, norm_box in zip(member["gt_boxes"], member["gt_boxes_1000"]):
                gt_by_box.setdefault(tuple(pixel_box), norm_box)
        representative = min(members, key=lambda item: item["canonical_path"])
        sample = dict(representative)
        sample.update(
            {
                "sample_id": stable_id("sample", f"{image_id}\0{task}"),
                "canonical_path": min(
                    path for member in members for path in aliases_by_content[member["content_id"]]
                ),
                "positive": bool(gt_by_box),
                "gt_count": len(gt_by_box),
                "gt_boxes": [list(box) for box in gt_by_box],
                "gt_boxes_1000": [gt_by_box[box] for box in gt_by_box],
                "source_records": [
                    {"source_file": member["source_file"], "line_no": member["line_no"]}
                    for member in members
                ],
                "same_task_polarity_conflict": len(statuses) > 1,
            }
        )
        task_samples.append(sample)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in task_samples:
        by_image[sample["image_id"]].append(sample)
    unique_images = []
    for image_id, samples in sorted(by_image.items()):
        representative = min(samples, key=lambda item: item["canonical_path"])
        aliases = sorted(aliases_by_content[representative["content_id"]])
        dimensions = {(sample["width"], sample["height"]) for sample in samples}
        if len(dimensions) != 1:
            raise ValueError(f"byte-identical image has inconsistent decoded dimensions: {image_id}")
        unique_images.append(
            {
                "image_id": image_id,
                "content_id": representative["content_id"],
                "image_path": representative["canonical_path"],
                "canonical_paths": aliases,
                "basename": Path(representative["canonical_path"]).name,
                "width": representative["width"],
                "height": representative["height"],
                "tasks": sorted({sample["task"] for sample in samples}),
            }
        )

    if args.max_unique_images:
        if args.max_unique_images < 1:
            raise ValueError("--max-unique-images must be 0 or positive")
        keep_ids = {row["image_id"] for row in unique_images[: args.max_unique_images]}
        unique_images = unique_images[: args.max_unique_images]
        task_samples = [sample for sample in task_samples if sample["image_id"] in keep_ids]

    print_preflight(
        args,
        unique_count=len(unique_images),
        readable_count=readable,
        detector_stage="all",
    )
    atomic_write_jsonl(paths.unique_images, unique_images)
    atomic_write_jsonl(paths.task_samples, task_samples)
    paths.shards.mkdir(parents=True, exist_ok=True)
    for stale in paths.shards.glob("shard_*.jsonl"):
        stale.unlink()
    for start in range(0, len(unique_images), args.shard_size):
        shard_index = start // args.shard_size
        atomic_write_jsonl(
            paths.shards / f"shard_{shard_index:05d}.jsonl",
            unique_images[start : start + args.shard_size],
        )
    ensure_detector_config(paths.detector_config, detector_config(args))
    prepare_summary = {
        "unique_images": len(unique_images),
        "task_samples": len(task_samples),
        "readable_canonical_paths": readable,
        "shards": math.ceil(len(unique_images) / args.shard_size),
        "same_content_cross_train_val": overlap["same_content_cross_train_val"],
        "cpt_enabled": False,
    }
    atomic_write_json(paths.manifest / "prepare_summary.json", prepare_summary)
    manifest_reporter.update(
        len(task_records),
        status="completed",
        detail=f"子步骤 3/3：已生成 {len(unique_images)} 张唯一图片、{prepare_summary['shards']} 个 shard",
        force=True,
    )
    return unique_images, task_samples


def load_parser_module(parser_root: Path, module_name: str):
    module_path = parser_root / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    if module_name in sys.modules:
        return sys.modules[module_name]
    # ui_region_cropper imports ui_region_parser by its real module name.
    if module_name == "ui_region_cropper" and "ui_region_parser" not in sys.modules:
        load_parser_module(parser_root, "ui_region_parser")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def completed_shard_valid(input_path: Path, output_path: Path, done_path: Path, stage: str) -> bool:
    if not output_path.is_file() or not done_path.is_file():
        return False
    try:
        expected = read_jsonl(input_path)
        actual = read_jsonl(output_path)
        marker = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_ids = [row.get("image_id") for row in expected]
    actual_ids = [row.get("image_id") for row in actual]
    return (
        len(actual_ids) == len(expected_ids)
        and len(set(actual_ids)) == len(actual_ids)
        and set(actual_ids) == set(expected_ids)
        and marker.get("stage") == stage
        and marker.get("count") == len(expected_ids)
        and marker.get("image_id_digest") == digest_ids(expected_ids)
    )


def _loaded_images(rows: Sequence[Mapping[str, Any]], threads: int) -> Iterator[tuple[Mapping[str, Any], Image.Image]]:
    def load(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Image.Image]:
        image = open_raw_image(Path(str(row["image_path"])))
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError(
                f"image dimensions changed for {row['image_id']}: manifest "
                f"{row['width']}x{row['height']}, current {image.width}x{image.height}"
            )
        return row, image

    if threads <= 1:
        for row in rows:
            yield load(row)
        return
    with ThreadPoolExecutor(max_workers=threads) as executor:
        yield from executor.map(load, rows)


def normalize_detector_items(items: Iterable[Mapping[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    result = []
    for item in items:
        bbox = [int(round(float(value))) for value in item["bbox"]]
        bbox = [
            max(0, min(width, bbox[0])),
            max(0, min(height, bbox[1])),
            max(0, min(width, bbox[2])),
            max(0, min(height, bbox[3])),
        ]
        bbox[0], bbox[2] = sorted((bbox[0], bbox[2]))
        bbox[1], bbox[3] = sorted((bbox[1], bbox[3]))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        normalized = dict(item)
        normalized["bbox"] = bbox
        if normalized.get("score") is not None:
            normalized["score"] = float(normalized["score"])
        result.append(normalized)
    return result


def run_detector_worker(args: argparse.Namespace) -> None:
    if args.worker_index is None or args.worker_count is None or not args.detector_stage:
        raise ValueError("internal detector worker arguments are incomplete")
    stage = args.detector_stage
    paths = AuditPaths(args.output_dir)
    config = detector_config(args)
    ensure_detector_config(paths.detector_config, config)
    module = load_parser_module(args.parser_root, "ui_region_parser")
    if stage == "text":
        settings = config["text"]
        detector = module.PaddleTextDetector(
            model_name=settings["model_name"],
            model_dir=Path(settings["model_dir"]) if settings["model_dir"] else None,
            device="cuda:0",
            engine=settings["engine"],
            limit_side_len=settings["long_side"],
            limit_type=settings["limit_type"],
            pixel_threshold=settings["pixel_threshold"],
            box_threshold=settings["box_threshold"],
            unclip_ratio=settings["unclip_ratio"],
            enable_mkldnn=True,
            min_area=0,
            min_width=0,
            min_height=0,
        )
    else:
        settings = config["icon"]
        detector = module.OmniParserYOLOv9Detector(Path(settings["model"]), "cuda:0")

    shard_paths = sorted(paths.shards.glob("shard_*.jsonl"))
    assigned = [
        path for index, path in enumerate(shard_paths) if index % args.worker_count == args.worker_index
    ]
    output_dir = paths.stage_dir(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_progress_path = output_dir / "progress" / f"worker_{args.worker_index:02d}.json"
    assigned_total = sum(len(read_jsonl(shard)) for shard in assigned)
    processed_this_run = 0

    def write_worker_progress(
        *,
        status: str,
        current_shard: str | None = None,
        current_completed: int = 0,
        current_total: int = 0,
    ) -> None:
        atomic_write_json(
            worker_progress_path,
            {
                "stage": stage,
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "status": status,
                "assigned_images": assigned_total,
                "processed_images_this_run": processed_this_run,
                "current_shard": current_shard,
                "current_shard_completed": current_completed,
                "current_shard_total": current_total,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    write_worker_progress(status="starting")
    worker_final_status = "failed"
    try:
        for shard in assigned:
            output_path = output_dir / shard.name
            done_path = output_dir / (shard.stem + ".done.json")
            if args.resume and completed_shard_valid(shard, output_path, done_path, stage):
                print(f"[{stage}] resume skip {shard.name}", flush=True)
                continue
            rows = read_jsonl(shard)
            outputs = []
            write_worker_progress(
                status="running",
                current_shard=shard.name,
                current_completed=0,
                current_total=len(rows),
            )
            for image_index, (row, image) in enumerate(
                _loaded_images(rows, args.image_loader_threads), 1
            ):
                try:
                    started = time.perf_counter()
                    if stage == "text":
                        detections = detector.predict(image)
                    else:
                        detections = detector.predict(
                            image=image,
                            confidence=settings["confidence"],
                            iou_threshold=settings["iou_threshold"],
                            long_side=settings["long_side"],
                            max_detections=settings["max_detections"],
                            min_area=0,
                            min_width=0,
                            min_height=0,
                        )
                    normalized = normalize_detector_items(detections, image.width, image.height)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                    outputs.append(
                        {
                            "image_id": row["image_id"],
                            "image": row["image_path"],
                            "width": image.width,
                            "height": image.height,
                            f"{stage}_detections": normalized,
                            "inference_ms": elapsed_ms,
                        }
                    )
                    processed_this_run += 1
                    if (
                        image_index % args.progress_every_images == 0
                        or image_index == len(rows)
                    ):
                        write_worker_progress(
                            status="running",
                            current_shard=shard.name,
                            current_completed=image_index,
                            current_total=len(rows),
                        )
                finally:
                    image.close()
            atomic_write_jsonl(output_path, outputs)
            atomic_write_json(
                done_path,
                {
                    "stage": stage,
                    "count": len(outputs),
                    "image_id_digest": digest_ids(row["image_id"] for row in outputs),
                    "input_shard": str(shard),
                },
            )
            write_worker_progress(status="running")
            print(f"[{stage}] completed {shard.name}: {len(outputs)} images", flush=True)
        worker_final_status = "completed"
    finally:
        close_model = getattr(getattr(detector, "model", None), "close", None)
        if callable(close_model):
            close_model()
        write_worker_progress(status=worker_final_status)


def detection_worker_command(args: argparse.Namespace, stage: str, worker_index: int, worker_count: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage", "_worker",
        "--detector-stage", stage,
        "--worker-index", str(worker_index),
        "--worker-count", str(worker_count),
        "--source-dir", str(args.source_dir),
        "--locany-data-dir", str(args.locany_data_dir),
        "--parser-root", str(args.parser_root),
        "--output-dir", str(args.output_dir),
        "--gpus", args.gpus,
        "--workers-per-gpu", str(args.workers_per_gpu),
        "--image-loader-threads", str(args.image_loader_threads),
        "--shard-size", str(args.shard_size),
        "--max-unique-images", str(args.max_unique_images),
        "--progress-interval-seconds", str(args.progress_interval_seconds),
        "--progress-every-images", str(args.progress_every_images),
        "--text-long-side", str(args.text_long_side),
        "--text-box-threshold", str(args.text_box_threshold),
        "--icon-long-side", str(args.icon_long_side),
        "--icon-confidence", str(args.icon_confidence),
    ]
    if args.text_model_dir:
        command.extend(("--text-model-dir", str(args.text_model_dir)))
    if args.icon_model:
        command.extend(("--icon-model", str(args.icon_model)))
    if args.resume:
        command.append("--resume")
    return command


def run_detection_stage(args: argparse.Namespace, stage: str) -> None:
    paths = AuditPaths(args.output_dir)
    unique = read_jsonl(paths.unique_images)
    print_preflight(args, unique_count=len(unique), readable_count=len(unique), detector_stage=stage)
    if args.workers_per_gpu == 2 and not args.allow_two_processes_per_gpu:
        raise ValueError(
            "2 processes/GPU requires --allow-two-processes-per-gpu after the 2,000-image "
            "benchmark confirms GPU <40%, memory <12GB, and higher throughput"
        )
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one GPU")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise FileNotFoundError("nvidia-smi is required to validate --gpus before detection")
    query = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    available = {line.strip() for line in query.stdout.splitlines() if line.strip()}
    missing_gpus = [gpu for gpu in gpus if gpu not in available]
    if missing_gpus:
        raise ValueError(f"requested GPUs are unavailable: {missing_gpus}; available={sorted(available)}")
    slots = [(gpu, slot) for gpu in gpus for slot in range(args.workers_per_gpu)]
    processes = []
    wall_started = time.perf_counter()
    baseline_completed = 0
    for shard in sorted(paths.shards.glob("shard_*.jsonl")):
        output_path = paths.stage_dir(stage) / shard.name
        done_path = paths.stage_dir(stage) / (shard.stem + ".done.json")
        if args.resume and completed_shard_valid(shard, output_path, done_path, stage):
            baseline_completed += len(read_jsonl(shard))
    reporter = ProgressReporter(
        stage=stage,
        total=len(unique),
        output_dir=args.output_dir,
        interval_seconds=args.progress_interval_seconds,
        initial_completed=baseline_completed,
    )
    reporter.update(
        baseline_completed,
        detail=f"启动 {len(slots)} 个常驻 GPU worker",
        force=True,
    )
    for worker_index, (gpu, _slot) in enumerate(slots):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = detection_worker_command(args, stage, worker_index, len(slots))
        processes.append((worker_index, gpu, subprocess.Popen(command, env=env)))
    def observed_completed() -> int:
        done_shards: set[str] = set()
        completed = 0
        for marker_path in paths.stage_dir(stage).glob("shard_*.done.json"):
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("stage") == stage:
                    completed += int(marker.get("count", 0))
                    done_shards.add(marker_path.name.replace(".done.json", ".jsonl"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        for progress_path in (paths.stage_dir(stage) / "progress").glob("worker_*.json"):
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            current_shard = progress.get("current_shard")
            if current_shard and current_shard not in done_shards:
                completed += int(progress.get("current_shard_completed", 0))
        return min(len(unique), completed)

    while any(process.poll() is None for _, _, process in processes):
        reporter.update(observed_completed())
        time.sleep(min(1.0, args.progress_interval_seconds))
    failures = [
        (worker_index, gpu, int(process.returncode))
        for worker_index, gpu, process in processes
        if process.returncode
    ]
    if failures:
        reporter.update(
            observed_completed(),
            status="failed",
            detail=f"GPU worker 失败：{failures}",
            force=True,
        )
        raise RuntimeError(f"{stage} detector workers failed: {failures}")
    stage_rows = [
        row
        for path in sorted(paths.stage_dir(stage).glob("shard_*.jsonl"))
        for row in read_jsonl(path)
    ]
    total_inference_ms = sum(float(row.get("inference_ms", 0.0)) for row in stage_rows)
    wall_seconds = time.perf_counter() - wall_started
    atomic_write_json(
        paths.stage_dir(stage) / "stage_summary.json",
        {
            "stage": stage,
            "images": len(stage_rows),
            "workers": len(slots),
            "workers_per_gpu": args.workers_per_gpu,
            "wall_seconds": round(wall_seconds, 3),
            "throughput_images_per_second": round(len(stage_rows) / wall_seconds, 6) if wall_seconds else 0.0,
            "sum_inference_ms": round(total_inference_ms, 3),
            "mean_inference_ms": round(total_inference_ms / len(stage_rows), 3) if stage_rows else 0.0,
        },
    )
    reporter.update(
        len(stage_rows),
        status="completed",
        detail=f"{len(slots)} 个 worker 已退出，检测结果已落盘",
        force=True,
    )


def merge_detections(args: argparse.Namespace) -> list[dict[str, Any]]:
    paths = AuditPaths(args.output_dir)
    if not paths.detector_config.is_file():
        raise FileNotFoundError(f"missing immutable detector config: {paths.detector_config}")
    ensure_detector_config(paths.detector_config, detector_config(args))
    expected_rows = read_jsonl(paths.unique_images)
    expected = {row["image_id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("unique_images.jsonl contains duplicate image_id")
    reporter = ProgressReporter(
        stage="merge",
        total=len(expected),
        output_dir=args.output_dir,
        interval_seconds=args.progress_interval_seconds,
    )
    reporter.update(0, detail="验证 text/icon shard 完整性", force=True)
    by_stage: dict[str, dict[str, dict[str, Any]]] = {}
    for stage in ("text", "icon"):
        invalid_shards = []
        for shard in sorted(paths.shards.glob("shard_*.jsonl")):
            output_path = paths.stage_dir(stage) / shard.name
            done_path = paths.stage_dir(stage) / (shard.stem + ".done.json")
            if not completed_shard_valid(shard, output_path, done_path, stage):
                invalid_shards.append(shard.name)
        if invalid_shards:
            raise ValueError(
                f"{stage} has incomplete or invalid shards: {invalid_shards[:20]}"
            )
        rows = [row for path in sorted(paths.stage_dir(stage).glob("shard_*.jsonl")) for row in read_jsonl(path)]
        indexed: dict[str, dict[str, Any]] = {}
        duplicates = []
        for row in rows:
            image_id = row["image_id"]
            if image_id in indexed:
                duplicates.append(image_id)
            indexed[image_id] = row
        if duplicates:
            raise ValueError(f"duplicate {stage} detections: {sorted(set(duplicates))[:20]}")
        missing = sorted(set(expected) - set(indexed))
        extra = sorted(set(indexed) - set(expected))
        if len(rows) != len(expected) or missing or extra:
            raise ValueError(
                f"{stage} count mismatch: unique={len(expected)}, rows={len(rows)}, "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )
        by_stage[stage] = indexed
    merged = []
    for merge_index, image_id in enumerate(sorted(expected), 1):
        manifest = expected[image_id]
        text_row = by_stage["text"][image_id]
        icon_row = by_stage["icon"][image_id]
        dimensions = {
            (int(manifest["width"]), int(manifest["height"])),
            (int(text_row["width"]), int(text_row["height"])),
            (int(icon_row["width"]), int(icon_row["height"])),
        }
        if len(dimensions) != 1:
            raise ValueError(f"dimension mismatch for {image_id}: {dimensions}")
        merged.append(
            {
                "image_id": image_id,
                "content_id": manifest["content_id"],
                "image": manifest["image_path"],
                "width": manifest["width"],
                "height": manifest["height"],
                "text_detections": text_row["text_detections"],
                "icon_detections": icon_row["icon_detections"],
            }
        )
        if merge_index % args.progress_every_images == 0 or merge_index == len(expected):
            reporter.update(merge_index, detail="按 image_id 合并 text/icon 检测")
    if len(merged) != len(expected):
        raise AssertionError("merged count changed unexpectedly")
    atomic_write_jsonl(paths.merged, merged)
    atomic_write_json(
        paths.merged.parent / "merge_summary.json",
        {
            "unique_images": len(expected),
            "text_rows": len(by_stage["text"]),
            "icon_rows": len(by_stage["icon"]),
            "merged_rows": len(merged),
            "image_id_digest": digest_ids(expected),
        },
    )
    reporter.update(
        len(merged),
        status="completed",
        detail="数量、重复、缺失和尺寸检查全部通过",
        force=True,
    )
    return merged


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def rect_contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def rect_intersects(left: Sequence[int], right: Sequence[int]) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])


def proposal_crops(
    cropper: Any,
    detection_record: Mapping[str, Any],
    config: Mapping[str, float],
    *,
    max_crops: int,
    boundary_margin_ratio: float,
) -> dict[str, Any]:
    width, height = int(detection_record["width"]), int(detection_record["height"])
    detections = []
    for source in ("text", "icon"):
        for source_index, item in enumerate(detection_record[f"{source}_detections"]):
            detections.append(
                cropper.DetectionBox(
                    index=len(detections),
                    source=source,
                    source_index=source_index,
                    bbox=tuple(int(value) for value in item["bbox"]),
                    score=float(item["score"]) if item.get("score") is not None else None,
                )
            )
    boxes = [detection.bbox for detection in detections]
    components, edge_count = cropper.build_connected_components(
        detections,
        width,
        height,
        config["horizontal_link_ratio"],
        config["vertical_link_ratio"],
    )
    initial_count = len(components)
    empty_fallback = False
    if components:
        groups, merge_history = cropper.merge_groups_to_limit(components, detections, max_crops)
        groups, overlap_history = cropper.merge_overlapping_group_envelopes(groups)
        crops, context_adjustments = cropper.make_non_overlapping_context_crops(
            groups, width, height, config["context_ratio"]
        )
    else:
        groups, merge_history, overlap_history, context_adjustments = [], [], [], []
        crops = [(0, 0, width, height)]
        empty_fallback = True
    crop_boxes = [tuple(int(value) for value in crop) for crop in crops]
    boundary_merge_history = []
    if boxes:
        crop_boxes = [
            cropper.make_boundary_safe_crop(
                crop, boxes, width, height, boundary_margin_ratio
            )[0]
            for crop in crop_boxes
        ]
        # Boundary-safe expansion is lossless.  If two expanded crops overlap,
        # merge them instead of allowing overlap or trimming through a detector.
        while True:
            pair = next(
                (
                    (left, right)
                    for left in range(len(crop_boxes))
                    for right in range(left + 1, len(crop_boxes))
                    if rect_intersects(crop_boxes[left], crop_boxes[right])
                ),
                None,
            )
            if pair is None:
                break
            left, right = pair
            merged = cropper.union_bbox([crop_boxes[left], crop_boxes[right]])
            merged = cropper.make_boundary_safe_crop(
                merged, boxes, width, height, boundary_margin_ratio
            )[0]
            boundary_merge_history.append(
                {"left": list(crop_boxes[left]), "right": list(crop_boxes[right]), "merged": list(merged)}
            )
            crop_boxes.pop(right)
            crop_boxes.pop(left)
            crop_boxes.append(merged)
        crop_boxes.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    cut = [
        {"crop": list(crop), "detector_box": list(box)}
        for crop in crop_boxes
        for box in boxes
        if rect_intersects(crop, box) and not rect_contains(crop, box)
    ]
    if cut:
        raise AssertionError(f"crop boundary cuts {len(cut)} detector boxes")
    if not 1 <= len(crop_boxes) <= max_crops:
        raise AssertionError(f"invalid crop count: {len(crop_boxes)}")
    return {
        "crop_boxes": [list(crop) for crop in crop_boxes],
        "detection_boxes": [list(box) for box in boxes],
        "detection_count": len(boxes),
        "edge_count": edge_count,
        "component_count_before_merge": initial_count,
        "forced_merge": bool(merge_history or boundary_merge_history),
        "merge_history": merge_history,
        "boundary_merge_history": boundary_merge_history,
        "overlap_merge_history": overlap_history,
        "context_adjustments": context_adjustments,
        "empty_detection_fallback": empty_fallback,
        "detector_boundary_cut_count": len(cut),
    }


def normalize_gt_in_crop(gt: Sequence[int], crop: Sequence[int]) -> dict[str, Any]:
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"invalid crop: {crop}")
    local = [gt[0] - crop[0], gt[1] - crop[1], gt[2] - crop[0], gt[3] - crop[1]]
    norm1000 = [
        round(local[0] / crop_width * 1000),
        round(local[1] / crop_height * 1000),
        round(local[2] / crop_width * 1000),
        round(local[3] / crop_height * 1000),
    ]
    reconstructed_local = [
        round(norm1000[0] / 1000 * crop_width),
        round(norm1000[1] / 1000 * crop_height),
        round(norm1000[2] / 1000 * crop_width),
        round(norm1000[3] / 1000 * crop_height),
    ]
    reconstructed_original = [
        reconstructed_local[0] + crop[0],
        reconstructed_local[1] + crop[1],
        reconstructed_local[2] + crop[0],
        reconstructed_local[3] + crop[1],
    ]
    max_error = max(abs(left - right) for left, right in zip(gt, reconstructed_original))
    return {
        "original_bbox": list(gt),
        "local_bbox": local,
        "norm1000": norm1000,
        "roundtrip_original_bbox": reconstructed_original,
        "roundtrip_max_error_px": max_error,
    }


def build_answer(label: str, boxes: Sequence[Sequence[int]]) -> str:
    if not boxes:
        return "<box>none</box>"
    return f"<ref>{label}</ref>" + "".join(
        f"<box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>" for box in boxes
    )


def build_preview_rows(
    sample: Mapping[str, Any],
    crop_boxes: Sequence[Sequence[int]],
    crop_paths: Sequence[Path],
    *,
    config_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_boxes = sample["gt_boxes"]
    preview = []
    failures = []
    negative_kept = False
    for crop_index, (crop, crop_path) in enumerate(zip(crop_boxes, crop_paths), 1):
        contained = [index for index, gt in enumerate(gt_boxes) if rect_contains(crop, gt)]
        partial = [
            index
            for index, gt in enumerate(gt_boxes)
            if rect_intersects(crop, gt) and not rect_contains(crop, gt)
        ]
        transformed = [normalize_gt_in_crop(gt_boxes[index], crop) for index in contained]
        max_error = max((item["roundtrip_max_error_px"] for item in transformed), default=0)
        training_eligible = not partial and max_error <= 1
        if partial:
            failure_type = "partial_intersection"
        elif max_error > 1:
            failure_type = "roundtrip_error"
        else:
            failure_type = None
        if failure_type:
            failures.append(
                {
                    "sample_id": sample["sample_id"],
                    "image_id": sample["image_id"],
                    "task": sample["task"],
                    "crop_id": crop_index,
                    "crop_bbox": list(crop),
                    "failure_type": failure_type,
                    "partial_gt_indices": partial,
                    "roundtrip_max_error_px": max_error,
                }
            )
        if not contained and not partial:
            if negative_kept:
                continue
            negative_kept = True
        norm_boxes = [item["norm1000"] for item in transformed]
        label = TASK_LABELS[sample["task"]]
        preview.append(
            {
                "sample_id": sample["sample_id"],
                "image_id": sample["image_id"],
                "task": sample["task"],
                "config": config_name,
                "source_image": sample["canonical_path"],
                "image": str(crop_path),
                "crop_id": crop_index,
                "crop_bbox": list(crop),
                "positive": bool(contained) if training_eligible else None,
                "gt_count": len(contained),
                "contained_gt_indices": contained,
                "partial_gt_indices": partial,
                "training_eligible": training_eligible,
                "roundtrip_max_error_px": max_error,
                "coordinate_transforms": transformed,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Locate all the instances that match the following description: {label}.",
                    },
                    {"from": "gpt", "value": build_answer(label, norm_boxes)},
                ],
            }
        )
    return preview, failures


def rectangle_union_area(rectangles: Sequence[Sequence[int]]) -> int:
    if not rectangles:
        return 0
    xs = sorted({coordinate for box in rectangles for coordinate in (box[0], box[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (box[1], box[3]) for box in rectangles if box[0] < right and box[2] > left
        )
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered += end - start
                start, end = next_start, next_end
        covered += end - start
        total += (right - left) * covered
    return total


def save_raw_crops(image: Image.Image, crop_boxes: Sequence[Sequence[int]], directory: Path, prefix: str = "crop") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, crop in enumerate(crop_boxes, 1):
        path = directory / f"{prefix}_{index:02d}.png"
        image.crop(tuple(crop)).save(path)
        paths.append(path.resolve())
    return paths


def uses_task_whole_image_policy(task: str) -> bool:
    """Global-defect policy is selected from task identity, never a filename or GT."""
    return task == "ui_content_missing"


def save_overview(
    image: Image.Image,
    crop_boxes: Sequence[Sequence[int]],
    gt_boxes: Sequence[Sequence[int]],
    path: Path,
) -> None:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    line = max(2, round(min(image.size) / 400))
    for index, crop in enumerate(crop_boxes, 1):
        draw.rectangle(tuple(crop), outline=(0, 210, 255), width=line)
        draw.text((crop[0] + line, crop[1] + line), f"crop {index}", fill=(0, 110, 170))
    for index, gt in enumerate(gt_boxes, 1):
        contained = any(rect_contains(crop, gt) for crop in crop_boxes)
        color = (255, 205, 0) if contained else (255, 40, 40)
        draw.rectangle(tuple(gt), outline=color, width=line)
        draw.text((gt[0] + line, gt[1] + line), f"GT {index}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path)
    rendered.close()


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def aggregate_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gt_total = sum(row["gt_count"] for row in rows)
    gt_contained = sum(row["gt_contained_count"] for row in rows)
    positive = [row for row in rows if row["positive"]]
    crop_counts = [row["crop_count"] for row in rows]
    area_ratios = [row["union_area_ratio"] for row in rows]
    gains = [gain for row in rows for gain in row["gt_enlargement_gains"]]
    total_original = sum(row["original_area"] for row in rows)
    total_union = sum(row["union_crop_area"] for row in rows)
    distribution = Counter(crop_counts)
    uncovered = gt_total - gt_contained
    partial_only = sum(row["partial_only_gt_count"] for row in rows)
    empty_fallback = len({row["image_id"] for row in rows if row["empty_detection_fallback"]})
    forced_merge = len({row["image_id"] for row in rows if row["forced_merge"]})
    boundary_cuts = sum(row["detector_boundary_cut_count"] for row in rows)
    roundtrip_failures = sum(row["roundtrip_error_over_1_count"] for row in rows)
    near_full = sum(row["union_area_ratio"] > 0.8 for row in rows)
    return {
        "samples": len(rows),
        "positive_samples": len(positive),
        "negative_samples": len(rows) - len(positive),
        "gt_count": gt_total,
        "gt_contained_count": gt_contained,
        "gt_box_containment_recall": gt_contained / gt_total if gt_total else 1.0,
        "positive_sample_success_count": sum(row["all_gt_contained"] for row in positive),
        "positive_sample_success_rate": (
            sum(row["all_gt_contained"] for row in positive) / len(positive) if positive else 1.0
        ),
        "uncovered_gt_count": uncovered,
        "partial_only_gt_count": partial_only,
        "partial_only_gt_ratio": (
            partial_only / gt_total if gt_total else 0.0
        ),
        "crop_count": {
            "mean": statistics.fmean(crop_counts) if crop_counts else 0.0,
            "p50": percentile(crop_counts, 0.50),
            "p90": percentile(crop_counts, 0.90),
            "max": max(crop_counts, default=0),
            "distribution_1_to_10": {str(value): distribution.get(value, 0) for value in range(1, 11)},
        },
        "union_area_ratio": {
            "mean": statistics.fmean(area_ratios) if area_ratios else 0.0,
            "p50": percentile(area_ratios, 0.50),
            "p90": percentile(area_ratios, 0.90),
        },
        "pixel_reduction_ratio": 1 - total_union / total_original if total_original else 0.0,
        "near_full_image_count": near_full,
        "near_full_image_ratio": near_full / len(rows) if rows else 0.0,
        "gt_gain_over_1_25_ratio": sum(gain > 1.25 for gain in gains) / len(gains) if gains else 0.0,
        "gt_gain_over_1_5_ratio": sum(gain > 1.5 for gain in gains) / len(gains) if gains else 0.0,
        "gt_gain_over_2_0_ratio": sum(gain > 2.0 for gain in gains) / len(gains) if gains else 0.0,
        "empty_detection_fallback_images": empty_fallback,
        "forced_merge_images": forced_merge,
        "detector_boundary_cut_count": boundary_cuts,
        "roundtrip_error_over_1_count": roundtrip_failures,
        "exception_count": (
            uncovered + partial_only + near_full + empty_fallback + forced_merge
            + boundary_cuts + roundtrip_failures
        ),
    }


def make_image_detail(
    sample: Mapping[str, Any],
    proposal: Mapping[str, Any],
    crop_boxes: Sequence[Sequence[int]],
    crop_paths: Sequence[Path],
    overview: Path,
    config_name: str,
    roundtrip_errors: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_coverage = []
    gains = []
    failures = []
    width, height = sample["width"], sample["height"]
    for gt_index, gt in enumerate(sample["gt_boxes"]):
        contained_by = [index + 1 for index, crop in enumerate(crop_boxes) if rect_contains(crop, gt)]
        partial_by = [
            index + 1
            for index, crop in enumerate(crop_boxes)
            if rect_intersects(crop, gt) and not rect_contains(crop, gt)
        ]
        if contained_by:
            gains.append(
                max(
                    min(width / (crop_boxes[index - 1][2] - crop_boxes[index - 1][0]), height / (crop_boxes[index - 1][3] - crop_boxes[index - 1][1]))
                    for index in contained_by
                )
            )
        else:
            failure_type = "partial_intersection" if partial_by else "uncovered"
            failures.append(
                {
                    "config": config_name,
                    "sample_id": sample["sample_id"],
                    "image_id": sample["image_id"],
                    "task": sample["task"],
                    "gt_index": gt_index,
                    "gt_bbox": gt,
                    "intersecting_crop_ids": partial_by,
                    "intersecting_crop_bboxes": [crop_boxes[index - 1] for index in partial_by],
                    "failure_type": failure_type,
                    "visualization": str(overview.resolve()),
                }
            )
        gt_coverage.append({"contained_by": contained_by, "partial_by": partial_by})
    original_area = width * height
    union_area = rectangle_union_area(crop_boxes)
    detail = {
        "config": config_name,
        "sample_id": sample["sample_id"],
        "image_id": sample["image_id"],
        "task": sample["task"],
        "positive": sample["positive"],
        "gt_count": len(sample["gt_boxes"]),
        "gt_contained_count": sum(bool(item["contained_by"]) for item in gt_coverage),
        "partial_only_gt_count": sum(not item["contained_by"] and bool(item["partial_by"]) for item in gt_coverage),
        "all_gt_contained": all(bool(item["contained_by"]) for item in gt_coverage),
        "crop_count": len(crop_boxes),
        "original_area": original_area,
        "union_crop_area": union_area,
        "union_area_ratio": union_area / original_area if original_area else 0.0,
        "pixel_reduction_ratio": 1 - union_area / original_area if original_area else 0.0,
        "gt_enlargement_gains": gains,
        "empty_detection_fallback": proposal["empty_detection_fallback"],
        "forced_merge": proposal["forced_merge"],
        "component_count_before_merge": proposal["component_count_before_merge"],
        "detector_boundary_cut_count": proposal["detector_boundary_cut_count"],
        "roundtrip_error_over_1_count": sum(error > 1 for error in roundtrip_errors),
        "overview": str(overview.resolve()),
        "source_image": sample["canonical_path"],
        "crop_boxes": [list(crop) for crop in crop_boxes],
        "crop_paths": [str(path.resolve()) for path in crop_paths],
    }
    return detail, failures


def task_overlap_rows(overlap: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for dataset_key, dataset_label in (("source_data", "original_source"), ("actual_training_data", "actual_training")):
        dataset = overlap[dataset_key]
        for identity_key, identity_label in (("path_overlap", "path"), ("content_overlap", "content")):
            for metric in ("counts", "jaccard"):
                rows.append([dataset_label, identity_label, metric, "task", *TASK_NAMES])
                matrix = dataset[identity_key][metric]
                for task in TASK_NAMES:
                    rows.append([dataset_label, identity_label, metric, task, *[matrix[task][other] for other in TASK_NAMES]])
                rows.append([])
    return rows


def write_statistics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "config", "sample_id", "image_id", "task", "positive", "gt_count",
        "gt_contained_count", "partial_only_gt_count", "all_gt_contained", "crop_count",
        "original_area", "union_crop_area", "union_area_ratio", "pixel_reduction_ratio",
        "empty_detection_fallback", "forced_merge", "detector_boundary_cut_count",
        "roundtrip_error_over_1_count", "overview", "source_image",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_excel_report(
    path: Path,
    summary: Mapping[str, Any],
    overlap: Mapping[str, Any],
    details: Sequence[Mapping[str, Any]],
    gt_failures: Sequence[Mapping[str, Any]],
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    summary_sheet = workbook.create_sheet("summary")
    summary_headers = [
        "config", "scope", "samples", "positive_samples", "negative_samples", "gt_count",
        "gt_contained_count", "gt_box_containment_recall", "positive_sample_success_rate",
        "uncovered_gt_count", "partial_only_gt_count", "partial_only_gt_ratio", "crop_mean",
        "crop_p50", "crop_p90", "crop_max", "union_area_mean", "union_area_p50",
        "union_area_p90", "pixel_reduction_ratio", "near_full_image_ratio",
        "gt_gain_over_1_25_ratio", "gt_gain_over_1_5_ratio", "gt_gain_over_2_0_ratio",
        "empty_detection_fallback_images", "forced_merge_images", "detector_boundary_cut_count",
        "roundtrip_error_over_1_count", "exception_count",
    ]
    summary_sheet.append(summary_headers)
    for config_name in CONFIGS:
        for scope in ("ALL", *TASK_NAMES):
            metric = summary["configs"][config_name]["by_scope"][scope]
            summary_sheet.append(
                [
                    config_name, scope, metric["samples"], metric["positive_samples"],
                    metric["negative_samples"], metric["gt_count"], metric["gt_contained_count"],
                    metric["gt_box_containment_recall"], metric["positive_sample_success_rate"],
                    metric["uncovered_gt_count"], metric["partial_only_gt_count"],
                    metric["partial_only_gt_ratio"], metric["crop_count"]["mean"],
                    metric["crop_count"]["p50"], metric["crop_count"]["p90"],
                    metric["crop_count"]["max"], metric["union_area_ratio"]["mean"],
                    metric["union_area_ratio"]["p50"], metric["union_area_ratio"]["p90"],
                    metric["pixel_reduction_ratio"], metric["near_full_image_ratio"],
                    metric["gt_gain_over_1_25_ratio"], metric["gt_gain_over_1_5_ratio"],
                    metric["gt_gain_over_2_0_ratio"], metric["empty_detection_fallback_images"],
                    metric["forced_merge_images"], metric["detector_boundary_cut_count"],
                    metric["roundtrip_error_over_1_count"], metric["exception_count"],
                ]
            )

    overlap_sheet = workbook.create_sheet("task_overlap")
    for row in task_overlap_rows(overlap):
        overlap_sheet.append(row)

    detail_sheet = workbook.create_sheet("image_detail")
    detail_headers = [
        "config", "sample_id", "image_id", "task", "positive", "gt_count",
        "gt_contained_count", "crop_count", "union_area_ratio", "pixel_reduction_ratio",
        "all_gt_contained", "empty_detection_fallback", "forced_merge", "overview", "source_image",
        *[f"crop_{index:02d}" for index in range(1, 11)],
    ]
    detail_sheet.append(detail_headers)
    for row in details:
        values = dict(row)
        for index, crop_path in enumerate(row.get("crop_paths", []), 1):
            values[f"crop_{index:02d}"] = crop_path
        detail_sheet.append([values.get(column) for column in detail_headers])

    failure_sheet = workbook.create_sheet("gt_failures")
    failure_headers = [
        "config", "sample_id", "image_id", "task", "gt_index", "gt_bbox",
        "intersecting_crop_ids", "intersecting_crop_bboxes", "failure_type", "visualization",
    ]
    failure_sheet.append(failure_headers)
    for row in gt_failures:
        failure_sheet.append(
            [
                row.get(column) if not isinstance(row.get(column), (list, dict)) else json.dumps(row.get(column), ensure_ascii=False)
                for column in failure_headers
            ]
        )

    compare_sheet = workbook.create_sheet("config_compare")
    compare_headers = [
        "config", "task", "link_ratio", "context_ratio", "gt_box_containment_recall",
        "positive_sample_success_rate", "union_area_mean", "pixel_reduction_ratio",
        "gt_gain_over_1_25_ratio", "gt_gain_over_1_5_ratio", "gt_gain_over_2_0_ratio",
        "uncovered_gt_count", "partial_only_gt_count", "detector_boundary_cut_count",
    ]
    compare_sheet.append(compare_headers)
    for config_name, config in CONFIGS.items():
        for scope in ("ALL", *TASK_NAMES):
            metric = summary["configs"][config_name]["by_scope"][scope]
            compare_sheet.append(
                [
                    config_name, scope, config["horizontal_link_ratio"], config["context_ratio"],
                    metric["gt_box_containment_recall"], metric["positive_sample_success_rate"],
                    metric["union_area_ratio"]["mean"], metric["pixel_reduction_ratio"],
                    metric["gt_gain_over_1_25_ratio"], metric["gt_gain_over_1_5_ratio"],
                    metric["gt_gain_over_2_0_ratio"], metric["uncovered_gt_count"],
                    metric["partial_only_gt_count"], metric["detector_boundary_cut_count"],
                ]
            )

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        if sheet.max_row >= 1 and sheet.max_column >= 1:
            sheet.auto_filter.ref = f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}"
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in sheet.columns:
            letter = get_column_letter(column_cells[0].column)
            width = min(60, max(10, max(len(str(cell.value or "")) for cell in column_cells) + 2))
            sheet.column_dimensions[letter].width = width

    percent_names = {
        "gt_box_containment_recall", "positive_sample_success_rate", "partial_only_gt_ratio",
        "union_area_mean", "union_area_p50", "union_area_p90", "pixel_reduction_ratio",
        "near_full_image_ratio", "gt_gain_over_1_25_ratio", "gt_gain_over_1_5_ratio",
        "gt_gain_over_2_0_ratio", "union_area_ratio",
    }
    for sheet in (summary_sheet, detail_sheet, compare_sheet):
        headers = {cell.value: cell.column for cell in sheet[1]}
        for name in percent_names & headers.keys():
            for row in range(2, sheet.max_row + 1):
                sheet.cell(row, headers[name]).number_format = "0.00%"
    hyperlink_columns = [
        (detail_sheet, "overview"),
        (detail_sheet, "source_image"),
        (failure_sheet, "visualization"),
        *[(detail_sheet, f"crop_{index:02d}") for index in range(1, 11)],
    ]
    for sheet, header_name in hyperlink_columns:
        headers = {cell.value: cell.column for cell in sheet[1]}
        column = headers[header_name]
        for row in range(2, sheet.max_row + 1):
            cell = sheet.cell(row, column)
            if cell.value:
                try:
                    cell.hyperlink = Path(str(cell.value)).resolve().as_uri()
                    cell.style = "Hyperlink"
                except ValueError:
                    pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    workbook.save(temporary)
    os.replace(temporary, path)


def run_crop_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = AuditPaths(args.output_dir)
    detections = {row["image_id"]: row for row in read_jsonl(paths.merged)}
    unique = read_jsonl(paths.unique_images)
    samples = read_jsonl(paths.task_samples)
    if set(detections) != {row["image_id"] for row in unique}:
        raise ValueError("merged detections do not exactly match unique manifest")
    overlap = json.loads((paths.manifest / "overlap" / "source_overlap.json").read_text(encoding="utf-8"))
    cropper = load_parser_module(args.parser_root, "ui_region_cropper")
    all_details: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"configs": {}, "cpt_enabled": False}
    crop_reporter = ProgressReporter(
        stage="crop-audit",
        total=len(unique) * len(CONFIGS),
        output_dir=args.output_dir,
        interval_seconds=args.progress_interval_seconds,
        unit="image-configs",
    )
    crop_completed = 0
    crop_reporter.update(
        0,
        detail="开始纯 CPU A/B/C crop、GT 关联和 overview",
        force=True,
    )

    samples_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_image[sample["image_id"]].append(sample)

    for config_name, config in CONFIGS.items():
        config_root = paths.crop_audit / f"config_{config_name}"
        config_details = []
        config_failures = []
        preview_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        anomaly_categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image_index, manifest in enumerate(unique, 1):
            image_id = manifest["image_id"]
            detection = detections[image_id]
            proposal = proposal_crops(
                cropper,
                detection,
                config,
                max_crops=args.max_crops,
                boundary_margin_ratio=args.boundary_margin_ratio,
            )
            with open_raw_image(Path(manifest["image_path"])) as image:
                region_boxes = proposal["crop_boxes"]
                region_paths = save_raw_crops(
                    image, region_boxes, config_root / "crops" / image_id, prefix="crop"
                )
                whole_box = list(
                    cropper.make_lightly_trimmed_whole_image_crop(
                        proposal["detection_boxes"],
                        image.width,
                        image.height,
                        args.whole_image_trim_ratio,
                        args.whole_image_detection_margin_ratio,
                    )
                )
                if any(
                    rect_intersects(whole_box, box) and not rect_contains(whole_box, box)
                    for box in proposal["detection_boxes"]
                ):
                    raise AssertionError("task-aware whole-image crop cuts a detector box")
                whole_paths: list[Path] | None = None
                for sample in samples_by_image[image_id]:
                    if uses_task_whole_image_policy(sample["task"]):
                        if whole_paths is None:
                            whole_paths = save_raw_crops(
                                image, [whole_box], config_root / "crops" / image_id, prefix="whole"
                            )
                        crop_boxes, crop_paths = [whole_box], whole_paths
                    else:
                        crop_boxes, crop_paths = region_boxes, region_paths
                    overview = config_root / "overviews" / sample["task"] / f"{image_id}.png"
                    save_overview(image, crop_boxes, sample["gt_boxes"], overview)
                    preview, preview_failures = build_preview_rows(
                        sample, crop_boxes, crop_paths, config_name=config_name
                    )
                    preview_by_task[sample["task"]].extend(preview)
                    errors = [row["roundtrip_max_error_px"] for row in preview]
                    detail, failures = make_image_detail(
                        sample, proposal, crop_boxes, crop_paths, overview, config_name, errors
                    )
                    config_details.append(detail)
                    config_failures.extend(failures)
                    if failures:
                        for failure in failures:
                            category = (
                                "gt_partial_only"
                                if failure["failure_type"] == "partial_intersection"
                                else "gt_uncovered"
                            )
                            anomaly_categories[category].append(failure)
                    if proposal["empty_detection_fallback"]:
                        anomaly_categories["detector_empty_fallback"].append(detail)
                    if detail["union_area_ratio"] > 0.8:
                        anomaly_categories["near_full_image"].append(detail)
                    if detail["crop_count"] == args.max_crops or proposal["forced_merge"]:
                        anomaly_categories["max_crops_or_forced_merge"].append(detail)
                    if detail["roundtrip_error_over_1_count"]:
                        anomaly_categories["roundtrip_error"].append(detail)
                    for failure in preview_failures:
                        if failure["failure_type"] == "roundtrip_error":
                            failure["visualization"] = str(overview.resolve())
                            anomaly_categories["roundtrip_error"].append(failure)
            crop_completed += 1
            crop_reporter.update(
                crop_completed,
                detail=(
                    f"config {config_name}，当前 {image_index}/{len(unique)}，"
                    f"{proposal['detection_count']} boxes -> {len(proposal['crop_boxes'])} crops"
                ),
            )
        # Basename collisions and shared-image task annotations are explicit
        # risks in the fixed parser's legacy basename aggregation path.  Link
        # each risk row to the already-generated task overview.
        detail_by_image_task = {
            (row["image_id"], row["task"]): row for row in config_details
        }
        for conflict in overlap["actual_training_data"]["basename_conflicts"]["details"]:
            conflict_paths = set(conflict["canonical_paths"])
            for row in config_details:
                if row["source_image"] in conflict_paths:
                    anomaly_categories["basename_or_multitask_annotation_risk"].append(
                        {
                            "risk_type": "same_basename_different_identity",
                            "basename": conflict["basename"],
                            "image_id": row["image_id"],
                            "task": row["task"],
                            "source_image": row["source_image"],
                            "visualization": row["overview"],
                        }
                    )
        for image_id, image_samples in samples_by_image.items():
            if len({sample["task"] for sample in image_samples}) < 2:
                continue
            signatures = {
                sample["task"]: tuple(tuple(box) for box in sample["gt_boxes"])
                for sample in image_samples
            }
            if len(set(signatures.values())) <= 1:
                continue
            for sample in image_samples:
                detail = detail_by_image_task[(image_id, sample["task"])]
                anomaly_categories["basename_or_multitask_annotation_risk"].append(
                    {
                        "risk_type": "shared_image_distinct_task_gt",
                        "image_id": image_id,
                        "task": sample["task"],
                        "gt_signature": signatures[sample["task"]],
                        "visualization": detail["overview"],
                    }
                )
        priority = (
            "gt_uncovered",
            "gt_partial_only",
            "basename_or_multitask_annotation_risk",
            "detector_empty_fallback",
            "near_full_image",
            "max_crops_or_forced_merge",
            "roundtrip_error",
        )
        ordered_anomalies = {
            category: anomaly_categories.get(category, []) for category in priority
        }
        for task in TASK_NAMES:
            atomic_write_jsonl(config_root / "preview" / f"{task}.jsonl", preview_by_task[task])
        atomic_write_jsonl(config_root / "task_aware_manifest.jsonl", config_details)
        atomic_write_jsonl(config_root / "gt_failures.jsonl", config_failures)
        atomic_write_json(config_root / "anomalies.json", ordered_anomalies)
        by_scope = {"ALL": aggregate_scope(config_details)}
        for task in TASK_NAMES:
            by_scope[task] = aggregate_scope([row for row in config_details if row["task"] == task])
        summary["configs"][config_name] = {
            "parameters": config,
            "by_scope": by_scope,
            "gt_failure_count": len(config_failures),
            "anomaly_counts": {name: len(rows) for name, rows in ordered_anomalies.items()},
        }
        all_details.extend(config_details)
        all_failures.extend(config_failures)

    local_tasks = [task for task in TASK_NAMES if task != "ui_content_missing"]
    # Select on the four local tasks: full GT containment first, then positive
    # sample success, pixel reduction, and useful enlargement.  The task-aware
    # near-full content_missing policy must not dominate this choice.
    def selection_key(name: str) -> tuple[float, float, float, float, float]:
        scopes = summary["configs"][name]["by_scope"]
        gt_total = sum(scopes[task]["gt_count"] for task in local_tasks)
        contained = sum(scopes[task]["gt_contained_count"] for task in local_tasks)
        positives = sum(scopes[task]["positive_samples"] for task in local_tasks)
        successful = sum(scopes[task]["positive_sample_success_count"] for task in local_tasks)
        local_rows = [row for row in all_details if row["config"] == name and row["task"] in local_tasks]
        local_efficiency = aggregate_scope(local_rows)
        return (
            contained / gt_total if gt_total else 1.0,
            min(scopes[task]["gt_box_containment_recall"] for task in local_tasks),
            successful / positives if positives else 1.0,
            local_efficiency["pixel_reduction_ratio"],
            local_efficiency["gt_gain_over_1_5_ratio"],
        )

    summary["recommended_config"] = max(CONFIGS, key=selection_key)
    recommended = summary["configs"][summary["recommended_config"]]["by_scope"]
    local_gt = sum(recommended[task]["gt_count"] for task in local_tasks)
    local_contained = sum(recommended[task]["gt_contained_count"] for task in local_tasks)
    summary["next_stage_gate"] = {
        "local_task_overall_recall": local_contained / local_gt if local_gt else 1.0,
        "local_task_min_recall": min(recommended[task]["gt_box_containment_recall"] for task in local_tasks),
        "detector_boundary_cut_count": recommended["ALL"]["detector_boundary_cut_count"],
        "passes": (
            (local_contained / local_gt if local_gt else 1.0) >= 0.99
            and all(recommended[task]["gt_box_containment_recall"] >= 0.98 for task in local_tasks)
            and recommended["ALL"]["detector_boundary_cut_count"] == 0
        ),
        "training_started": False,
    }
    atomic_write_json(paths.crop_audit / "summary.json", summary)
    write_statistics_csv(paths.crop_audit / "statistics.csv", all_details)
    atomic_write_jsonl(paths.crop_audit / "task_aware_manifest.jsonl", all_details)
    atomic_write_jsonl(paths.crop_audit / "gt_failures.jsonl", all_failures)
    crop_reporter.update(
        crop_completed,
        detail="crop 已完成，正在写 summary、CSV 和 Excel",
        force=True,
    )
    write_excel_report(
        paths.crop_audit / "ui5_crop_audit.xlsx", summary, overlap, all_details, all_failures
    )
    crop_reporter.update(
        crop_completed,
        status="completed",
        detail="A/B/C 报告、preview、异常清单和 Excel 已完成",
        force=True,
    )
    return summary


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.source_dir = args.source_dir.expanduser().resolve(strict=True)
    args.locany_data_dir = args.locany_data_dir.expanduser().resolve(strict=True)
    args.parser_root = args.parser_root.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    if args.text_model_dir:
        args.text_model_dir = args.text_model_dir.expanduser().resolve(strict=False)
    if args.icon_model:
        args.icon_model = args.icon_model.expanduser().resolve(strict=False)
    if not 1 <= args.max_crops <= 10:
        raise ValueError("--max-crops must be in [1, 10]")
    if not 2 <= args.image_loader_threads <= 4:
        raise ValueError("--image-loader-threads must be in [2, 4]")
    if args.progress_interval_seconds <= 0:
        raise ValueError("--progress-interval-seconds must be positive")
    if args.progress_every_images <= 0:
        raise ValueError("--progress-every-images must be positive")
    return args


def run(args: argparse.Namespace) -> Any:
    args = normalize_args(args)
    if args.stage == "_worker":
        return run_detector_worker(args)
    stages = PIPELINE_STAGES if args.stage == "all" else (args.stage,)
    result = None
    for stage in stages:
        stage_index = PIPELINE_STAGES.index(stage) + 1
        print(
            f"\n[流水线] 开始阶段 {stage_index}/{len(PIPELINE_STAGES)}：{stage}",
            flush=True,
        )
        stage_started = time.monotonic()
        try:
            if stage == "prepare":
                result = build_task_aware_manifest(args)
            elif stage in {"text", "icon"}:
                result = run_detection_stage(args, stage)
            elif stage == "merge":
                print_preflight(args)
                result = merge_detections(args)
            elif stage == "crop-audit":
                print_preflight(args)
                result = run_crop_audit(args)
        except Exception as exc:
            status_path = args.output_dir / "run_status.json"
            try:
                payload = (
                    json.loads(status_path.read_text(encoding="utf-8"))
                    if status_path.is_file()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                payload = {}
            payload.update(
                {
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_total": len(PIPELINE_STAGES),
                    "status": "failed",
                    "error": str(exc),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            atomic_write_json(status_path, payload)
            print(
                f"[流水线] 阶段 {stage_index}/{len(PIPELINE_STAGES)} 失败：{stage}：{exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            f"[流水线] 完成阶段 {stage_index}/{len(PIPELINE_STAGES)}：{stage}，"
            f"本阶段耗时 {format_duration(time.monotonic() - stage_started)}",
            flush=True,
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run(args)
        if isinstance(result, Mapping):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
