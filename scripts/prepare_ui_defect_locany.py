#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert five MS-Swift UI-defect JSONL files to LocateAnything JSONL + recipe JSON.

Expected source record:
{
  "messages": [...],
  "images": ["/absolute/path/to/image.jpg"],
  "objects": {
    "ref": ["文字溢出容器"],
    "bbox": [[x1, y1, x2, y2], ...],
    "bbox_type": "real"
  }
}

LocateAnything output:
{
  "conversations": [
    {"from": "human", "value": "<image-1>\nLocate all ..."},
    {"from": "gpt", "value": "<ref>...</ref><box><x1><y1><x2><y2></box>..."}
  ],
  "image": "mnt/bn/.../image.jpg"
}

The recipe uses root="/", so image paths are stored without the leading slash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image, UnidentifiedImageError


# Defaults are repository-relative so every cluster path remains CLI-overridable.
# In particular, do not concatenate a hand-edited ``root_path`` string here: the
# previous H20 default omitted a trailing slash and produced
# ``...sicheng_workspacecode/Eagle/Embodied``.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = DEFAULT_PROJECT_ROOT / "data"

TASKS = [
    {
        "name": "ui_occlusion",
        "file": "train_ui_occlusion_wcnt.jsonl",
        "zh": "元素重叠",
        "en": "overlapping elements",
    },
    {
        "name": "ui_cropping",
        "file": "train_ui_cropping_wcnt.jsonl",
        "zh": "元素被裁切",
        "en": "cropped element",
    },
    {
        "name": "ui_text_overflow",
        "file": "train_ui_text_overflow_wcnt.jsonl",
        "zh": "文字溢出容器",
        "en": "text overflow",
    },
    {
        "name": "ui_text_ellipsis",
        "file": "train_ui_text_ellipsis_wcnt.jsonl",
        "zh": "文字省略异常",
        "en": "abnormal text ellipsis",
    },
    {
        "name": "ui_content_missing",
        "file": "train_ui_content_missing_wcnt.jsonl",
        "zh": "内容未展示",
        "en": "missing content",
    },
]


@dataclass
class TaskStats:
    total_lines: int = 0
    written_train: int = 0
    written_val: int = 0
    positives: int = 0
    negatives: int = 0
    total_boxes: int = 0
    skipped_negative_downsample: int = 0
    missing_images: int = 0
    invalid_json: int = 0
    invalid_records: int = 0
    invalid_boxes: int = 0
    count_mismatches: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert five UI-defect JSONL files to LocateAnything format."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/ui_defect_locany",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/recipe",
    )
    parser.add_argument(
        "--label-style",
        choices=("zh", "en", "bilingual"),
        default="en",
        help="Category text used in both prompt and <ref>.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Validation ratio split by image path, not by individual task sample.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--negative-keep-ratio",
        type=float,
        default=1.0,
        help="Keep ratio for no-box records. Default keeps all negatives.",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("auto", "pixel", "norm01", "norm1000"),
        default="auto",
        help="How source bbox coordinates are represented.",
    )
    parser.add_argument(
        "--bbox-coord-mode",
        choices=("xyxy", "xywh"),
        default="xyxy",
        help="Shape of each source bbox.",
    )
    parser.add_argument(
        "--prompt-language",
        choices=("en", "zh"),
        default="en",
        help="Instruction language. Labels are controlled separately.",
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=0,
        help="0 means all. Use a small value for a smoke test.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip missing/unreadable images instead of aborting.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on malformed JSON, records, or invalid boxes.",
    )
    return parser.parse_args()


def stable_fraction(text: str, seed: int) -> float:
    payload = f"{seed}\0{text}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64)


def choose_split(image_path: Path, val_ratio: float, seed: int) -> str:
    return "val" if stable_fraction(str(image_path), seed) < val_ratio else "train"


def format_label(task: dict[str, str], style: str) -> str:
    if style == "zh":
        return task["zh"]
    if style == "en":
        return task["en"]
    return f'{task["zh"]} ({task["en"]})'


# def build_prompt(label: str, language: str) -> str:
#     if language == "zh":
#         return f"<image-1>\n找出图中所有符合以下描述的区域：{label}。"
#     return (
#         "<image-1>\n"
#         f"Locate all the instances that match the following description: {label}."
#     )

def build_prompt(label: str) -> str:
    validate_label(label)
    return (
        f"Locate all the instances that match the following description: {label}."
    )


def extract_image_path(record: dict[str, Any], source_dir: Path) -> Path:
    images = record.get("images")
    image = record.get("image")
    if isinstance(images, list) and images:
        raw = images[0]
    elif isinstance(image, str) and image:
        raw = image
    else:
        raise ValueError("record has neither non-empty images nor image")

    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = source_dir / path
    return path.resolve(strict=False)


def iter_numeric_boxes(value: Any, coord_mode: str) -> Iterator[list[float]]:
    """Yield boxes from common nested/list/dict/string representations."""
    if value is None:
        return

    if isinstance(value, dict):
        key_sets = [
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("xmin", "ymin", "xmax", "ymax"),
        ]
        for keys in key_sets:
            if all(k in value for k in keys):
                yield [float(value[k]) for k in keys]
                return
        if all(k in value for k in ("x", "y", "w", "h")):
            x, y, w, h = (float(value[k]) for k in ("x", "y", "w", "h"))
            yield [x, y, x + w, y + h]
            return
        for nested_key in ("bbox", "box", "boxes", "value"):
            if nested_key in value:
                yield from iter_numeric_boxes(value[nested_key], coord_mode)
                return
        return

    if isinstance(value, str):
        numbers = [
            float(x)
            for x in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        ]
        for i in range(0, len(numbers) - 3, 4):
            chunk = numbers[i : i + 4]
            if coord_mode == "xywh":
                x, y, w, h = chunk
                chunk = [x, y, x + w, y + h]
            yield chunk
        return

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if (
            len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        ):
            box = [float(x) for x in value]
            if coord_mode == "xywh":
                x, y, w, h = box
                box = [x, y, x + w, y + h]
            yield box
            return
        for item in value:
            yield from iter_numeric_boxes(item, coord_mode)


def infer_bbox_format(
    box: Sequence[float], bbox_type: str, requested: str
) -> str:
    if requested != "auto":
        return requested

    bbox_type_lower = bbox_type.lower()
    if any(k in bbox_type_lower for k in ("real", "pixel", "absolute")):
        return "pixel"
    if "norm" in bbox_type_lower:
        return "norm01" if max(abs(v) for v in box) <= 1.5 else "norm1000"

    max_abs = max(abs(v) for v in box)
    if max_abs <= 1.5:
        return "norm01"
    # The supplied UI data declares bbox_type="real"; for unknown types,
    # pixel coordinates are the safest default.
    return "pixel"


def normalize_box(
    box: Sequence[float],
    width: int,
    height: int,
    bbox_type: str,
    requested_format: str,
) -> tuple[int, int, int, int] | None:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    if not all(math.isfinite(v) for v in box):
        return None

    x1, y1, x2, y2 = map(float, box)
    source_format = infer_bbox_format(box, bbox_type, requested_format)

    if source_format == "pixel":
        x1, x2 = x1 / width * 1000.0, x2 / width * 1000.0
        y1, y2 = y1 / height * 1000.0, y2 / height * 1000.0
    elif source_format == "norm01":
        x1, y1, x2, y2 = [v * 1000.0 for v in (x1, y1, x2, y2)]
    elif source_format == "norm1000":
        pass
    else:
        raise ValueError(f"unsupported bbox format: {source_format}")

    # Repair reversed corners and clip to the LocateAnything coordinate range.
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1, x2, y2 = [
        min(1000, max(0, int(round(v)))) for v in (x1, y1, x2, y2)
    ]

    # A zero-area box carries no localization signal.
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def validate_label(label: str) -> None:
    if not label or any(token in label for token in ("<", ">", "</c>")):
        raise ValueError(f"invalid LocateAnything label: {label!r}")


def build_answer(label: str, boxes: Sequence[tuple[int, int, int, int]]) -> str:
    validate_label(label)
    if not boxes:
        return "<box>none</box>"
    box_text = "".join(
        f"<box><{x1}><{y1}><{x2}><{y2}></box>" for x1, y1, x2, y2 in boxes
    )
    # Official multi-instance phrase-grounding format uses one <ref>
    # followed by all matching boxes.
    return f"<ref>{label}</ref>{box_text}"


def get_declared_count(record: dict[str, Any]) -> int | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            match = re.search(r"(\d+)\s*个问题", str(message.get("content", "")))
            return int(match.group(1)) if match else None
    return None


def write_jsonl_line(handle, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not 0 < args.negative_keep_ratio <= 1:
        raise ValueError("--negative-keep-ratio must be in (0, 1]")

    project_root = args.project_root.resolve(strict=False)
    source_dir = args.source_dir.resolve(strict=False)
    output_dir = (
        args.output_dir.resolve(strict=False)
        if args.output_dir
        else project_root / "data" / "ui_defect_locany"
    )
    recipe_dir = (
        args.recipe_dir.resolve(strict=False)
        if args.recipe_dir
        else project_root / "recipe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, TaskStats] = {task["name"]: TaskStats() for task in TASKS}
    train_annotations: list[str] = []
    val_annotations: list[str] = []
    first_positive: dict[str, Any] | None = None
    first_negative: dict[str, Any] | None = None

    for task in TASKS:
        task_name = task["name"]
        input_path = source_dir / task["file"]
        if not input_path.is_file():
            raise FileNotFoundError(f"missing source JSONL: {input_path}")

        label = format_label(task, args.label_style)
        train_path = output_dir / f"{task_name}_train.jsonl"
        val_path = output_dir / f"{task_name}_val.jsonl"
        train_annotations.append(str(train_path))
        val_annotations.append(str(val_path))

        task_stats = stats[task_name]
        with (
            input_path.open("r", encoding="utf-8") as src,
            train_path.open("w", encoding="utf-8") as train_out,
            val_path.open("w", encoding="utf-8") as val_out,
        ):
            for line_idx, line in enumerate(src, start=1):
                if args.max_samples_per_file and line_idx > args.max_samples_per_file:
                    break
                task_stats.total_lines += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    task_stats.invalid_json += 1
                    message = f"{input_path}:{line_idx}: invalid JSON: {exc}"
                    if args.strict:
                        raise ValueError(message) from exc
                    print(f"[WARN] {message}", file=sys.stderr)
                    continue

                try:
                    image_path = extract_image_path(record, source_dir)
                    if not image_path.is_file():
                        raise FileNotFoundError(str(image_path))
                    with Image.open(image_path) as image:
                        width, height = image.size

                    objects = record.get("objects") or {}
                    raw_bbox = objects.get("bbox", [])
                    bbox_type = str(objects.get("bbox_type", ""))
                    raw_boxes = list(
                        iter_numeric_boxes(raw_bbox, args.bbox_coord_mode)
                    )
                    boxes: list[tuple[int, int, int, int]] = []
                    for raw_box in raw_boxes:
                        normalized = normalize_box(
                            raw_box,
                            width=width,
                            height=height,
                            bbox_type=bbox_type,
                            requested_format=args.bbox_format,
                        )
                        if normalized is None:
                            task_stats.invalid_boxes += 1
                            if args.strict:
                                raise ValueError(
                                    f"invalid/zero-area bbox {raw_box} for image {image_path}"
                                )
                            continue
                        boxes.append(normalized)

                    # Remove exact duplicates while preserving source order.
                    # boxes = list(dict.fromkeys(boxes))
                    # Remove duplicates, then use LocateAnything's default X-Y corner order:
                    # first sort by top-left x, then by top-left y.
                    boxes = sorted(
                        dict.fromkeys(boxes),
                        key=lambda box: (box[0], box[1], box[2], box[3]),
                    )

                    declared_count = get_declared_count(record)
                    if declared_count is not None and declared_count != len(boxes):
                        task_stats.count_mismatches += 1

                    if not boxes and args.negative_keep_ratio < 1:
                        negative_key = f"{task_name}\0{image_path}"
                        if (
                            stable_fraction(negative_key, args.seed + 1)
                            >= args.negative_keep_ratio
                        ):
                            task_stats.skipped_negative_downsample += 1
                            continue

                    sample = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": build_prompt(label),
                            },
                            {
                                "from": "gpt",
                                "value": build_answer(label, boxes),
                            },
                        ],
                        # Keep the annotation path relative to recipe root="/".
                        "image": image_path.as_posix().lstrip("/"),
                    }

                    split = choose_split(image_path, args.val_ratio, args.seed)
                    if split == "val":
                        write_jsonl_line(val_out, sample)
                        task_stats.written_val += 1
                    else:
                        write_jsonl_line(train_out, sample)
                        task_stats.written_train += 1

                    if boxes:
                        task_stats.positives += 1
                        task_stats.total_boxes += len(boxes)
                        if first_positive is None:
                            first_positive = sample
                    else:
                        task_stats.negatives += 1
                        if first_negative is None:
                            first_negative = sample

                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    if isinstance(exc, FileNotFoundError):
                        task_stats.missing_images += 1
                        if not args.skip_missing_images:
                            raise
                    else:
                        task_stats.invalid_records += 1
                        if args.strict:
                            raise
                    print(
                        f"[WARN] {input_path}:{line_idx}: skipped record: {exc}",
                        file=sys.stderr,
                    )
                    continue

        print(
            f"[OK] {task_name}: "
            f"train={task_stats.written_train}, val={task_stats.written_val}, "
            f"positive={task_stats.positives}, negative={task_stats.negatives}, "
            f"boxes={task_stats.total_boxes}"
        )

    train_recipe = {
        "ui_defect_5class_train": {
            "annotation": train_annotations,
            "root": "/",
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    val_recipe = {
        "ui_defect_5class_val": {
            "annotation": val_annotations,
            "root": "/",
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }

    train_recipe_path = recipe_dir / "ui_defect_5class_train.json"
    val_recipe_path = recipe_dir / "ui_defect_5class_val.json"
    train_recipe_path.write_text(
        json.dumps(train_recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    val_recipe_path.write_text(
        json.dumps(val_recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "project_root": str(project_root),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "train_recipe": str(train_recipe_path),
        "val_recipe": str(val_recipe_path),
        "label_style": args.label_style,
        "prompt_language": args.prompt_language,
        "val_ratio": args.val_ratio,
        "negative_keep_ratio": args.negative_keep_ratio,
        "bbox_format": args.bbox_format,
        "bbox_coord_mode": args.bbox_coord_mode,
        "stats": {name: asdict(value) for name, value in stats.items()},
        "first_positive": first_positive,
        "first_negative": first_negative,
    }
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_train = sum(s.written_train for s in stats.values())
    total_val = sum(s.written_val for s in stats.values())
    total_boxes = sum(s.total_boxes for s in stats.values())
    print("=" * 80)
    print(f"train samples : {total_train}")
    print(f"val samples   : {total_val}")
    print(f"total boxes   : {total_boxes}")
    print(f"train recipe  : {train_recipe_path}")
    print(f"val recipe    : {val_recipe_path}")
    print(f"summary       : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
