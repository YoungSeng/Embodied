#!/usr/bin/env python3
"""Convert the documented UI-Lens single-interface JSONL schema to UI5 eval.

No training data, prompts, crops, or new negative examples are generated.
Each source task retains its own annotated evaluation population. This adapter
is intentionally strict: unknown labels/layouts require inspecting the source,
not guessing a ground truth. Requires Pillow, but no GPU or model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


TASKS = {
    "container_overlap": ("occlusion", "元素重叠"),
    "cropped_content": ("cropping", "元素被裁切"),
    "text_overflow": ("text_overflow", "文字溢出容器"),
    "abnormal_text_ellipsis": ("text_ellipsis", "文字省略异常"),
    "undisplayed_content": ("content_missing", "内容未展示"),
}
ALIASES = {
    "occlusion": ["container overlap", "元素重叠", "元素遮挡", "容器重叠"],
    "cropping": ["cropped content", "元素被裁切", "元素裁切", "内容裁切"],
    "text_overflow": ["text overflow", "文字溢出容器", "文字溢出", "文本溢出"],
    "text_ellipsis": ["abnormal text ellipsis", "文字省略异常", "文本省略", "文本省略异常"],
    "content_missing": ["undisplayed content", "内容未展示", "内容缺失"],
}


def norm(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold()


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("infos"), dict):
                        raise ValueError("expected an object with an infos object")
                    yield number, row
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def annotation_dir(root: Path) -> Path:
    choices = [root / name for name in ("label_for_single_UIs_cn", "label_for_single_UIs")]
    found = [path for path in choices if path.is_dir()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one single-UI annotation directory: {choices}")
    return found[0]


def resolve_image(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("infos.image_path must be a nonempty relative path")
    relative = PurePosixPath(raw.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in raw:
        raise ValueError(f"Expected a safe dataset-relative image path: {raw!r}")
    parts = relative.parts
    if not parts:
        raise ValueError("Empty image path")
    # The card uses single_UIs, while the current files use single_UIs_cn.
    if parts[0] in ("single_UIs", "single_UIs_cn"):
        candidates = [root / name / Path(*parts[1:]) for name in ("single_UIs_cn", "single_UIs")]
    else:
        candidates = [root / Path(*parts), root / "single_UIs_cn" / Path(*parts),
                      root / "single_UIs" / Path(*parts)]
    found = list(dict.fromkeys(path.absolute() for path in candidates if path.is_file()))
    if len(found) != 1:
        raise ValueError(f"Image path has {len(found)} matches, expected one: {raw!r}")
    return found[0]


def convert_boxes(raw, width: int, height: int, boundary_policy: str = "error") -> list[list[float]]:
    if boundary_policy not in ("error", "clip"):
        raise ValueError(f"Unknown bbox boundary policy: {boundary_policy!r}")
    if not isinstance(raw, list):
        raise ValueError("infos.box_list must be a list (explicit [] for a negative)")
    result = []
    for box in raw:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Expected an xywh box, got {box!r}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError(f"Box coordinates must be finite numbers: {box!r}")
        x, y, w, h = box
        if w <= 0 or h <= 0 or not all(math.isfinite(v) for v in (x + w, y + h)):
            raise ValueError(f"xywh box must have positive size and finite corners: {box!r}")
        corners = [x, y, x + w, y + h]
        bounded = [max(0, min(width, corners[0])), max(0, min(height, corners[1])),
                   max(0, min(width, corners[2])), max(0, min(height, corners[3]))]
        if boundary_policy == "error" and corners != bounded:
            raise ValueError(f"xywh box {box!r} is invalid for image size {(width, height)}; "
                             "no automatic clipping (use --bbox-boundary-policy clip to record and clip intersecting boxes)")
        if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
            raise ValueError(f"xywh box {box!r} has no positive-area intersection with image size {(width, height)}; "
                             "refusing to drop a GT box or turn a positive into a negative")
        result.append(bounded if boundary_policy == "clip" else corners)
    return result


def normalize_image_size(raw) -> list[int]:
    """Accept UI-Lens [W, H] and the observed singleton-wrapped [[W, H]]."""
    size = raw
    if isinstance(size, list) and len(size) == 1 and isinstance(size[0], list):
        size = size[0]
    if not isinstance(size, list) or len(size) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size
    ):
        raise ValueError(f"infos.image_size must be [W,H] or [[W,H]] with positive integers, got {raw!r}")
    return list(size)


def prepare(root: Path, output: Path | None = None, bbox_boundary_policy: str = "error") -> dict:
    if bbox_boundary_policy not in ("error", "clip"):
        raise ValueError(f"Unknown bbox boundary policy: {bbox_boundary_policy!r}")
    root = root.expanduser().resolve(strict=True)
    source = annotation_dir(root)
    report = {"source": str(root), "coordinate_format": "pixel_xyxy",
              "bbox_boundary_policy": bbox_boundary_policy, "tasks": {}}
    all_records = {}
    clipping_audit = []
    image_sizes = {}
    for source_name, (task, issue) in TASKS.items():
        path = source / f"{source_name}.jsonl"
        records, seen_paths, seen_stems = [], set(), {}
        target_values, label_values = Counter(), Counter()
        positive, box_count = 0, 0
        clipped_images, clipped_boxes, max_clip_pixels, max_removed_area_ratio = 0, 0, 0, 0.0
        for line, row in read_rows(path):
            try:
                info = row["infos"]
                image = resolve_image(root, info["image_path"])
                if image not in image_sizes:
                    with Image.open(image) as opened:
                        image_sizes[image] = opened.size
                        if opened.getexif().get(274, 1) != 1:
                            raise ValueError("Nontrivial EXIF orientation; reconcile image and annotation coordinates first")
                        opened.load()
                width, height = image_sizes[image]
                if normalize_image_size(info.get("image_size")) != [width, height]:
                    raise ValueError(f"infos.image_size {info.get('image_size')!r} != actual size {[width, height]}")
                target = info.get("target_problem")
                allowed = {norm(v) for v in [task, source_name, *ALIASES[task]]}
                if not isinstance(target, str) or norm(target) not in allowed:
                    raise ValueError(f"Unrecognized target_problem {target!r} for {task}; inspect its meaning before adding an alias")
                labels = info.get("label_list")
                boxes = convert_boxes(info.get("box_list"), width, height, bbox_boundary_policy)
                corrections = []
                for box_index, (raw_box, converted) in enumerate(zip(info["box_list"], boxes)):
                    x, y, w, h = raw_box
                    original_xyxy = [x, y, x + w, y + h]
                    if original_xyxy != converted:
                        corrections.append({
                            "box_index": box_index, "original_xywh": raw_box,
                            "original_xyxy": original_xyxy, "clipped_xyxy": converted,
                            "max_clip_pixels": max(abs(a - b) for a, b in zip(original_xyxy, converted)),
                            "removed_area_ratio": 1 - ((converted[2] - converted[0]) / w) * ((converted[3] - converted[1]) / h),
                        })
                if not isinstance(labels, list) or len(labels) != len(boxes):
                    raise ValueError("label_list and box_list must be parallel lists; inspect any negative sentinels explicitly")
                # The task category is declared by target_problem and its source
                # file. label_list may describe elements; retain it as metadata.
                target_values[target] += 1
                label_values.update(json.dumps(label, ensure_ascii=False, sort_keys=True) for label in labels)
                if image in seen_paths:
                    raise ValueError(f"Repeated image within a task: {image}; merge/check duplicate annotations first")
                if image.stem in seen_stems and seen_stems[image.stem] != image:
                    raise ValueError(f"Duplicate filename stem: {image.stem}; the legacy scorer joins predictions by stem")
                if "figma" in str(image).lower():
                    raise ValueError("Path contains 'figma', which the legacy scorer filters; use a neutral dataset path")
                seen_paths.add(image)
                seen_stems[image.stem] = image
                positive += bool(boxes)
                box_count += len(boxes)
                if corrections:
                    clipped_images += 1
                    clipped_boxes += len(corrections)
                    max_clip_pixels = max(max_clip_pixels, *(c["max_clip_pixels"] for c in corrections))
                    max_removed_area_ratio = max(max_removed_area_ratio, *(c["removed_area_ratio"] for c in corrections))
                    clipping_audit.append({"task": task, "image": str(image), "source_file": str(path),
                                           "source_line": line, "image_size": [width, height], "clipped_boxes": corrections})
                records.append({
                    "id": row.get("id", f"{task}:{line}"),
                    "images": [str(image)],
                    "answer": {"bbox": boxes, "types": [issue] * len(boxes)},
                    "extra_info": {"dataset": "UI_lens", "task": task,
                                   "source_file": str(path), "source_line": line,
                                   "original_infos": info,
                                   "bbox_conversion": {"boundary_policy": bbox_boundary_policy,
                                                       "clipped_boxes": corrections}},
                })
            except (KeyError, ValueError, TypeError, OSError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
        if not records:
            raise ValueError(f"No records in {path}")
        report["tasks"][task] = {
            "rows": len(records), "positive": positive,
            "negative": len(records) - positive, "boxes": box_count,
            "clipped_images": clipped_images, "clipped_boxes": clipped_boxes,
            "max_clip_pixels": max_clip_pixels, "max_removed_area_ratio": max_removed_area_ratio,
            "target_problem_values": dict(target_values), "label_values": dict(label_values),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        all_records[task] = records
    report["unique_images"] = len(image_sizes)
    report["clipped_image_task_records"] = len(clipping_audit)
    report["clipped_boxes"] = sum(t["clipped_boxes"] for t in report["tasks"].values())
    if output is not None:
        output = output.expanduser().absolute()
        # Validate all five files before writing, and use a new output directory.
        output.mkdir(parents=True, exist_ok=False)
        for task, records in all_records.items():
            with (output / f"test_ui_{task}_wcnt_no_figma.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (output / "conversion_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (output / "bbox_clipping_audit.jsonl").open("w", encoding="utf-8") as handle:
            for record in clipping_audit:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Omit to validate and report without writing")
    parser.add_argument("--bbox-boundary-policy", choices=("error", "clip"), default="error",
                        help="error: reject out-of-bounds boxes; clip: intersect with the image and retain a clipping audit")
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_dir, args.bbox_boundary_policy), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
