#!/usr/bin/env python3
"""Render GT, TC-MSED coarse slots, final boxes, and slot bindings."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from locany_ui5_common import TASK_ISSUE_NAMES, TASK_JSONL, TASKS


COLORS = {
    "gt": (0, 220, 90),
    "coarse": (255, 170, 0),
    "final": (255, 55, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=10)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
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


def load_scorer(root: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    path = root / "qwen3vl_merge_and_score_fixed_5tasks.py"
    spec = importlib.util.spec_from_file_location("ui5_scorer_for_slot_render", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_gt_payload, module.extract_bboxes_for_issue


def sample_image_path(sample: dict[str, Any], source: Path) -> Path | None:
    value = sample.get("images", sample.get("image"))
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    return (source.parent / path).resolve(strict=False) if not path.is_absolute() else path


def load_gt(
    gt_dir: Path,
    get_gt_payload: Callable[..., Any],
    extract_bboxes: Callable[..., Any],
) -> dict[str, dict[str, list[list[float]]]]:
    output: dict[str, dict[str, list[list[float]]]] = {}
    for task in TASKS:
        source = gt_dir / TASK_JSONL[task]
        task_gt: dict[str, list[list[float]]] = {}
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                image_path = sample_image_path(sample, source)
                if image_path is None:
                    continue
                boxes = extract_bboxes(
                    get_gt_payload(sample), TASK_ISSUE_NAMES[task]
                ) or []
                parsed = [[float(value) for value in box[:4]] for box in boxes]
                task_gt[str(image_path)] = parsed
                task_gt[image_path.name] = parsed
        output[task] = task_gt
    return output


def pixel_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    maximum = max(abs(value) for value in box)
    if maximum <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif maximum <= 1000.5:
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    return [x1, y1, x2, y2]


def draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: list[list[float]],
    *,
    color: tuple[int, int, int],
    prefix: str,
    width: int,
    height: int,
    slot_indices: list[int | None] | None = None,
) -> None:
    font = ImageFont.load_default()
    for index, raw in enumerate(boxes):
        box = pixel_box(raw, width, height)
        draw.rectangle(box, outline=color, width=3)
        slot = slot_indices[index] if slot_indices and index < len(slot_indices) else None
        label = f"{prefix}{index}" + (f"/s{slot}" if slot is not None else "")
        draw.text((box[0] + 2, box[1] + 2), label, fill=color, font=font)


def main() -> int:
    args = parse_args()
    if args.per_task <= 0:
        raise ValueError("--per-task must be positive")
    prediction_dir = args.prediction_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    scorer_root = args.scorer_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    get_gt_payload, extract_bboxes = load_scorer(scorer_root)
    ground_truth = load_gt(gt_dir, get_gt_payload, extract_bboxes)
    manifest: dict[str, Any] = {"schema_version": 1, "tasks": {}}

    for task in TASKS:
        records: list[dict[str, Any]] = []
        for sidecar in sorted((prediction_dir / task / "gate").glob("*.json")):
            value = json.loads(sidecar.read_text(encoding="utf-8"))
            image_path = Path(str(value.get("image_path", "")))
            gt_boxes = ground_truth[task].get(
                str(image_path), ground_truth[task].get(image_path.name, [])
            )
            value["_sidecar"] = str(sidecar)
            value["_gt_boxes"] = gt_boxes
            records.append(value)
        # Prefer multi-box and duplicate-slot cases, then deterministic names.
        records.sort(
            key=lambda row: (
                -len(row["_gt_boxes"]),
                -float(row.get("duplicate_slot_rate") or 0.0),
                str(row.get("image_path", "")),
            )
        )
        selected = records[: args.per_task]
        task_rows = []
        for row in selected:
            image_path = Path(str(row["image_path"]))
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            gt_boxes = row["_gt_boxes"]
            coarse = row.get("coarse_boxes_px") or []
            if not coarse and row.get("coarse_boxes"):
                raise RuntimeError(
                    "legacy coarse_boxes lack pixel coordinate metadata; run "
                    "scripts/recompute_ui5_coarse_sidecars.py first"
                )
            if coarse and isinstance(coarse[0], list) and coarse[0] and isinstance(coarse[0][0], list):
                coarse = coarse[0]
            final_boxes = row.get("final_boxes_normalized_1000") or []
            bindings = row.get("box_slot_bindings") or []
            bound_slots = [binding.get("slot_index") for binding in bindings]
            draw_boxes(draw, gt_boxes, color=COLORS["gt"], prefix="GT", width=width, height=height)
            draw_boxes(draw, coarse, color=COLORS["coarse"], prefix="S", width=width, height=height)
            draw_boxes(
                draw, final_boxes, color=COLORS["final"], prefix="P",
                width=width, height=height, slot_indices=bound_slots,
            )
            destination = output_dir / task / f"{image_path.stem}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, quality=92)
            task_rows.append(
                {
                    "image_path": str(image_path),
                    "rendered_path": str(destination),
                    "gt_boxes": gt_boxes,
                    "coarse_slots": coarse,
                    "final_boxes": final_boxes,
                    "box_slot_bindings": bindings,
                    "duplicate_slot_rate": row.get("duplicate_slot_rate"),
                }
            )
        manifest["tasks"][task] = task_rows
    atomic_write_json(output_dir / "slot_diagnostics_manifest.json", manifest)
    print(output_dir / "slot_diagnostics_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
