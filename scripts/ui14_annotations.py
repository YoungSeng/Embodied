"""UI14 labels use the same CPU parser as prepare_ui9_datasets.py v2.1."""
from __future__ import annotations
import math
from pathlib import Path
from ui9_source_parser import (Resolver, primary_image, gt_boxes, location_boxes,
                              coordinate_factors, positive_number, out_of_bounds)

_RESOLVER = Resolver([])


def synthetic_boxes(location):
    return [box for box, _ in location_boxes(location)]


def source_box_details(record, width, height, *, synthetic, task_config=None, images=None):
    kind = "synthetic" if synthetic else "annotated"
    task = {"kind": kind, "bbox": {"mode": "logical-width", "width": 375} if synthetic else {"mode": "auto"},
            **(task_config or {})}
    if task["kind"] != kind:
        raise ValueError("Manifest task kind disagrees with registered source")
    if synthetic:
        config = task["bbox"]
        if config.get("mode") != "logical-width" or positive_number(config.get("width")) != 375:
            raise ValueError("Synthetic manifest must declare logical-width 375")
        if positive_number(record.get("BBoxCanvasWidth", 375)) != 375:
            raise ValueError("Synthetic BBoxCanvasWidth must be 375")
    selected = gt_boxes(record, kind)
    if not selected:
        return {"boxes_px": [], "selected_gt": [], "scale_xy": None, "coordinate_basis": "explicit negative"}
    factors = coordinate_factors(record, task, width, height, images or {})
    if factors is None:
        raise ValueError("Source coordinates are unconfirmed")
    sx, sy, basis = factors
    boxes = [[v * (sx if i % 2 == 0 else sy) for i, v in enumerate(box)] for box, _ in selected]
    for box in boxes:
        if not all(math.isfinite(v) for v in box) or out_of_bounds(box, width, height):
            raise ValueError(f"Repaired GT is outside screenshot: {box}; intake never performs another repair")
    # Keep every selected GT in preparation order, including repeated boxes.
    return {"boxes_px": boxes, "selected_gt": [{"bbox": b, "field": f} for b, f in selected],
            "scale_xy": [sx, sy], "coordinate_basis": basis}


def source_boxes(record, width, height, *, synthetic, task_config=None, images=None):
    return source_box_details(record, width, height, synthetic=synthetic,
                              task_config=task_config, images=images)["boxes_px"]


def resolve_image(value, task_root):
    path, found = _RESOLVER.resolve(value, str(task_root))
    if not found:
        raise FileNotFoundError(f"Unreadable prepared image: {value} (resolved {path})")
    return path.resolve()


def main_image(record, task_root, synthetic):
    # Exactly one screenshot (or LocalImgURL for explicitly empty Objects).
    # No filename guesses, RawImgURL fallback, or first-image truncation.
    return resolve_image(primary_image(record, "synthetic" if synthetic else "annotated"), task_root)


def norm1000(box, width, height):
    values = [max(0, min(1000, round(v * 1000 / (width if i % 2 == 0 else height)))) for i, v in enumerate(box)]
    for low, high, extent in ((0, 2, width), (1, 3, height)):
        if values[high] <= values[low]:
            # Preserve sub-unit intersections after clipping at an input-only
            # seam: outward quantization is at most one norm1000 unit.
            values[low] = max(0, min(999, math.floor(box[low]*1000/extent)))
            values[high] = min(1000, max(values[low]+1, math.ceil(box[high]*1000/extent)))
    return values


def answer(boxes, width, height, label):
    if not boxes:
        return "<box>none</box>"
    return "".join(f"<ref>{label}</ref><box>" + "".join(f"<{v}>" for v in norm1000(box, width, height)) + "</box>" for box in boxes)


def crop_boxes(boxes, crop):
    x1, y1, x2, y2 = crop
    result, fully_contained = [], []
    for i, box in enumerate(boxes):
        b = [max(x1, box[0]), max(y1, box[1]), min(x2, box[2]), min(y2, box[3])]
        if b[0] < b[2] and b[1] < b[3]:
            result.append([b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1])
        if b == box:
            fully_contained.append(i)
    return result, fully_contained


def training_record(row, task, image, boxes, width, height, crop_id="full"):
    row = {key: value for key, value in row.items() if key != "source_metadata"}
    return {**row, "image": str(image), "task_key": task.task_key, "task_id": task.task_id,
            "source_width": row.get("width", width), "source_height": row.get("height", height),
            "source_boxes_px": row.get("boxes_px", boxes), "boxes_px": boxes, "width": width, "height": height,
            "defect_type": task.task_id, "relation_family": task.family_id,
            "crop_id": crop_id, "view_policy": task.view_policy,
            "_ui5_record_kind": "full" if task.view_policy == "full_image" else "crop",
            "_ui5_crop_source": "full_image" if task.view_policy == "full_image" else "detector_scan",
            "_ui5_source_image": row["source_image"], "_ui5_image_id": row["source_image_id"],
            "_ui5_task": task.task_key, "_ui5_sample_id": row["source_record_id"] + ":" + crop_id,
            "conversations": [{"from": "human", "value": "<image>\n" + task.prompt},
                              {"from": "gpt", "value": answer(boxes, width, height, task.prompt_label)}]}
