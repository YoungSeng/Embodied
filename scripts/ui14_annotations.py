"""Source-coordinate conversion. No prompts or answers from source messages are used."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


def unpack(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            raise ValueError(f"Expected JSON coordinates, got {value[:100]!r}")
    return value


def rectangle(value, bbox_type="xyxy"):
    value = unpack(value)
    if isinstance(value, dict):
        v = {str(k).lower(): val for k, val in value.items()}
        for names in (("left", "top", "right", "bottom"), ("x1", "y1", "x2", "y2")):
            if all(k in v for k in names):
                return rectangle([v[k] for k in names])
        for names in (("x", "y", "width", "height"), ("left", "top", "width", "height"), ("x", "y", "w", "h")):
            if all(k in v for k in names):
                return rectangle([v[k] for k in names], "xywh")
        raise ValueError(f"Unsupported rectangle keys: {list(value)}")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected four rectangle coordinates: {value!r}")
    coords = list(map(float, value))
    mode = str(bbox_type).lower().replace("_", "")
    if mode in ("xywh", "ltwh"):
        coords[2] += coords[0]
        coords[3] += coords[1]
    elif mode not in ("xyxy", "ltrb", "real", "pixel", "pixels", "absolute", "0-1000", "norm1000", "norm01", "0-1", "norm", "normalized", "relative"):
        raise ValueError(f"Unsupported bbox_type: {bbox_type}")
    if not all(math.isfinite(x) for x in coords) or coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError(f"Invalid bbox: {coords}")
    return coords


def many_rectangles(value, bbox_type="xyxy"):
    value = unpack(value)
    if value is None or value == []:
        return []
    if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
        return [rectangle(box, bbox_type) for box in value]
    return [rectangle(value, bbox_type)]


def synthetic_boxes(location):
    location = unpack(location)
    if isinstance(location, list):
        return [box for entry in location for box in synthetic_boxes(entry)]
    if not isinstance(location, dict):
        raise ValueError("Objects[].Location must be a JSON object with rect fields")
    groups = [
        [k for k in location if k == "rect_err"],
        [k for k in location if k == "rect_mbr"],
        sorted(k for k in location if re.fullmatch(r"rect.*_shift", k)),
        [k for k in location if k == "rect_combine"],
        sorted((k for k in location if re.fullmatch(r"rect\d+", k)), key=lambda k: int(k[4:])),
    ]
    for keys in groups:
        boxes = [box for key in keys for box in many_rectangles(location[key])]
        if boxes:
            return boxes
    raise ValueError(f"No supported rect fields in Location: {list(location)}")


def source_boxes(record, width, height, *, synthetic):
    if synthetic:
        canvas = float(record.get("BBoxCanvasWidth", 375))
        if canvas != 375:
            raise ValueError(f"Synthetic BBoxCanvasWidth must be 375, got {canvas}")
        objects = unpack(record.get("Objects", []))
        boxes = [box for obj in objects for box in synthetic_boxes(obj["Location"])]
        if objects and not boxes:
            raise ValueError("Nonempty synthetic Objects has no boxes; cannot infer a negative")
        scale = width / canvas
        boxes = [[v * scale for v in box] for box in boxes]
    else:
        objects = unpack(record.get("objects", []))
        if isinstance(objects, dict):
            # Exported column-oriented objects.bbox / objects.bbox_type.
            bboxes = objects.get("bbox", [])
            kinds = objects.get("bbox_type", record.get("bbox_type", "xyxy"))
            if isinstance(kinds, list):
                if len(kinds) != len(bboxes):
                    raise ValueError("objects.bbox/bbox_type lengths disagree")
                objects = [{"bbox": b, "bbox_type": k} for b, k in zip(bboxes, kinds)]
            else:
                objects = [{"bbox": bboxes, "bbox_type": kinds}]
        boxes = []
        for obj in objects:
            kind = obj.get("bbox_type", record.get("bbox_type", "xyxy"))
            for box in many_rectangles(obj["bbox"], kind):
                mode = str(kind).lower()
                if mode in ("norm", "normalized"):
                    mode = "norm01" if max(abs(v) for v in box) <= 1.5 else "norm1000"
                if mode in ("norm1000", "0-1000"):
                    box = [v * (width if i % 2 == 0 else height) / 1000 for i, v in enumerate(box)]
                elif mode in ("norm01", "0-1", "relative"):
                    box = [v * (width if i % 2 == 0 else height) for i, v in enumerate(box)]
                boxes.append(box)
    clipped = []
    for box in boxes:
        box = [max(0., min(float(width if i % 2 == 0 else height), v)) for i, v in enumerate(box)]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Source bbox lies entirely outside screenshot")
        if box not in clipped:
            clipped.append(box)
    return clipped


def _image_values(record):
    value = record.get("images", record.get("image", []))
    if not isinstance(value, list):
        value = [value]
    return [str(v.get("path", v.get("url", ""))) if isinstance(v, dict) else str(v) for v in value]


def main_image(record, task_root, synthetic):
    task_root = Path(task_root)
    screenshot = record.get("ScreenShotURL") if synthetic else None
    values = _image_values(record)
    # Screenshot URL determines the defective input. Reference fields never add samples.
    candidates = []
    if screenshot:
        screenshot = str(screenshot)
        filename = Path(unquote(urlparse(screenshot).path)).name
        candidates += [screenshot, str(task_root / "sample_imgs" / filename)]
        candidates += [v for v in values if Path(unquote(urlparse(v).path)).name == filename]
        for field in ("local_images", "image_map", "downloaded_images"):
            mapping = record.get(field)
            if isinstance(mapping, dict) and screenshot in mapping:
                candidates.insert(0, str(mapping[screenshot]))
    elif synthetic:
        # An existing, explicitly empty annotation is a negative record. It may
        # designate a reference image as its input; never manufacture a new row.
        if "Objects" in record and unpack(record["Objects"]) == []:
            candidates = values[:1] + [str(record[k]) for k in ("RawImgURL", "LocalImgURL") if record.get(k)]
        else:
            raise ValueError("Synthetic record is missing ScreenShotURL; refusing reference-image fallback")
    else:
        candidates = values[:1]
    for candidate in candidates:
        if candidate.startswith(("http://", "https://")):
            continue
        path = Path(candidate)
        for option in ([path] if path.is_absolute() else [task_root / path, task_root / "sample_imgs" / path.name]):
            if option.is_file():
                return option.resolve()
    raise FileNotFoundError(f"Cannot resolve primary screenshot for {record.get('id')}: {candidates}")


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
