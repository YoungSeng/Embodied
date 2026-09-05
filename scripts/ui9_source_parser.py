"""CPU parser extracted verbatim from the supplied prepare_ui9_datasets.py v2.1.

No repair, copy, split, or GPU code is included. See ui9_parser_provenance.json.
Changes to these functions must be reconciled with that preparation source.
"""
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from PIL import Image

HL = "/mnt/bn/intelligent-service-arnold-hl"
IES = "/mnt/bn/ies-genai"

def remap(value, old, new):
    old = old.rstrip("/")
    if value == old or value.startswith(old + "/"):
        return new.rstrip("/") + value[len(old):]
    return None


class Resolver:
    def __init__(self, pairs):
        self.pairs = pairs

    @lru_cache(maxsize=200000)
    def resolve(self, raw, base="", directory=False):
        raw = os.path.expandvars(os.path.expanduser(str(raw)))
        if raw.startswith("file://"):
            raw = raw[7:]
        if base and raw.startswith(("/sample_imgs/", "/local_imgs/", "/raw_imgs/")):
            raw = str(Path(base) / raw.lstrip("/"))
        paths = [remap(raw, a, b) for a, b in self.pairs] + [raw]
        paths += [remap(raw, HL, IES), remap(raw, IES, HL)]
        candidates = []
        for value in paths:
            if not value or "://" in value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = Path(base or ".") / path
            for candidate in (path, path.with_name(path.name.replace(":", "_"))):
                candidate = Path(os.path.abspath(candidate))
                candidates.append(candidate)
                if (candidate.is_dir() if directory else candidate.is_file()):
                    return candidate, True
        # Keep the intended local destination even when the source is missing.
        return (candidates[0] if candidates else Path(raw)), False


def image_slots(record):
    """Yield mutable (container, key, source string, role) image references."""
    if not isinstance(record, dict):
        return

    def leaves(container, key, role):
        value = container[key]
        if isinstance(value, str):
            if value.strip():
                yield container, key, value, role
        elif isinstance(value, list):
            for index in range(len(value)):
                yield from leaves(value, index, role)
        elif isinstance(value, dict):
            for field in ("path", "image", "url", "image_url"):
                if field in value:
                    yield from leaves(value, field, role)

    for field, role in (("images", "main"), ("image", "main"),
                        ("ScreenShotURL", "sample"), ("LocalImgURL", "normal"),
                        ("RawImgURL", "raw")):
        if field in record:
            yield from leaves(record, field, role)
    for conversation_key in ("messages", "conversations"):
        for message in record.get(conversation_key, []) or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        for field in ("image", "image_url"):
                            if field in block:
                                yield from leaves(block, field, "message")


def primary_image(record, kind):
    slots = list(image_slots(record))
    if kind == "synthetic":
        paths = [v for _, _, v, role in slots if role == "sample"]
        if not paths and record.get("Objects") == []:
            paths = [v for _, _, v, role in slots if role == "normal"]
    else:
        paths = [v for _, _, v, role in slots if role == "main"]
        if not paths:
            paths = [v for _, _, v, role in slots if role == "message"]
    paths = list(dict.fromkeys(paths))
    if len(paths) != 1:
        raise ValueError(f"需要唯一一张标注主图，实际找到 {len(paths)} 张")
    return paths[0]


def bbox_values(value):
    if isinstance(value, dict):
        if "bbox" in value:
            return bbox_values(value["bbox"])
        for names in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"),
                      ("left", "top", "right", "bottom")):
            if all(name in value for name in names):
                value = [value[name] for name in names]
                break
        else:
            if all(k in value for k in ("x", "y", "width", "height")):
                x, y, w, h = (float(value[k]) for k in ("x", "y", "width", "height"))
                value = [x, y, x + w, y + h]
            else:
                raise ValueError("无法识别 bbox 字典")
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in value):
        value = [*value[0], *value[1]]
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(isinstance(v, bool) for v in value):
        raise ValueError("bbox 必须为 xyxy 四元组或两点坐标")
    box = list(map(float, value))
    if not all(math.isfinite(v) for v in box):
        raise ValueError("bbox 包含 NaN/Inf")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"bbox 宽高非正：{box}")
    return box


def is_box(value):
    return isinstance(value, list) and (
        (len(value) == 4 and all(not isinstance(v, (list, dict)) for v in value)) or
        (len(value) == 2 and all(isinstance(v, list) and len(v) == 2 and
                                all(not isinstance(n, (list, dict)) for n in v) for v in value)))


def location_boxes(value, field="Location"):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or value == [] or value == {}:
        return []
    if isinstance(value, dict):
        patterns = [r"rect_err(?:_?\d+)?", r"rect_mbr(?:_?\d+)?", r"rect\d*_shift",
                    r"rect_combine(?:_?\d+)?", r"rect\d*"]
        recognized = False
        for pattern in patterns:
            keys = sorted(k for k in value if re.fullmatch(pattern, k, re.I))
            if keys:
                recognized = True
                boxes = []
                for key in keys:
                    boxes.extend(location_boxes(value[key], field + "." + key))
                if boxes:
                    return boxes
        return [] if recognized else [(bbox_values(value), field)]
    if is_box(value):
        return [(bbox_values(value), field)]
    if isinstance(value, list):
        return [box for i, item in enumerate(value) for box in location_boxes(item, f"{field}[{i}]")]
    raise ValueError(f"{field} 不是有效 bbox")


def gt_boxes(record, kind):
    if kind == "synthetic":
        objects = record.get("Objects")
        if isinstance(objects, dict):
            objects = [objects]
        if not isinstance(objects, list):
            raise ValueError("缺少 Objects 列表，不能当成负样本")
        boxes = []
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"Objects[{i}] 不是对象")
            selected = location_boxes(obj.get("Location"), f"Objects[{i}].Location")
            if not selected:
                raise ValueError(f"Objects[{i}] 有异常但没有有效标注框")
            boxes.extend(selected)
        return boxes
    objects = record.get("objects")
    if not isinstance(objects, dict) or "bbox" not in objects:
        raise ValueError("缺少 objects.bbox，不能当成负样本")
    values = objects["bbox"]
    if isinstance(values, dict) or is_box(values):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("objects.bbox 不是列表")
    return [(bbox_values(value), f"objects.bbox[{i}]") for i, value in enumerate(values)]


def page_key(record):
    a, b = record.get("FigmaKey"), record.get("FigmaNodeID")
    return f"{a}:{b}" if a and b else None


def positive_number(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("坐标参考尺寸/倍率必须为有限正数")
    return value


def coordinate_factors(record, task, width, height, images):
    config = task.get("bbox", {"mode": "auto"})
    mode = config.get("mode", "auto")
    if mode == "auto":
        if task["kind"] != "synthetic":
            mode = str(record.get("objects", {}).get("bbox_type") or "real").lower()
        else:
            for scope_name, scope in (("record", record), ("Extra", record.get("Extra", {}))):
                if not isinstance(scope, dict):
                    continue
                for name in ("BBoxCanvasSize", "BBoxReferenceSize", "bbox_canvas_size", "bbox_reference_size"):
                    value = scope.get(name)
                    if value is not None:
                        if isinstance(value, dict):
                            value = [value.get("width", value.get("Width")), value.get("height", value.get("Height"))]
                        w, h = map(positive_number, value)
                        return width / w, height / h, f"{scope_name}.{name}"
                for name in ("BBoxCanvasWidth", "BBoxReferenceWidth", "BBoxLogicalWidth",
                             "bbox_canvas_width", "bbox_reference_width", "bbox_logical_width"):
                    if scope.get(name) is not None:
                        scale = width / positive_number(scope[name])
                        return scale, scale, f"{scope_name}.{name}"
            return None
    if mode in ("pixels", "pixel", "real", "absolute", "abs"):
        return 1.0, 1.0, "声明为截图像素"
    if mode in ("norm1", "normalized_1", "relative_1"):
        return width, height, "[0,1] 归一化"
    if mode in ("norm1000", "normalized_1000"):
        return width / 1000, height / 1000, "[0,1000] 归一化"
    if mode == "logical-width":
        scale = width / positive_number(config["width"])
        return scale, scale, "显式逻辑宽度"
    if mode == "canvas":
        return width / positive_number(config["width"]), height / positive_number(config["height"]), "显式 bbox 画布"
    if mode == "scale":
        sx = positive_number(config["sx"])
        return sx, positive_number(config.get("sy", sx)), "显式倍率"
    if mode == "raw-export":
        raw = record.get("RawImgURL")
        info = images.get(raw, {})
        if not info.get("ok"):
            raise ValueError("raw-export 需要可读 RawImgURL")
        scale = width / info["width"] * positive_number(config["export_scale"])
        return scale, scale, "sample宽/raw宽 × 显式raw导出倍率"
    raise ValueError(f"未识别的 bbox 坐标尺度 {mode!r}，请配置 bbox.mode")


def inspect_image(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.load()  # Decode every image; do not apply EXIF rotation or resize.
        if width <= 0 or height <= 0:
            raise ValueError("图片宽高无效")
        return {"ok": True, "width": width, "height": height}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "missing": not Path(path).is_file()}


def iter_records(path):
    with path.open("rb") as reader:
        for line_no, raw in enumerate(reader, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(record, dict):
                    raise ValueError("JSON 行不是对象")
                yield line_no, record, None
            except (ValueError, UnicodeError) as exc:
                yield line_no, None, str(exc)


def out_of_bounds(box, width, height):
    return box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height
