#!/usr/bin/env python3
"""Normalize the raw UI v4.1 mixture into LocateAnything CPT JSONL files.

The converter is streaming: it never loads a source JSONL into memory.  Each
task family becomes one recipe entry with an explicit, equal sampling weight.
Grounding targets are rewritten to LocateAnything's ``<ref>/<box>`` grammar;
captioning, VQA, action prediction, and region description keep natural text.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/bn/intelligent-service-arnold-hl/dataset/gui/gui_base/sample/"
    "raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/"
    "data/locany_cpt_v4"
)

SOURCE_GLOBS = (
    "caption/captions/category_8_dy1_washed.jsonl",
    "grounding/agent/*.jsonl",
    "grounding/multi/*.jsonl",
    "grounding/single/*.jsonl",
    "ocr/*.jsonl",
    "referring/category_2_dy1_397k_n.jsonl",
    "referring/category_3_dy1_297k_n.jsonl",
    "vqa/*.jsonl",
)

KNOWN_TASK_ORDER = (
    "ui_caption",
    "agent_action",
    "agent_grounding",
    "ui_defect",
    "all_ui_elements",
    "single_grounding",
    "ocr",
    "referring_kg",
    "referring",
    "vqa",
)
GROUNDING_TASKS = {
    "agent_grounding",
    "ui_defect",
    "all_ui_elements",
    "single_grounding",
    "ocr",
    "agent_other",
    "multi_grounding_other",
}

QWEN_BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\)?\s*,\s*\(?\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?"
    r"\s*<\|box_end\|>",
    re.IGNORECASE,
)
QWEN_REF_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", re.DOTALL
)
LOCANY_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
OCR_LOCANY_LINE_RE = re.compile(
    r"^\s*text\t(.*?)\t\s*(<box><\d+><\d+><\d+><\d+></box>)\s*$"
)
TABULAR_GROUNDING_LINE_RE = re.compile(
    r"^\s*(.*?)\s*\t+\s*(<box><\d+><\d+><\d+><\d+></box>)\s*$"
)
LOCANY_REF_BOX_RE = re.compile(
    r"<ref>[^\n]*?</ref>\s*<box><\d+><\d+><\d+><\d+></box>"
)
FORBIDDEN_MARKERS = ("<|box_start|>", "<|box_end|>", "<|object_ref_start|>", "<|object_ref_end|>")


@dataclass
class TaskStats:
    source_records: int = 0
    written_records: int = 0
    known_dropped_records: int = 0
    rejected_records: int = 0
    grounding_records: int = 0
    natural_language_records: int = 0


class NormalizeError(ValueError):
    pass


class KnownDataDrop(NormalizeError):
    """A malformed annotation that is safe to exclude but must stay observable."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


_UNSUPPORTED_FSYNC_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def best_effort_fsync(handle: Any, path: Path) -> None:
    """Durably flush when supported, but tolerate ByteNAS ENOSYS/ENOTSUP."""

    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        warnings.warn(
            f"filesystem does not support fsync for {path} ({exc}); "
            "continuing with close + atomic replace"
        )


def classify_source(relative_path: Path) -> str:
    path = relative_path.as_posix()
    name = relative_path.name
    if path.startswith("caption/captions/"):
        return "ui_caption"
    if path.startswith("grounding/agent/"):
        if name.startswith("category_1_"):
            return "agent_action"
        if name.startswith("category_5_"):
            return "agent_grounding"
        return "agent_other"
    if path.startswith("grounding/multi/"):
        if name.startswith("category_6_"):
            return "ui_defect"
        if name.startswith("category_9_"):
            return "all_ui_elements"
        return "multi_grounding_other"
    if path.startswith("grounding/single/"):
        return "single_grounding"
    if path.startswith("ocr/"):
        return "ocr"
    if name.startswith("category_2_") and path.startswith("referring/"):
        return "referring_kg"
    if name.startswith("category_3_") and path.startswith("referring/"):
        return "referring"
    if path.startswith("vqa/"):
        return "vqa"
    raise NormalizeError(f"cannot classify source path: {relative_path}")


def discover_sources(source_root: Path) -> OrderedDict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        for path in sorted(source_root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            task = classify_source(path.relative_to(source_root))
            grouped.setdefault(task, []).append(path)

    order = list(KNOWN_TASK_ORDER)
    order.extend(sorted(set(grouped) - set(order)))
    return OrderedDict((name, grouped[name]) for name in order if grouped.get(name))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") in {None, "text"}:
                pieces.append(str(item.get("text", item.get("value", ""))))
        return "\n".join(piece for piece in pieces if piece)
    if content is None:
        return ""
    return str(content)


def normalize_conversations(record: dict[str, Any]) -> list[dict[str, str]]:
    raw_turns = record.get("conversations") or record.get("messages")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise NormalizeError("missing conversations/messages")

    turns: list[dict[str, str]] = []
    system_prefix: list[str] = []
    for raw in raw_turns:
        if not isinstance(raw, dict):
            raise NormalizeError("conversation turn is not an object")
        role = str(raw.get("from", raw.get("role", ""))).lower()
        value = _content_to_text(raw.get("value", raw.get("content"))).strip()
        if role == "system":
            if value:
                system_prefix.append(value)
            continue
        if role in {"human", "user"}:
            role = "human"
        elif role in {"gpt", "assistant"}:
            role = "gpt"
        else:
            raise NormalizeError(f"unsupported conversation role: {role!r}")
        if not value:
            raise NormalizeError(f"empty {role} turn")
        if turns and turns[-1]["from"] == role:
            turns[-1]["value"] += "\n" + value
        else:
            turns.append({"from": role, "value": value})

    if system_prefix:
        prefix = "\n".join(system_prefix)
        if turns and turns[0]["from"] == "human":
            turns[0]["value"] = prefix + "\n" + turns[0]["value"]
        else:
            turns.insert(0, {"from": "human", "value": prefix})
    if len(turns) < 2 or turns[0]["from"] != "human" or turns[-1]["from"] != "gpt":
        raise NormalizeError("conversation must start with human and end with gpt")
    if any(turns[i]["from"] == turns[i - 1]["from"] for i in range(1, len(turns))):
        raise NormalizeError("conversation roles do not alternate")
    return turns


def _flatten_image_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        candidate = value.get("path") or value.get("image") or value.get("file")
        if isinstance(candidate, str) and candidate:
            yield candidate
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_image_values(item)


def extract_image_values(record: dict[str, Any]) -> list[str]:
    # Prefer fields that usually contain canonical absolute paths.
    for key in ("images", "original_images", "image", "image_list"):
        values = list(_flatten_image_values(record.get(key)))
        if values:
            return values
    return []


@lru_cache(maxsize=200000)
def _resolve_image_path_cached(
    raw_path: str,
    source_parent: str,
    source_root: str,
    check_exists: bool,
) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    candidates = [expanded] if expanded.is_absolute() else [
        Path(source_root) / expanded,
        Path(source_parent) / expanded,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    if check_exists:
        raise NormalizeError(
            f"image does not exist: {raw_path!r}; tried "
            + ", ".join(str(path) for path in candidates)
        )
    return str(candidates[0].resolve())


def resolve_image_path(
    raw_path: str,
    source_file: Path,
    source_root: Path,
    check_exists: bool,
) -> Path:
    return Path(
        _resolve_image_path_cached(
            raw_path,
            str(source_file.parent.resolve()),
            str(source_root.resolve()),
            check_exists,
        )
    )


def _positive_image_size(width: Any, height: Any) -> tuple[float, float] | None:
    try:
        result = float(width), float(height)
    except (TypeError, ValueError):
        return None
    return result if result[0] > 0 and result[1] > 0 else None


def _image_size_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        for width_key, height_key in (
            ("width", "height"),
            ("image_width", "image_height"),
            ("img_width", "img_height"),
            ("w", "h"),
        ):
            if width_key in value and height_key in value:
                size = _positive_image_size(value[width_key], value[height_key])
                if size is not None:
                    return size
    elif isinstance(value, (list, tuple)):
        # The v4.1 records store sizes as [[width, height]] even for one image.
        if value and isinstance(value[0], (dict, list, tuple)):
            return _image_size_from_value(value[0])
        if len(value) >= 2:
            return _positive_image_size(value[0], value[1])
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX,]\s*(\d+(?:\.\d+)?)\s*", value)
        if match:
            return _positive_image_size(match.group(1), match.group(2))
    return None


@lru_cache(maxsize=200000)
def _image_size_from_file(path_value: str) -> tuple[float, float] | None:
    """Read dimensions with Pillow when available, then use dependency-free headers."""
    path = Path(path_value)
    try:
        from PIL import Image

        with Image.open(path) as image:
            size = _positive_image_size(image.width, image.height)
            if size is not None:
                return size
    except (ImportError, OSError, ValueError):
        pass

    # Data preparation is often launched from the minimal base environment.
    # PNG/JPEG/GIF/BMP headers are sufficient for obtaining dimensions and do
    # not require decoding or copying the whole screenshot.
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return _positive_image_size(width, height)
            if header[:6] in {b"GIF87a", b"GIF89a"} and len(header) >= 10:
                width, height = struct.unpack("<HH", header[6:10])
                return _positive_image_size(width, height)
            if header.startswith(b"BM") and len(header) >= 26:
                width, height = struct.unpack("<ii", header[18:26])
                return _positive_image_size(abs(width), abs(height))
            if header.startswith(b"\xff\xd8"):
                handle.seek(2)
                start_of_frame = {
                    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
                }
                while True:
                    byte = handle.read(1)
                    while byte and byte != b"\xff":
                        byte = handle.read(1)
                    if not byte:
                        break
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if not marker:
                        break
                    marker_value = marker[0]
                    if marker_value in {0x01, *range(0xD0, 0xD9)}:
                        continue
                    length_raw = handle.read(2)
                    if len(length_raw) != 2:
                        break
                    segment_length = int.from_bytes(length_raw, "big")
                    if segment_length < 2:
                        break
                    if marker_value in start_of_frame:
                        frame = handle.read(5)
                        if len(frame) != 5:
                            break
                        height = int.from_bytes(frame[1:3], "big")
                        width = int.from_bytes(frame[3:5], "big")
                        return _positive_image_size(width, height)
                    handle.seek(segment_length - 2, os.SEEK_CUR)
    except OSError:
        pass
    return None


def extract_image_size(record: dict[str, Any], image_path: Path | None) -> tuple[float, float] | None:
    containers = [record]
    containers.extend(
        value
        for key in ("infos", "info", "metadata", "meta", "image_info")
        if isinstance((value := record.get(key)), dict)
    )
    for container in containers:
        direct = _image_size_from_value(container)
        if direct is not None:
            return direct
        for key in (
            "image_size",
            "original_size",
            "ori_size",
            "width_height",
            "resolution",
        ):
            size = _image_size_from_value(container.get(key))
            if size is not None:
                return size
    if image_path is not None and image_path.is_file():
        return _image_size_from_file(str(image_path.resolve()))
    return None


def extract_input_size(record: dict[str, Any]) -> tuple[float, float] | None:
    """Return the resized canvas used to produce model-space annotations."""
    containers = [record]
    containers.extend(
        value
        for key in ("infos", "info", "metadata", "meta", "image_info")
        if isinstance((value := record.get(key)), dict)
    )
    for container in containers:
        for key in ("input_size", "resized_size", "processed_size"):
            size = _image_size_from_value(container.get(key))
            if size is not None:
                return size
    return None


def _coerce_one_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        for keys in (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        ):
            if all(key in value for key in keys):
                return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
        for key in ("bbox_2d", "bbox", "box"):
            if key in value:
                return _coerce_one_box(value[key])
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        if (
            len(value) == 2
            and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value)
        ):
            return float(value[0][0]), float(value[0][1]), float(value[1][0]), float(value[1][1])
    return None


def _coerce_boxes(value: Any) -> list[tuple[float, float, float, float]]:
    one = _coerce_one_box(value)
    if one is not None:
        return [one]
    if isinstance(value, list):
        boxes = []
        for item in value:
            box = _coerce_one_box(item)
            if box is not None:
                boxes.append(box)
        return boxes
    return []


def normalize_box(
    box: Sequence[float],
    bbox_type: str | None,
    image_size: tuple[float, float] | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in box)
    kind = (bbox_type or "").lower()
    maximum = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if maximum <= 1.0:
        x1, y1, x2, y2 = (value * 1000.0 for value in (x1, y1, x2, y2))
    elif kind in {"real", "pixel", "pixels", "absolute", "abs"} or maximum > 1000.0:
        if image_size is None:
            raise NormalizeError(f"pixel bbox requires image size: {box}")
        width, height = image_size
        x1, x2 = x1 / width * 1000.0, x2 / width * 1000.0
        y1, y2 = y1 / height * 1000.0, y2 / height * 1000.0
    result = tuple(max(0, min(1000, int(round(value)))) for value in (x1, y1, x2, y2))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise KnownDataDrop(
            "invalid_or_degenerate_bbox",
            f"invalid or degenerate bbox after normalization: {box} -> {result}",
        )
    return result  # type: ignore[return-value]


def format_ref(value: Any, fallback: str = "UI element") -> str:
    if isinstance(value, dict):
        label = value.get("label") or value.get("text") or value.get("ref") or fallback
        element_type = value.get("type")
        text = f"{label} | type={element_type}" if element_type else str(label)
    elif isinstance(value, list):
        text = " / ".join(str(item) for item in value)
    else:
        text = str(value or fallback)
    return text.replace("<ref>", "").replace("</ref>", "").strip() or fallback


def format_locany_target(refs: Sequence[Any], boxes: Sequence[tuple[int, int, int, int]]) -> str:
    lines = []
    for index, box in enumerate(boxes):
        ref = format_ref(refs[index] if index < len(refs) else "UI element")
        lines.append(
            f"<ref>{ref}</ref><box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"
        )
    return "\n".join(lines)


def structured_grounding_target(
    record: dict[str, Any],
    image_size: tuple[float, float] | None,
    default_bbox_type: str | None = None,
) -> str | None:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return None
    raw_boxes = objects.get("bbox")
    if raw_boxes is None:
        raw_boxes = objects.get("bboxes", objects.get("bbox_2d"))
    boxes = _coerce_boxes(raw_boxes)
    if not boxes:
        return None
    bbox_type = objects.get("bbox_type") or record.get("bbox_type") or default_bbox_type
    if not bbox_type and any(max(abs(value) for value in box) > 1000.0 for box in boxes):
        # A single pixel-space box identifies the coordinate system for every
        # object in the record.  Do not normalize different objects using
        # different coordinate systems merely because some happen to be <1000.
        bbox_type = "pixel"
    normalized = [normalize_box(box, str(bbox_type or ""), image_size) for box in boxes]
    refs = objects.get("ref", objects.get("label", objects.get("labels", [])))
    if not isinstance(refs, list):
        refs = [refs]
    return format_locany_target(refs, normalized)


def _walk_json_boxes(value: Any) -> Iterator[tuple[Any, Any]]:
    if isinstance(value, dict):
        raw_box = value.get("bbox_2d", value.get("bbox", value.get("box")))
        if raw_box is not None and _coerce_one_box(raw_box) is not None:
            ref = value.get("label", value.get("text", value.get("ref", value.get("type"))))
            yield ref, raw_box
        for child in value.values():
            yield from _walk_json_boxes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_boxes(child)


def json_grounding_target(
    text: str,
    image_size: tuple[float, float] | None,
    bbox_type: str | None = None,
) -> str | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    pairs = list(_walk_json_boxes(parsed))
    if not pairs:
        return None
    refs, boxes = zip(*pairs)
    coerced_boxes = [_coerce_one_box(box) or () for box in boxes]
    if not bbox_type and any(max(abs(value) for value in box) > 1000.0 for box in coerced_boxes):
        bbox_type = "pixel"
    normalized = [normalize_box(box, bbox_type, image_size) for box in coerced_boxes]
    return format_locany_target(refs, normalized)


def convert_qwen_markup(
    text: str,
    bbox_type: str = "norm1000",
    coordinate_size: tuple[float, float] | None = None,
) -> str:
    text = QWEN_REF_RE.sub(lambda match: f"<ref>{format_ref(match.group(1))}</ref>", text)

    def replace_box(match: re.Match[str]) -> str:
        box = normalize_box(
            tuple(float(match.group(i)) for i in range(1, 5)),
            bbox_type,
            coordinate_size,
        )
        return f"<box><{box[0]}><{box[1]}><{box[2]}><{box[3]}></box>"

    return QWEN_BOX_RE.sub(replace_box, text)


def convert_ocr_markup(text: str, coordinate_size: tuple[float, float] | None) -> str:
    converted = convert_qwen_markup(text, bbox_type="pixel", coordinate_size=coordinate_size)
    output_lines = []
    for line in converted.splitlines():
        match = OCR_LOCANY_LINE_RE.fullmatch(line)
        if match:
            output_lines.append(f"<ref>{format_ref(match.group(1), 'text')}</ref>{match.group(2)}")
        else:
            output_lines.append(line)
    return "\n".join(output_lines)


def canonicalize_tabular_grounding(text: str) -> str:
    """Convert ``label<TAB><box>`` rows to LocateAnything ref/box pairs."""
    output_lines = []
    for line in text.splitlines():
        match = TABULAR_GROUNDING_LINE_RE.fullmatch(line)
        if match and "<ref>" not in match.group(1):
            output_lines.append(
                f"<ref>{format_ref(match.group(1), 'UI element')}</ref>{match.group(2)}"
            )
        else:
            output_lines.append(line)
    return "\n".join(output_lines)


def validate_grounding_target(text: str) -> None:
    box_count = len(LOCANY_BOX_RE.findall(text))
    if box_count and len(LOCANY_REF_BOX_RE.findall(text)) != box_count:
        raise KnownDataDrop(
            "noncanonical_ref_box_pair",
            "grounding answer contains a box without a canonical <ref>...</ref> pair",
        )


def validate_locany_text(text: str, role: str) -> None:
    if any(marker in text for marker in FORBIDDEN_MARKERS):
        raise NormalizeError(f"unconverted Qwen marker remains in {role} text")
    for match in LOCANY_BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(value) for value in match.groups())
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise NormalizeError(f"invalid LocateAnything bbox in {role} text: {match.group(0)}")


def normalize_record(
    record: dict[str, Any],
    task: str,
    source_file: Path,
    source_root: Path,
    check_images: bool,
) -> tuple[dict[str, Any], bool]:
    turns = normalize_conversations(record)
    raw_images = extract_image_values(record)
    if not raw_images:
        raise NormalizeError("missing image path")
    images = [
        resolve_image_path(path, source_file, source_root, check_exists=check_images)
        for path in raw_images
    ]
    if not any(re.search(r"<image(?:-\d+)?>", turn["value"]) for turn in turns if turn["from"] == "human"):
        turns[0]["value"] = "<image>" * len(images) + turns[0]["value"]

    image_size = extract_image_size(record, images[0] if images else None)
    # OCR annotations were generated on infos.input_size, e.g. 672x1456,
    # while the stored screenshot is image_size, e.g. 1080x2340.
    annotation_size = (extract_input_size(record) or image_size) if task == "ocr" else image_size
    is_grounding = False
    for turn in turns:
        if task == "ocr":
            turn["value"] = convert_ocr_markup(turn["value"], annotation_size)
        else:
            turn["value"] = convert_qwen_markup(turn["value"])
        if task in GROUNDING_TASKS and turn["from"] == "gpt":
            turn["value"] = canonicalize_tabular_grounding(turn["value"])
    if task in GROUNDING_TASKS:
        # category_7 OCR annotations use absolute screenshot pixels.  Mark the
        # complete record as pixel-space so small boxes below coordinate 1000
        # are not accidentally treated as norm1000 boxes.
        default_bbox_type = "pixel" if task == "ocr" else None
        target = structured_grounding_target(record, annotation_size, default_bbox_type)
        if target is None:
            target = json_grounding_target(turns[-1]["value"], annotation_size, default_bbox_type)
        if target is not None:
            turns[-1]["value"] = target
            is_grounding = True
        elif LOCANY_BOX_RE.search(turns[-1]["value"]):
            is_grounding = True

    if is_grounding:
        validate_grounding_target(turns[-1]["value"])

    for turn in turns:
        validate_locany_text(turn["value"], turn["from"])

    normalized: dict[str, Any] = {
        "conversations": turns,
        "image": str(images[0]) if len(images) == 1 else [str(path) for path in images],
        "cpt_task": task,
    }
    if record.get("id") is not None:
        normalized["id"] = record["id"]
    return normalized, is_grounding


def portable_image_path(source: Path, images_dir: Path) -> Path:
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    suffix = source.suffix.lower() or ".img"
    destination = images_dir / f"{digest}-{source.stem}{suffix}"
    if not destination.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination.resolve()


def make_portable(record: dict[str, Any], images_dir: Path, output_dir: Path) -> None:
    values = record["image"] if isinstance(record["image"], list) else [record["image"]]
    copied = [portable_image_path(Path(value), images_dir) for value in values]
    relative = [path.relative_to(output_dir) for path in copied]
    record["image"] = str(relative[0]) if len(relative) == 1 else [str(path) for path in relative]


def atomic_text_writer(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=destination.name + ".", delete=False
    )
    return handle, Path(handle.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recipe-name", default="locany_cpt_train.json")
    parser.add_argument(
        "--max-records-per-task",
        type=int,
        default=0,
        help="0 keeps all rows; a small positive value creates a smoke set",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images for a portable smoke set")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--max-error-rate", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--split-progress-every",
        type=int,
        default=1000,
        help="row interval for the three full split/hash passes",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="only normalize the combined pool (intended for converter smoke tests)",
    )
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--val-fast-per-task", type=int, default=200)
    parser.add_argument("--group-id-mode", choices=("sha256", "path"), default="sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.max_records_per_task < 0:
        raise SystemExit("--max-records-per-task cannot be negative")
    if not 0.0 <= args.max_error_rate < 1.0:
        raise SystemExit("--max-error-rate must be in [0, 1)")
    if not source_root.is_dir():
        raise SystemExit(f"source root does not exist: {source_root}")

    sources = discover_sources(source_root)
    missing_known = [task for task in KNOWN_TASK_ORDER if task not in sources]
    if missing_known:
        raise SystemExit(f"missing required CPT task sources: {missing_known}")

    annotations_dir = output_dir / "annotations"
    recipe_dir = output_dir / "recipe"
    images_dir = output_dir / "images"
    rejected_path = output_dir / "rejected.jsonl"
    destinations = {task: annotations_dir / f"{task}.jsonl" for task in sources}
    protected = [*destinations.values(), recipe_dir / args.recipe_name, output_dir / "manifest.json"]
    if not args.overwrite and any(path.exists() for path in protected):
        raise SystemExit(f"output exists; pass --overwrite to replace files under {output_dir}")

    rejected_handle, rejected_tmp = atomic_text_writer(rejected_path)
    stats = {task: TaskStats() for task in sources}
    temp_outputs: dict[str, tuple[Any, Path]] = {}
    try:
        for task, destination in destinations.items():
            temp_outputs[task] = atomic_text_writer(destination)

        for task, task_sources in sources.items():
            output_handle = temp_outputs[task][0]
            stop_task = False
            for source_file in task_sources:
                with source_file.open("r", encoding="utf-8") as source_handle:
                    for line_number, line in enumerate(source_handle, start=1):
                        if args.max_records_per_task and stats[task].written_records >= args.max_records_per_task:
                            stop_task = True
                            break
                        if not line.strip():
                            continue
                        stats[task].source_records += 1
                        raw: Any = None
                        try:
                            raw = json.loads(line)
                            if not isinstance(raw, dict):
                                raise NormalizeError("JSONL row is not an object")
                            normalized, is_grounding = normalize_record(
                                raw,
                                task=task,
                                source_file=source_file,
                                source_root=source_root,
                                check_images=not args.skip_image_check,
                            )
                            normalized["cpt_source"] = str(
                                source_file.relative_to(source_root).as_posix()
                            )
                            normalized["cpt_source_line"] = line_number
                            if args.copy_images:
                                make_portable(normalized, images_dir, output_dir)
                            output_handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
                            stats[task].written_records += 1
                            if is_grounding:
                                stats[task].grounding_records += 1
                            else:
                                stats[task].natural_language_records += 1
                        except KnownDataDrop as exc:
                            stats[task].known_dropped_records += 1
                            rejected_handle.write(
                                json.dumps(
                                    {
                                        "task": task,
                                        "source": str(source_file),
                                        "line": line_number,
                                        "id": raw.get("id") if isinstance(raw, dict) else None,
                                        "disposition": "known_data_drop",
                                        "category": exc.category,
                                        "reason": f"{type(exc).__name__}: {exc}",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        except NormalizeError as exc:
                            stats[task].rejected_records += 1
                            rejected_handle.write(
                                json.dumps(
                                    {
                                        "task": task,
                                        "source": str(source_file),
                                        "line": line_number,
                                        "id": raw.get("id") if isinstance(raw, dict) else None,
                                        "disposition": "unexpected_normalize_reject",
                                        "category": None,
                                        "reason": f"{type(exc).__name__}: {exc}",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "unexpected CPT preparation failure: "
                                f"task={task}, source={source_file}, line={line_number}, "
                                f"error={type(exc).__name__}: {exc}"
                            ) from exc
                        if args.progress_every and stats[task].source_records % args.progress_every == 0:
                            print(
                                f"[{task}] read={stats[task].source_records:,} "
                                f"written={stats[task].written_records:,} "
                                f"known_dropped={stats[task].known_dropped_records:,} "
                                f"rejected={stats[task].rejected_records:,}",
                                flush=True,
                            )
                if stop_task:
                    break
            print(
                f"[{task}] DONE read={stats[task].source_records:,} "
                f"written={stats[task].written_records:,} "
                f"known_dropped={stats[task].known_dropped_records:,} "
                f"rejected={stats[task].rejected_records:,}",
                flush=True,
            )

        print("[prepare] phase=finalize_normalized_annotations state=START", flush=True)
        for handle, _ in temp_outputs.values():
            handle.flush()
            best_effort_fsync(handle, Path(handle.name))
            handle.close()
        rejected_handle.flush()
        best_effort_fsync(rejected_handle, Path(rejected_handle.name))
        rejected_handle.close()

        for task, destination in destinations.items():
            os.replace(temp_outputs[task][1], destination)
        os.replace(rejected_tmp, rejected_path)
        print("[prepare] phase=finalize_normalized_annotations state=DONE", flush=True)
    except Exception:
        for handle, temporary in [*temp_outputs.values(), (rejected_handle, rejected_tmp)]:
            try:
                handle.close()
            except Exception:
                pass
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise

    total_source = sum(item.source_records for item in stats.values())
    total_known_dropped = sum(item.known_dropped_records for item in stats.values())
    total_rejected = sum(item.rejected_records for item in stats.values())
    empty = [task for task, item in stats.items() if item.written_records == 0]
    if empty:
        raise SystemExit(f"no valid records for tasks: {empty}")
    known_drop_rate = total_known_dropped / max(total_source, 1)
    error_rate = total_rejected / max(total_source, 1)
    if error_rate > args.max_error_rate:
        raise SystemExit(
            f"unexpected rejected rate {error_rate:.6%} exceeds "
            f"--max-error-rate={args.max_error_rate:.6%}; "
            f"inspect {rejected_path}"
        )

    recipe = OrderedDict()
    for task in sources:
        recipe[f"locany_cpt_{task}"] = {
            "annotation": [f"../annotations/{destinations[task].name}"],
            "root": ".." if args.copy_images else "",
            "paths_relative_to_meta": True,
            "repeat_time": 1.0,
            "sampling_weight": 1.0,
            "cpt_task": task,
            "cpt_split": "all",
            "data_augment": False,
        }
    combined_recipe_name = args.recipe_name if args.no_split else "locany_cpt_all.json"
    recipe_path = recipe_dir / combined_recipe_name
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_summary = None
    if not args.no_split:
        from split_locany_cpt import split_recipe

        print(
            "[prepare] phase=group_level_split state=START "
            f"mode={args.group_id_mode}",
            flush=True,
        )
        split_summary = split_recipe(
            recipe_path,
            output_dir,
            seed=args.split_seed,
            val_fraction=args.val_fraction,
            val_fast_per_task=args.val_fast_per_task,
            group_id_mode=args.group_id_mode,
            train_recipe_name=args.recipe_name,
            progress_every=args.split_progress_every,
        )
        print("[prepare] phase=group_level_split state=DONE", flush=True)
        recipe_path = recipe_dir / args.recipe_name

    manifest = {
        "format": "LocateAnything conversations/image with <ref>/<box> coordinates in [0,1000]",
        "sampling": "equal task-family sampling via sampling_weight=1.0",
        "source_root": str(source_root),
        "recipe": str(recipe_path),
        "combined_recipe": str(recipe_dir / combined_recipe_name),
        "split": split_summary,
        "portable_images": bool(args.copy_images),
        "max_records_per_task": args.max_records_per_task,
        "tasks": {task: asdict(item) for task, item in stats.items()},
        "total_written": sum(item.written_records for item in stats.values()),
        "total_known_dropped": total_known_dropped,
        "known_drop_rate": known_drop_rate,
        "total_rejected": total_rejected,
        "rejected_rate": error_rate,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"recipe={recipe_path}")
    print(f"manifest={manifest_path}")
    for task, item in stats.items():
        print(
            f"{task:24s} written={item.written_records:9,d} "
            f"known_dropped={item.known_dropped_records:6,d} "
            f"rejected={item.rejected_records:6,d} weight=1.0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
