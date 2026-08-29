#!/usr/bin/env python3
"""Migrate existing UI5 gate sidecars from norm1000 coarse boxes to pixels.

This is intentionally generation-free: it only adds explicit coordinate-space
metadata to sidecars that already contain ``coarse_boxes`` and ``image_size``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def canonical_boxes(value):
    boxes = value or []
    while isinstance(boxes, list) and len(boxes) == 1 and boxes and isinstance(boxes[0], list) and boxes[0] and isinstance(boxes[0][0], list):
        boxes = boxes[0]
    return [[float(v) for v in box] for box in boxes if isinstance(box, list) and len(box) == 4]


def migrate(path: Path) -> bool:
    record = json.loads(path.read_text(encoding="utf-8"))
    boxes = canonical_boxes(record.get("coarse_boxes_norm1000", record.get("coarse_boxes")))
    if not boxes:
        return False
    size = record.get("image_size")
    if not isinstance(size, dict) or not size.get("width") or not size.get("height"):
        raise RuntimeError(f"missing image_size for coarse boxes: {path}")
    width, height = float(size["width"]), float(size["height"])
    record.update(
        {
            "coordinate_space": "norm1000",
            "image_width": width,
            "image_height": height,
            "coarse_boxes_norm1000": boxes,
            "coarse_boxes_px": [
                [box[0] * width / 1000.0, box[1] * height / 1000.0,
                 box[2] * width / 1000.0, box[3] * height / 1000.0]
                for box in boxes
            ],
        }
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    migrated = 0
    for root in args.prediction_dirs:
        for path in root.rglob("gate/*.json"):
            migrated += int(migrate(path))
    print(json.dumps({"migrated_sidecars": migrated}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
