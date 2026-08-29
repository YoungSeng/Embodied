#!/usr/bin/env python3
"""Return 10 after two validation points improve neither raw macro metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args(argv)


def evaluate(rows: list[dict], *, patience: int, min_delta: float) -> dict:
    validation = sorted(
        (
            row for row in rows
            if row.get("evaluation_status") == "success"
            and row.get("evaluation_split", "validation") == "validation"
            and int(row.get("step", 0)) > 0
        ),
        key=lambda row: int(row["step"]),
    )
    best_image = float("-inf")
    best_bbox = float("-inf")
    stale = 0
    points = []
    for row in validation:
        image = float(row["image_macro_f1"])
        bbox = float(row["bbox_macro_f1"])
        improved_image = image > best_image + min_delta
        improved_bbox = bbox > best_bbox + min_delta
        improved = improved_image or improved_bbox
        stale = 0 if improved else stale + 1
        best_image = max(best_image, image)
        best_bbox = max(best_bbox, bbox)
        points.append(
            {
                "step": int(row["step"]),
                "image_macro_f1": image,
                "bbox_macro_f1": bbox,
                "improved_image": improved_image,
                "improved_bbox": improved_bbox,
                "stale_points": stale,
            }
        )
    return {
        "schema_version": 1,
        "validation_points": points,
        "patience": patience,
        "best_image_macro_f1": None if best_image == float("-inf") else best_image,
        "best_bbox_macro_f1": None if best_bbox == float("-inf") else best_bbox,
        "consecutive_non_improving_points": stale,
        "should_stop": stale >= patience,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.patience <= 0 or args.min_delta < 0:
        raise ValueError("patience must be positive and min-delta non-negative")
    payload = json.loads(args.history.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    rows = payload.get("evaluations", []) if isinstance(payload, dict) else payload
    result = evaluate(rows, patience=args.patience, min_delta=args.min_delta)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 10 if result["should_stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
