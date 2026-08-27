#!/usr/bin/env python3
"""GT-free, lossless rectangular tiling for UI5 inference.

The module is intentionally independent from the training-only GT repair
pipeline.  Tiles cover every source pixel, overlap at their seams, and carry
enough geometry to map local predictions back to the original image before
cross-tile de-duplication.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BBox = tuple[int, int, int, int]


def _normalized_detector_boxes(
    boxes: Iterable[Sequence[int] | Mapping[str, Any]],
    width: int,
    height: int,
) -> list[BBox]:
    normalized: list[BBox] = []
    for item in boxes:
        raw = item.get("bbox") if isinstance(item, Mapping) else item
        if raw is None or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 > x1 and y2 > y1:
            normalized.append((x1, y1, x2, y2))
    return sorted(set(normalized), key=lambda box: (box[1], box[3], box[0], box[2]))


def _vertical_connected_bands(
    boxes: Sequence[BBox],
    *,
    height: int,
    vertical_link_ratio: float,
    context_ratio: float,
    min_context_image_ratio: float,
) -> list[tuple[int, int]]:
    """Collapse vertically adjacent detector boxes into protected scan bands.

    X distance is deliberately ignored: two detected elements on the left and
    right of the same row protect the complete horizontal strip between them.
    This is the detector-only counterpart of the cropper connected graph.
    """

    if not boxes:
        return []
    link_px = max(0, math.ceil(height * vertical_link_ratio))
    components: list[list[int]] = []
    for _x1, y1, _x2, y2 in boxes:
        if components and y1 <= components[-1][1] + link_px:
            components[-1][1] = max(components[-1][1], y2)
        else:
            components.append([y1, y2])

    protected: list[tuple[int, int]] = []
    for y1, y2 in components:
        component_height = max(1, y2 - y1)
        padding = max(
            math.ceil(component_height * context_ratio),
            math.ceil(height * min_context_image_ratio),
        )
        band = (max(0, y1 - padding), min(height, y2 + padding))
        if protected and band[0] <= protected[-1][1]:
            protected[-1] = (protected[-1][0], max(protected[-1][1], band[1]))
        else:
            protected.append(band)
    return protected


def detector_boundary_cut_count(
    tiles: Sequence[Sequence[int]], detector_boxes: Sequence[Sequence[int]]
) -> int:
    """Count detector boxes crossed by any saved tile boundary."""

    cuts = 0
    for box in detector_boxes:
        x1, y1, x2, y2 = map(int, box)
        if any(
            (x1 < int(tile[0]) < x2)
            or (x1 < int(tile[2]) < x2)
            or (y1 < int(tile[1]) < y2)
            or (y1 < int(tile[3]) < y2)
            for tile in tiles
        ):
            cuts += 1
    return cuts


def generate_detector_scan_plan(
    width: int,
    height: int,
    detector_boxes: Iterable[Sequence[int] | Mapping[str, Any]],
    *,
    task: str | None = None,
    max_tiles: int = 10,
    target_tile_height: int = 960,
    overlap_ratio: float = 0.12,
    vertical_link_ratio: float = 0.025,
    context_ratio: float = 0.20,
    min_context_image_ratio: float = 0.015,
    dense_band_ratio: float = 0.80,
) -> dict[str, Any]:
    """Build GT-free, full-width horizontal scan crops.

    A regular overlapping scan guarantees that every source pixel is present.
    Detector boxes are then joined into vertical connected bands.  Tile edges
    are expanded outwards whenever they would cross a band, so text/icons are
    never cut and the undetected space between left/right neighbours remains in
    the same full-width crop.  The result contains at most ten plain rectangles.
    """

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not 1 <= int(max_tiles) <= 10:
        raise ValueError("max_tiles must be in [1, 10]")
    if target_tile_height <= 0:
        raise ValueError("target_tile_height must be positive")
    if not 0.0 < overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in (0, 1)")
    if min(vertical_link_ratio, context_ratio, min_context_image_ratio) < 0:
        raise ValueError("link/context ratios cannot be negative")
    if not 0.0 < dense_band_ratio <= 1.0:
        raise ValueError("dense_band_ratio must be in (0, 1]")

    boxes = _normalized_detector_boxes(detector_boxes, width, height)
    bands = _vertical_connected_bands(
        boxes,
        height=height,
        vertical_link_ratio=vertical_link_ratio,
        context_ratio=context_ratio,
        min_context_image_ratio=min_context_image_ratio,
    )
    normalized_task = str(task or "").removeprefix("ui_")
    fallback_reason: str | None = None
    if normalized_task == "content_missing":
        fallback_reason = "content_missing_requires_global_view"
    elif not boxes:
        fallback_reason = "detector_empty_full_image"
    elif bands and max(end - start for start, end in bands) / height > dense_band_ratio:
        fallback_reason = "dense_connected_band"
    elif height <= target_tile_height:
        fallback_reason = "short_page_single_scan"

    if fallback_reason is not None:
        tiles = [[0, 0, width, height]]
    else:
        row_count = min(max_tiles, max(2, math.ceil(height / target_tile_height)))
        overlap_px = max(1, round(min(height, target_tile_height) * overlap_ratio))
        spans = [list(span) for span in _axis_tiles(height, row_count, overlap_px)]

        # Moving an edge outwards preserves lossless coverage.  Iterate because
        # expanding through one protected band can expose an edge to another.
        for span in spans:
            changed = True
            while changed:
                changed = False
                for band_start, band_end in bands:
                    if band_start < span[0] < band_end:
                        span[0] = band_start
                        changed = True
                    if band_start < span[1] < band_end:
                        span[1] = band_end
                        changed = True
        # Exact duplicates occur on dense rows after boundary expansion; one
        # physical crop is enough.  Do not drop merely overlapping scans.
        unique_spans = list(dict.fromkeys((int(y1), int(y2)) for y1, y2 in spans))
        tiles = [[0, y1, width, y2] for y1, y2 in unique_spans]

    assert_lossless_coverage(width, height, tiles)
    cuts = detector_boundary_cut_count(tiles, boxes)
    if cuts:
        raise AssertionError(f"detector scan cut {cuts} detector boxes")
    not_contained = [
        list(box)
        for box in boxes
        if not any(
            tile[0] <= box[0]
            and tile[1] <= box[1]
            and tile[2] >= box[2]
            and tile[3] >= box[3]
            for tile in tiles
        )
    ]
    if not_contained:
        raise AssertionError(f"detector boxes not contained by a scan: {not_contained[:5]}")

    original_area = width * height
    processed_area = sum((tile[2] - tile[0]) * (tile[3] - tile[1]) for tile in tiles)
    gains = [height / max(1, tile[3] - tile[1]) for tile in tiles]
    return {
        "mode": "detector_scan",
        "tiles": tiles,
        "tile_count": len(tiles),
        "detector_box_count": len(boxes),
        "connected_band_count": len(bands),
        "protected_vertical_bands": [list(band) for band in bands],
        "lossless_pixel_coverage_ratio": union_area(tiles) / original_area,
        "processed_pixel_ratio_with_overlap": processed_area / original_area,
        "mean_vertical_linear_gain": sum(gains) / len(gains),
        "max_vertical_linear_gain": max(gains),
        "near_full_tile_count": sum(
            ((tile[2] - tile[0]) * (tile[3] - tile[1])) / original_area > 0.8
            for tile in tiles
        ),
        "detector_boundary_cut_count": cuts,
        "fallback_reason": fallback_reason,
        "gt_used": False,
    }


def generate_detector_scan_tiles(
    width: int,
    height: int,
    detector_boxes: Iterable[Sequence[int] | Mapping[str, Any]],
    **kwargs: Any,
) -> list[list[int]]:
    return generate_detector_scan_plan(
        width, height, detector_boxes, **kwargs
    )["tiles"]


def _axis_tiles(length: int, count: int, overlap_px: int) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError("axis length must be positive")
    if count <= 1:
        return [(0, length)]
    overlap_px = max(1, min(int(overlap_px), length - 1))
    tile_length = min(length, math.ceil((length + overlap_px * (count - 1)) / count))
    last_start = length - tile_length
    starts = [round(index * last_start / (count - 1)) for index in range(count)]
    spans = [(start, min(length, start + tile_length)) for start in starts]
    spans[0] = (0, spans[0][1])
    spans[-1] = (spans[-1][0], length)
    for left, right in zip(spans, spans[1:]):
        if left[1] < right[0]:
            raise AssertionError(f"lossless axis tiling produced a gap: {spans}")
    return spans


def _choose_grid(
    width: int,
    height: int,
    *,
    max_tiles: int,
    target_long_side: int,
) -> tuple[int, int]:
    candidates: list[tuple[float, int, int, int]] = []
    for rows in range(1, max_tiles + 1):
        for columns in range(1, max_tiles // rows + 1):
            count = rows * columns
            tile_width = math.ceil(width / columns)
            tile_height = math.ceil(height / rows)
            excess = max(tile_width, tile_height) / max(1, target_long_side)
            aspect_penalty = abs((tile_width / max(1, tile_height)) - 1.0) * 0.03
            unused_penalty = (max_tiles - count) * 0.002
            candidates.append((max(1.0, excess) + aspect_penalty + unused_penalty, count, rows, columns))
    _, _, rows, columns = min(candidates)
    return rows, columns


def generate_lossless_tiles(
    width: int,
    height: int,
    *,
    task: str | None = None,
    max_tiles: int = 10,
    target_long_side: int = 1600,
    overlap_ratio: float = 0.10,
    min_linear_gain: float = 1.15,
    dense_page: bool = False,
) -> list[list[int]]:
    """Return 1--10 overlapping boxes whose union is the complete image.

    No annotation or detector result is accepted by this API.  Dense pages,
    global ``ui_content_missing`` samples, small images, and layouts with weak
    scale gain deliberately retain one full-image view.
    """

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not 1 <= int(max_tiles) <= 10:
        raise ValueError("max_tiles must be in [1, 10]")
    if target_long_side <= 0:
        raise ValueError("target_long_side must be positive")
    if not 0.0 < overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in (0, 1)")
    full = [[0, 0, width, height]]
    if task == "ui_content_missing" or dense_page or max(width, height) <= target_long_side:
        return full

    rows, columns = _choose_grid(
        width,
        height,
        max_tiles=max_tiles,
        target_long_side=target_long_side,
    )
    if rows * columns <= 1:
        return full
    approximate_tile_width = math.ceil(width / columns)
    approximate_tile_height = math.ceil(height / rows)
    linear_gain = min(
        width / max(1, approximate_tile_width),
        height / max(1, approximate_tile_height),
    )
    # Long one-dimensional pages have useful gain along only one axis.
    if rows == 1:
        linear_gain = width / max(1, approximate_tile_width)
    elif columns == 1:
        linear_gain = height / max(1, approximate_tile_height)
    if linear_gain < min_linear_gain:
        return full

    overlap_x = max(1, round(min(width, target_long_side) * overlap_ratio))
    overlap_y = max(1, round(min(height, target_long_side) * overlap_ratio))
    xs = _axis_tiles(width, columns, overlap_x)
    ys = _axis_tiles(height, rows, overlap_y)
    tiles = [[x1, y1, x2, y2] for y1, y2 in ys for x1, x2 in xs]
    if not 1 <= len(tiles) <= max_tiles:
        raise AssertionError(f"invalid tile count: {len(tiles)}")
    assert_lossless_coverage(width, height, tiles)
    return tiles


def union_area(boxes: Sequence[Sequence[int]]) -> int:
    if not boxes:
        return 0
    xs = sorted({int(value) for box in boxes for value in (box[0], box[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (int(box[1]), int(box[3]))
            for box in boxes
            if int(box[0]) < right and int(box[2]) > left
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
        area += (right - left) * (covered + end - start)
    return area


def assert_lossless_coverage(
    width: int, height: int, tiles: Sequence[Sequence[int]]
) -> None:
    for tile in tiles:
        if len(tile) != 4:
            raise ValueError(f"invalid tile: {tile}")
        x1, y1, x2, y2 = map(int, tile)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"tile is outside image bounds: {tile} at {width}x{height}")
    covered = union_area(tiles)
    expected = int(width) * int(height)
    if covered != expected:
        raise AssertionError(
            f"tile union does not cover the full image: covered={covered}, expected={expected}"
        )


def global_bbox_to_tile(
    bbox: Sequence[float], tile: Sequence[int], *, clip: bool = True
) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    tx1, ty1, tx2, ty2 = map(float, tile)
    local = [x1 - tx1, y1 - ty1, x2 - tx1, y2 - ty1]
    if clip:
        local = [
            max(0.0, min(tx2 - tx1, local[0])),
            max(0.0, min(ty2 - ty1, local[1])),
            max(0.0, min(tx2 - tx1, local[2])),
            max(0.0, min(ty2 - ty1, local[3])),
        ]
    return local


def tile_bbox_to_global(
    bbox: Sequence[float],
    tile: Sequence[int],
    *,
    image_size: tuple[int, int] | None = None,
) -> list[float]:
    tx1, ty1, _, _ = map(float, tile)
    result = [
        float(bbox[0]) + tx1,
        float(bbox[1]) + ty1,
        float(bbox[2]) + tx1,
        float(bbox[3]) + ty1,
    ]
    if image_size is not None:
        width, height = image_size
        result = [
            max(0.0, min(width, result[0])),
            max(0.0, min(height, result[1])),
            max(0.0, min(width, result[2])),
            max(0.0, min(height, result[3])),
        ]
    return result


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-12, left_area + right_area - intersection)


def merge_tile_predictions(
    predictions: Iterable[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Map tile-local predictions globally, then class-aware greedy NMS."""

    mapped: list[dict[str, Any]] = []
    for prediction in predictions:
        row = dict(prediction)
        tile = row.pop("tile_bbox")
        row["bbox"] = tile_bbox_to_global(
            row["bbox"], tile, image_size=image_size
        )
        row["source_tile_bbox"] = list(tile)
        mapped.append(row)
    mapped.sort(key=lambda row: float(row.get("score", 1.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for row in mapped:
        label = row.get("label")
        if any(
            other.get("label") == label
            and _iou(row["bbox"], other["bbox"]) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(row)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--target-long-side", type=int, default=1600)
    parser.add_argument("--overlap-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tiles = generate_lossless_tiles(
        args.width,
        args.height,
        task=args.task,
        max_tiles=args.max_tiles,
        target_long_side=args.target_long_side,
        overlap_ratio=args.overlap_ratio,
    )
    payload = {"width": args.width, "height": args.height, "tiles": tiles}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
