#!/usr/bin/env python3
"""GT-free, lossless rectangular tiling for UI5 inference.

The module is intentionally independent from the training-only GT repair
pipeline.  Tiles cover every source pixel, overlap at their seams, and carry
enough geometry to map local predictions back to the original image before
cross-tile de-duplication.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
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


def _occupied_y_intervals(
    boxes: Sequence[BBox], *, height: int, margin: int
) -> list[tuple[int, int]]:
    """Return only genuinely overlapping/touching detector y projections.

    The old implementation linked nearby rows and then padded a whole connected
    component by 20% of its height.  On dense pages that turns most of the page
    into one protected band.  Seam selection needs the much narrower notion of
    detector occupancy used here.
    """

    projected = sorted(
        (max(0, box[1] - margin), min(height, box[3] + margin)) for box in boxes
    )
    merged: list[list[int]] = []
    for start, end in projected:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _free_y_gaps(
    occupied: Sequence[tuple[int, int]], *, height: int
) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in occupied:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < height:
        gaps.append((cursor, height))
    return gaps


def _choose_horizontal_seams(
    *,
    height: int,
    count: int,
    gaps: Sequence[tuple[int, int]],
    search_ratio: float,
    minimum_core_height: int,
) -> tuple[list[int], list[str]]:
    """Choose increasing, balanced seams, preferring nearby detector-free gaps."""

    if count <= 1:
        return [], []
    target_core = height / count
    window = target_core * search_ratio
    seams: list[int] = []
    sources: list[str] = []
    used_gaps: set[int] = set()
    for index in range(1, count):
        desired = index * target_core
        lower = (seams[-1] if seams else 0) + minimum_core_height
        remaining = count - index
        upper = height - remaining * minimum_core_height
        candidates: list[tuple[float, int, int, int]] = []
        for gap_index, (gap_start, gap_end) in enumerate(gaps):
            if gap_index in used_gaps:
                continue
            allowed_start = max(gap_start, math.ceil(desired - window), lower)
            allowed_end = min(gap_end, math.floor(desired + window), upper)
            if allowed_end < allowed_start:
                continue
            candidate = int(round(min(max(desired, allowed_start), allowed_end)))
            # Nearest is primary; a wider gap wins ties and is more robust to
            # small detector-coordinate changes.
            candidates.append(
                (abs(candidate - desired), -(gap_end - gap_start), candidate, gap_index)
            )
        if candidates:
            _, _, seam, gap_index = min(candidates)
            used_gaps.add(gap_index)
            source = "detector_gap"
        else:
            seam = int(round(min(max(desired, lower), upper)))
            source = "balanced_fallback"
        if not lower <= seam <= upper:
            raise AssertionError(
                f"cannot create balanced seam {index}/{count}: {seam} not in [{lower}, {upper}]"
            )
        seams.append(seam)
        sources.append(source)
    if any(right <= left for left, right in zip(seams, seams[1:])):
        raise AssertionError(f"horizontal seams are not strictly increasing: {seams}")
    return seams, sources


def _tile_relation_counts(tiles: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    duplicate = len(tiles) - len({tuple(map(int, tile)) for tile in tiles})
    nested = 0
    for index, left in enumerate(tiles):
        for right in tiles[index + 1 :]:
            if (
                left[0] <= right[0]
                and left[1] <= right[1]
                and left[2] >= right[2]
                and left[3] >= right[3]
            ) or (
                right[0] <= left[0]
                and right[1] <= left[1]
                and right[2] >= left[2]
                and right[3] >= left[3]
            ):
                nested += 1
    full_in_multi = 0
    if len(tiles) > 1:
        x1 = min(int(tile[0]) for tile in tiles)
        y1 = min(int(tile[1]) for tile in tiles)
        x2 = max(int(tile[2]) for tile in tiles)
        y2 = max(int(tile[3]) for tile in tiles)
        full_in_multi = sum(list(map(int, tile)) == [x1, y1, x2, y2] for tile in tiles)
    return full_in_multi, duplicate, nested


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
    detector_margin_ratio: float = 0.003,
    detector_margin_min: int = 2,
    detector_margin_max: int = 12,
    seam_search_ratio: float = 0.25,
    context_pixels: int = 48,
    minimum_core_height_ratio: float = 0.35,
) -> dict[str, Any]:
    """Build GT-free, balanced full-width horizontal scan crops.

    Continuous, non-overlapping cores partition the image exactly.  Their seams
    prefer nearby detector-free gaps; when no such gap exists a balanced seam is
    retained.  Small context and per-box containment expansion are applied only
    after core construction, so dense pages do not balloon into several nested
    near-full crops.
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

    if not 0 <= detector_margin_ratio <= 0.05:
        raise ValueError("detector_margin_ratio must be in [0, 0.05]")
    if not 0 <= detector_margin_min <= detector_margin_max:
        raise ValueError("detector margin bounds are invalid")
    if not 0 <= seam_search_ratio <= 0.5:
        raise ValueError("seam_search_ratio must be in [0, 0.5]")
    if context_pixels < 0:
        raise ValueError("context_pixels cannot be negative")
    if not 0 < minimum_core_height_ratio <= 1:
        raise ValueError("minimum_core_height_ratio must be in (0, 1]")

    boxes = _normalized_detector_boxes(detector_boxes, width, height)
    detector_margin = min(
        detector_margin_max,
        max(detector_margin_min, round(detector_margin_ratio * height)),
    )
    occupied = _occupied_y_intervals(boxes, height=height, margin=detector_margin)
    gaps = _free_y_gaps(occupied, height=height)
    normalized_task = str(task or "").removeprefix("ui_")
    fallback_reason: str | None = None
    if normalized_task == "content_missing":
        fallback_reason = "content_missing_requires_global_view"
    elif height <= target_tile_height or max_tiles == 1:
        fallback_reason = "short_page_single_scan"

    if fallback_reason is not None:
        tiles = [[0, 0, width, height]]
        seams: list[int] = []
        seam_sources: list[str] = []
        cores = [[0, height]]
    else:
        row_count = min(max_tiles, max(1, math.ceil(height / target_tile_height)))
        minimum_core_height = max(
            32,
            min(
                max(32, height // row_count),
                round((height / row_count) * minimum_core_height_ratio),
            ),
        )
        # Ensure the bound always permits the requested number of cores.
        minimum_core_height = min(minimum_core_height, max(1, height // row_count))
        seams, seam_sources = _choose_horizontal_seams(
            height=height,
            count=row_count,
            gaps=gaps,
            search_ratio=seam_search_ratio,
            minimum_core_height=minimum_core_height,
        )
        core_edges = [0, *seams, height]
        cores = [[left, right] for left, right in zip(core_edges, core_edges[1:])]
        spans = [
            [max(0, y1 - context_pixels), min(height, y2 + context_pixels)]
            for y1, y2 in cores
        ]
        # Assign each detector box to exactly one balanced core by vertical
        # center, then minimally expand that crop until the box is complete.
        for box in boxes:
            center = (box[1] + box[3]) / 2
            owner = min(len(spans) - 1, bisect.bisect_right(seams, center))
            spans[owner][0] = min(spans[owner][0], box[1])
            spans[owner][1] = max(spans[owner][1], box[3])
        tiles = [[0, int(y1), width, int(y2)] for y1, y2 in spans]

        full_in_multi, duplicate, nested = _tile_relation_counts(tiles)
        if full_in_multi or duplicate or nested:
            # A detector bbox spanning almost the complete page can make a
            # non-nested multi-plan mathematically impossible.  One full view
            # is honest and lossless; ordinary dense pages never enter here.
            tiles = [[0, 0, width, height]]
            seams = []
            seam_sources = []
            cores = [[0, height]]
            fallback_reason = "oversized_detector_requires_global_view"

    assert_lossless_coverage(width, height, tiles)
    cuts = detector_boundary_cut_count(tiles, boxes)
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

    full_in_multi, duplicate, nested = _tile_relation_counts(tiles)
    if full_in_multi or duplicate or nested:
        raise AssertionError(
            "invalid detector scan plan: "
            f"full_in_multi={full_in_multi}, duplicate={duplicate}, nested={nested}"
        )
    contained_count = len(boxes) - len(not_contained)
    seam_crossed = sum(
        any(box[1] < seam < box[3] for seam in seams) for box in boxes
    )
    core_heights = [right - left for left, right in cores]
    overlaps = [
        max(0, int(left[3]) - int(right[1])) / height
        for left, right in zip(tiles, tiles[1:])
    ]

    original_area = width * height
    processed_area = sum((tile[2] - tile[0]) * (tile[3] - tile[1]) for tile in tiles)
    gains = [height / max(1, tile[3] - tile[1]) for tile in tiles]
    return {
        "mode": "detector_scan",
        "tiles": tiles,
        "tile_count": len(tiles),
        "detector_box_count": len(boxes),
        "connected_band_count": len(occupied),
        "protected_vertical_bands": [list(band) for band in occupied],
        "detector_margin_pixels": detector_margin,
        "horizontal_seams": seams,
        "horizontal_seam_count": len(seams),
        "seam_source": seam_sources,
        "seam_source_counts": dict(sorted(Counter(seam_sources).items())),
        "core_spans": cores,
        "minimum_core_height": min(core_heights),
        "maximum_core_height": max(core_heights),
        "core_height_ratio": max(core_heights) / max(1, min(core_heights)),
        "min_crop_height_ratio": min(tile[3] - tile[1] for tile in tiles) / height,
        "max_crop_height_ratio": max(tile[3] - tile[1] for tile in tiles) / height,
        "adjacent_overlap_ratio_mean": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "lossless_pixel_coverage_ratio": union_area(tiles) / original_area,
        "processed_pixel_ratio_with_overlap": processed_area / original_area,
        "mean_vertical_linear_gain": sum(gains) / len(gains),
        "max_vertical_linear_gain": max(gains),
        "near_full_tile_count": sum(
            ((tile[2] - tile[0]) * (tile[3] - tile[1])) / original_area > 0.8
            for tile in tiles
        ),
        "detector_boundary_cut_count": cuts,
        "detector_bbox_contained_count": contained_count,
        "detector_bbox_containment_rate": contained_count / len(boxes) if boxes else 1.0,
        "uncontained_detector_bbox_count": len(not_contained),
        "seam_crossed_detector_bbox_count": seam_crossed,
        "full_tile_in_multi_plan_count": full_in_multi,
        "duplicate_tile_count": duplicate,
        "nested_tile_count": nested,
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
