#!/usr/bin/env python3
"""GT-free, strict non-overlapping rectangular tiling for UI5 inference.

The module is intentionally independent from the training-only GT repair
pipeline.  Horizontal tiles partition every source pixel exactly once and carry
enough geometry to map local predictions back to the original image before
cross-tile de-duplication.
"""
from __future__ import annotations

import argparse
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


def _choose_safe_horizontal_seams(
    *,
    height: int,
    count: int,
    gaps: Sequence[tuple[int, int]],
    minimum_core_height: int,
) -> list[int] | None:
    """Choose globally safe seams with a small dynamic-programming search.

    One seam candidate is derived from every detector-free gap for every ideal
    division point.  The DP minimizes core-height imbalance while requiring
    increasing seam coordinates and the requested minimum core height.  No
    detector-crossing fallback exists: callers reduce the tile count instead.
    """

    if count <= 1:
        return []
    target_core = height / count
    candidates_by_index: list[list[tuple[int, int, int]]] = []
    for seam_index in range(1, count):
        desired = seam_index * target_core
        global_lower = seam_index * minimum_core_height
        global_upper = height - (count - seam_index) * minimum_core_height
        candidates: list[tuple[float, int, int, int]] = []
        for gap_index, (gap_start, gap_end) in enumerate(gaps):
            allowed_start = max(1, gap_start, global_lower)
            allowed_end = min(height - 1, gap_end, global_upper)
            if allowed_end < allowed_start:
                continue
            position = int(round(min(max(desired, allowed_start), allowed_end)))
            gap_width = max(0, gap_end - gap_start)
            local_cost = ((position - desired) / max(1.0, target_core)) ** 2
            # Width is only a tie-breaker after safety/count/height balance.
            local_cost -= min(1.0, gap_width / max(1, height)) * 1e-6
            candidates.append((local_cost, position, gap_index, gap_width))
        # Retain the globally best candidates, plus extremes that can be needed
        # for a feasible monotonic combination on pages with hundreds of gaps.
        candidates.sort(key=lambda item: (item[0], -item[3], item[1]))
        retained = candidates[:96]
        if candidates:
            retained.extend((min(candidates, key=lambda item: item[1]), max(candidates, key=lambda item: item[1])))
        dedup = {
            (position, gap_index): (position, gap_index, gap_width)
            for _cost, position, gap_index, gap_width in retained
        }
        candidates_by_index.append(sorted(dedup.values(), key=lambda item: item[0]))

    # state: candidate -> (cost, seam tuple).  Cost includes completed core
    # heights; the final core is added after the last seam.
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for position, gap_index, gap_width in candidates_by_index[0]:
        core_cost = ((position - target_core) / max(1.0, target_core)) ** 2
        core_cost -= min(1.0, gap_width / max(1, height)) * 1e-6
        states[(position, gap_index)] = (core_cost, (position,))
    for candidates in candidates_by_index[1:]:
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for position, gap_index, gap_width in candidates:
            best: tuple[float, tuple[int, ...]] | None = None
            for (previous, previous_gap), (cost, seams) in states.items():
                if gap_index < previous_gap or position - previous < minimum_core_height:
                    continue
                segment_cost = ((position - previous - target_core) / max(1.0, target_core)) ** 2
                total = cost + segment_cost - min(1.0, gap_width / max(1, height)) * 1e-6
                candidate_state = (total, (*seams, position))
                if best is None or candidate_state < best:
                    best = candidate_state
            if best is not None:
                next_states[(position, gap_index)] = best
        states = next_states
        if not states:
            return None
    feasible = [
        (cost + ((height - position - target_core) / max(1.0, target_core)) ** 2, seams)
        for (position, _gap), (cost, seams) in states.items()
        if height - position >= minimum_core_height
    ]
    if not feasible:
        return None
    return list(min(feasible)[1])


def strict_vertical_partition_metrics(
    width: int, height: int, tiles: Sequence[Sequence[int]]
) -> dict[str, Any]:
    overlaps = [max(0, int(left[3]) - int(right[1])) for left, right in zip(tiles, tiles[1:])]
    gaps = [max(0, int(right[1]) - int(left[3])) for left, right in zip(tiles, tiles[1:])]
    original_area = int(width) * int(height)
    sum_tile_area = sum(
        (int(tile[2]) - int(tile[0])) * (int(tile[3]) - int(tile[1])) for tile in tiles
    )
    union_tile_area = union_area(tiles)
    strict = bool(tiles) and (
        list(map(int, tiles[0]))[:2] == [0, 0]
        and int(tiles[-1][2]) == int(width)
        and int(tiles[-1][3]) == int(height)
        and all(
            int(tile[0]) == 0
            and int(tile[2]) == int(width)
            and int(tile[3]) > int(tile[1])
            for tile in tiles
        )
        and all(int(left[3]) == int(right[1]) for left, right in zip(tiles, tiles[1:]))
    )
    return {
        "strict_vertical_partition": strict,
        "adjacent_overlap_pixels": overlaps,
        "adjacent_overlap_pixels_total": sum(overlaps),
        "adjacent_gap_pixels": gaps,
        "adjacent_gap_pixels_total": sum(gaps),
        "sum_tile_area": sum_tile_area,
        "union_tile_area": union_tile_area,
        "original_area": original_area,
        "duplicate_pixel_area": sum_tile_area - union_tile_area,
        "processed_pixel_ratio": sum_tile_area / original_area,
    }


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
    context_pixels: int = 0,
    minimum_core_height_ratio: float = 0.35,
    strict_vertical_partition: bool = True,
) -> dict[str, Any]:
    """Build a GT-free strict partition from detector-safe horizontal seams.

    Every final tile is exactly one continuous core.  If the desired number of
    safe seams is unavailable, the tile count is reduced; no seam may cross a
    detector box and no overlap/expansion fallback exists.
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
    if context_pixels != 0:
        raise ValueError("strict detector scan requires context_pixels=0")
    if strict_vertical_partition is not True:
        raise ValueError("detector scan requires strict_vertical_partition=true")
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
    desired_tile_count = min(max_tiles, max(1, math.ceil(height / target_tile_height)))
    if normalized_task == "content_missing":
        fallback_reason = "content_missing_requires_global_view"
        desired_tile_count = 1
    elif height <= target_tile_height or max_tiles == 1:
        fallback_reason = "short_page_single_scan"

    if fallback_reason is not None:
        tiles = [[0, 0, width, height]]
        seams: list[int] = []
        seam_sources: list[str] = []
        cores = [[0, height]]
        tile_count_reduction_reason: str | None = None
    else:
        seams = []
        actual_tile_count = 1
        for candidate_count in range(desired_tile_count, 1, -1):
            target_core = height / candidate_count
            minimum_core_height = min(
                max(32, round(target_core * minimum_core_height_ratio)),
                max(1, height // candidate_count),
            )
            candidate_seams = _choose_safe_horizontal_seams(
                height=height,
                count=candidate_count,
                gaps=gaps,
                minimum_core_height=minimum_core_height,
            )
            if candidate_seams is not None:
                seams = candidate_seams
                actual_tile_count = candidate_count
                break
        seam_sources = ["detector_gap"] * len(seams)
        core_edges = [0, *seams, height]
        cores = [[left, right] for left, right in zip(core_edges, core_edges[1:])]
        tiles = [[0, int(y1), width, int(y2)] for y1, y2 in cores]
        tile_count_reduction_reason = (
            None
            if actual_tile_count == desired_tile_count
            else "insufficient_safe_detector_free_seams"
        )
        if actual_tile_count == 1:
            fallback_reason = "dense_page_no_safe_seam"

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
    unique_containment_count = sum(
        sum(
            tile[0] <= box[0]
            and tile[1] <= box[1]
            and tile[2] >= box[2]
            and tile[3] >= box[3]
            for tile in tiles
        )
        == 1
        for box in boxes
    )
    seam_crossed = sum(
        any(box[1] < seam < box[3] for seam in seams) for box in boxes
    )
    core_heights = [right - left for left, right in cores]
    partition = strict_vertical_partition_metrics(width, height, tiles)
    if not partition["strict_vertical_partition"]:
        raise AssertionError(f"detector scan is not a strict vertical partition: {tiles}")
    original_area = partition["original_area"]
    processed_area = partition["sum_tile_area"]
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
        "safe_seam_count": sum(
            max(1, start) <= min(height - 1, end) for start, end in gaps
        ),
        "seam_source": seam_sources,
        "seam_source_counts": dict(sorted(Counter(seam_sources).items())),
        "core_spans": cores,
        "minimum_core_height": min(core_heights),
        "maximum_core_height": max(core_heights),
        "core_height_ratio": max(core_heights) / max(1, min(core_heights)),
        "min_crop_height_ratio": min(tile[3] - tile[1] for tile in tiles) / height,
        "max_crop_height_ratio": max(tile[3] - tile[1] for tile in tiles) / height,
        "adjacent_overlap_ratio_mean": 0.0,
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
        "detector_bbox_unique_containment_count": unique_containment_count,
        "uncontained_detector_bbox_count": len(not_contained),
        "seam_crossed_detector_bbox_count": seam_crossed,
        "full_tile_in_multi_plan_count": full_in_multi,
        "duplicate_tile_count": duplicate,
        "nested_tile_count": nested,
        "balanced_fallback_seam_count": 0,
        "desired_tile_count": desired_tile_count,
        "actual_tile_count": len(tiles),
        "tile_count_reduction_reason": tile_count_reduction_reason,
        "fallback_reason": fallback_reason,
        "gt_used": False,
        **partition,
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
