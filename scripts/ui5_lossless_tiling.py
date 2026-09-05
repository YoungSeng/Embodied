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


def _normalized_detector_records(
    boxes: Iterable[Sequence[int] | Mapping[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for detector_index, item in enumerate(boxes):
        raw = item.get("bbox") if isinstance(item, Mapping) else item
        if raw is None or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 > x1 and y2 > y1:
            source = str(item.get("source", "unknown")) if isinstance(item, Mapping) else "unknown"
            source = source if source in {"text", "icon"} else "unknown"
            detector_id = (
                str(item.get("id") or item.get("detector_id") or f"{source}_{detector_index:06d}")
                if isinstance(item, Mapping)
                else f"unknown_{detector_index:06d}"
            )
            normalized.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "source": source,
                    "detector_index": detector_index,
                    "detector_id": detector_id,
                }
            )
    return normalized


def _normalized_detector_boxes(
    boxes: Iterable[Sequence[int] | Mapping[str, Any]],
    width: int,
    height: int,
) -> list[BBox]:
    return [record["bbox"] for record in _normalized_detector_records(boxes, width, height)]


def build_raw_detector_edge_geometry(
    width: int,
    height: int,
    detector_boxes: Iterable[Sequence[int] | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build raw bbox edges, provenance, and globally safe seam candidates."""

    records = _normalized_detector_records(detector_boxes, int(width), int(height))
    matches_by_edge: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        x1, y1, x2, y2 = record["bbox"]
        for edge_name, coordinate in (("y1", y1), ("y2", y2)):
            if coordinate in {0, int(height)}:
                continue
            matches_by_edge.setdefault(int(coordinate), []).append(
                {
                    "bbox_id": str(record["detector_id"]),
                    "detector_index": int(record["detector_index"]),
                    "source": str(record["source"]),
                    "edge": edge_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
    raw_edges = sorted(matches_by_edge)
    safe_edges: list[int] = []
    unsafe_edges: list[int] = []
    for edge in raw_edges:
        if any(record["bbox"][1] < edge < record["bbox"][3] for record in records):
            unsafe_edges.append(edge)
        else:
            safe_edges.append(edge)
    edge_provenance = {
        edge: {
            "seam": edge,
            "matched_bbox_ids": [item["bbox_id"] for item in matches_by_edge[edge]],
            "matched_sources": [item["source"] for item in matches_by_edge[edge]],
            "matched_edges": [item["edge"] for item in matches_by_edge[edge]],
            "matches": matches_by_edge[edge],
            "nearest_raw_edge_distance": 0,
        }
        for edge in safe_edges
    }
    return {
        "records": records,
        "raw_boxes": [record["bbox"] for record in records],
        "raw_edge_candidates": raw_edges,
        "safe_raw_edge_candidates": safe_edges,
        "unsafe_raw_edge_candidates": unsafe_edges,
        "edge_provenance": edge_provenance,
    }


def _choose_raw_detector_edge_seams(
    *,
    height: int,
    count: int,
    candidate_edges: Sequence[int],
    minimum_core_height: int,
) -> list[int] | None:
    """Select only globally safe raw detector edges with deterministic DP."""

    if count <= 1:
        return []
    candidates = sorted(
        set(int(value) for value in candidate_edges if 0 < int(value) < int(height))
    )
    # Cost is multiplied by N^2, so all comparisons remain exact integers:
    # (segment - H/N)^2 * N^2 == (segment*N - H)^2.
    states: dict[int, tuple[int, int, tuple[int, ...]]] = {}
    for seam_index in range(1, count):
        remaining_segments = count - seam_index
        next_states: dict[int, tuple[int, int, tuple[int, ...]]] = {}
        for position in candidates:
            if position < seam_index * minimum_core_height:
                continue
            if height - position < remaining_segments * minimum_core_height:
                continue
            best: tuple[int, int, tuple[int, ...]] | None = None
            if seam_index == 1:
                segment_height = position
                rank = (
                    (segment_height * count - height) ** 2,
                    segment_height,
                    (position,),
                )
                best = rank
            else:
                for previous, (cost, maximum_height, seams) in states.items():
                    segment_height = position - previous
                    if segment_height < minimum_core_height:
                        continue
                    rank = (
                        cost + (segment_height * count - height) ** 2,
                        max(maximum_height, segment_height),
                        (*seams, position),
                    )
                    if best is None or rank < best:
                        best = rank
            if best is not None:
                next_states[position] = best
        states = next_states
        if not states:
            return None
    feasible: list[tuple[int, int, tuple[int, ...]]] = []
    for position, (cost, maximum_height, seams) in states.items():
        last_height = height - position
        if last_height < minimum_core_height:
            continue
        feasible.append(
            (
                cost + (last_height * count - height) ** 2,
                max(maximum_height, last_height),
                seams,
            )
        )
    return list(min(feasible)[2]) if feasible else None


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


def detector_task_context_bands(records, width, task):
    """Keep adjacent detector text lines / containers / element pairs together.

    Horizontal scan already preserves the full width. Protect vertical pairs
    only when their horizontal extents overlap and their vertical gap is small.
    The thresholds scale with screenshot width, independent of resolution/GT.
    Raw detector containment and the original DP objective remain unchanged.
    """
    text_tasks = {"change_line_illegal_v3", "synth_loneword"}
    boundary_tasks = {"synth_cropping", "synth_radius", "synth_inner_margin"}
    pairwise_tasks = {"synth_occlusion", "synth_small_margin"}
    if task not in text_tasks | boundary_tasks | pairwise_tasks:
        return []
    selected = [r for r in records if task not in text_tasks or r["source"] == "text"]
    selected.sort(key=lambda r: (r["bbox"][1], r["bbox"][0], r["detector_index"]))
    bands = set()
    for i, left in enumerate(selected):
        a = left["bbox"]
        for right in selected[i + 1:]:
            b = right["bbox"]
            gap = b[1] - a[3]
            # For text, two line heights covers a paragraph line break. For
            # boundaries/pairs retain the nearby detector-defined container.
            limit = max(width * .025, 2 * max(a[3]-a[1], b[3]-b[1])) if task in text_tasks else width * .08
            overlap = min(a[2], b[2]) - max(a[0], b[0])
            if gap <= limit and overlap >= .25 * min(a[2]-a[0], b[2]-b[0]):
                bands.add((min(a[1], b[1]), max(a[3], b[3])))
    return [list(band) for band in sorted(bands)]


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
    target_guard_ratio: float = 0.0,
    target_guard_min_pixels: int = 0,
    target_guard_max_pixels: int = 0,
    seam_edge_reference: str = "raw-detector-bbox",
    seam_candidates: str = "safe-raw-detector-edges-only",
) -> dict[str, Any]:
    """Build a GT-free strict partition from globally safe raw bbox edges only.

    Every final tile is exactly one continuous core.  If the desired number of
    safe raw detector edges is unavailable, the tile count is reduced; no seam
    may be synthesized inside a gap and no overlap/expansion fallback exists.
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
    if (target_guard_ratio, target_guard_min_pixels, target_guard_max_pixels) != (
        0.0,
        0,
        0,
    ):
        raise ValueError("raw detector-edge mode requires target guard 0/0/0")
    if seam_edge_reference != "raw-detector-bbox":
        raise ValueError("detector scan requires seam_edge_reference=raw-detector-bbox")
    if seam_candidates != "safe-raw-detector-edges-only":
        raise ValueError(
            "detector scan requires seam_candidates=safe-raw-detector-edges-only"
        )
    if not 0 < minimum_core_height_ratio <= 1:
        raise ValueError("minimum_core_height_ratio must be in (0, 1]")

    raw_geometry = build_raw_detector_edge_geometry(width, height, detector_boxes)
    boxes = raw_geometry["raw_boxes"]
    raw_edge_candidates = raw_geometry["raw_edge_candidates"]
    safe_raw_edge_candidates = raw_geometry["safe_raw_edge_candidates"]
    unsafe_raw_edge_candidates = raw_geometry["unsafe_raw_edge_candidates"]
    edge_provenance = raw_geometry["edge_provenance"]
    # New task context only removes existing raw-edge candidates. It never
    # moves/expands a detector box, adds a seam, or inspects task annotations.
    context_bands = detector_task_context_bands(raw_geometry["records"], width, task)
    context_safe_edges = [edge for edge in safe_raw_edge_candidates
                          if not any(top < edge < bottom for top, bottom in context_bands)]
    normalized_task = str(task or "").removeprefix("ui_")
    fallback_reason: str | None = None
    desired_tile_count = min(max_tiles, max(1, math.ceil(height / target_tile_height)))
    if normalized_task == "content_missing":
        fallback_reason = "content_missing_requires_global_view"
        desired_tile_count = 1
    elif height <= target_tile_height or max_tiles == 1:
        fallback_reason = "short_page_single_scan"
    elif not boxes:
        fallback_reason = "detector_empty_full_image"

    if fallback_reason is not None:
        tiles = [[0, 0, width, height]]
        seams: list[int] = []
        seam_sources: list[str] = []
        cores = [[0, height]]
        tile_count_reduction_reason: str | None = (
            fallback_reason if desired_tile_count > 1 else None
        )
    else:
        seams = []
        actual_tile_count = 1
        for candidate_count in range(desired_tile_count, 1, -1):
            target_core = height / candidate_count
            minimum_core_height = min(
                max(32, round(target_core * minimum_core_height_ratio)),
                max(1, height // candidate_count),
            )
            candidate_seams = _choose_raw_detector_edge_seams(
                height=height,
                count=candidate_count,
                candidate_edges=context_safe_edges,
                minimum_core_height=minimum_core_height,
            )
            if candidate_seams is not None:
                seams = candidate_seams
                actual_tile_count = candidate_count
                break
        seam_sources = ["raw_detector_edge"] * len(seams)
        core_edges = [0, *seams, height]
        cores = [[left, right] for left, right in zip(core_edges, core_edges[1:])]
        tiles = [[0, int(y1), width, int(y2)] for y1, y2 in cores]
        tile_count_reduction_reason = (
            None
            if actual_tile_count == desired_tile_count
            else "insufficient_safe_raw_detector_edges"
        )
        if actual_tile_count == 1:
            fallback_reason = "dense_page_no_valid_raw_detector_edge_seam"

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
    non_raw_edge_seam_count = sum(
        seam not in safe_raw_edge_candidates for seam in seams
    )
    nearest_raw_edge_distances = [
        min((abs(seam - edge) for edge in raw_edge_candidates), default=height)
        for seam in seams
    ]
    seam_raw_edge_provenance = []
    for seam_index, seam in enumerate(seams, 1):
        provenance = dict(edge_provenance[seam])
        provenance["distance_to_ideal_partition"] = round(
            abs(seam - seam_index * height / len(tiles))
        )
        seam_raw_edge_provenance.append(provenance)
    if non_raw_edge_seam_count or seam_crossed:
        raise AssertionError(
            "raw-edge-only seam invariant failed: "
            f"non_raw_edge={non_raw_edge_seam_count}, raw_crossed={seam_crossed}"
        )
    if unique_containment_count != len(boxes):
        raise AssertionError("not every raw detector bbox belongs to exactly one crop")
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
        "connected_band_count": 0,
        "protected_vertical_bands": context_bands,
        "task_context_policy": "ui14_detector_neighbors_v1" if context_bands or str(task or "").startswith(("synth_", "change_line_")) else None,
        "task_context_key": task,
        "context_safe_raw_detector_edge_candidates": context_safe_edges,
        "detector_margin_pixels": 0,
        "target_guard_ratio": 0.0,
        "target_guard_pixels_min": 0,
        "target_guard_pixels_max": 0,
        "target_guard_pixels_effective": 0,
        "seam_edge_reference": "raw_detector_bbox",
        "raw_detector_edge_candidates": raw_edge_candidates,
        "safe_raw_detector_edge_candidates": safe_raw_edge_candidates,
        "unsafe_raw_detector_edge_candidates": unsafe_raw_edge_candidates,
        "raw_detector_edge_candidate_count": len(raw_edge_candidates),
        "safe_raw_detector_edge_candidate_count": len(safe_raw_edge_candidates),
        "horizontal_seams": seams,
        "horizontal_seam_count": len(seams),
        "safe_seam_count": len(safe_raw_edge_candidates),
        "seam_source": seam_sources,
        "seam_source_counts": dict(sorted(Counter(seam_sources).items())),
        "seam_raw_edge_provenance": seam_raw_edge_provenance,
        "seam_nearest_raw_detector_edge_distance_pixels": nearest_raw_edge_distances,
        "non_raw_edge_seam_count": non_raw_edge_seam_count,
        "every_seam_is_raw_detector_edge": non_raw_edge_seam_count == 0,
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
        "detector_bbox_unique_containment_rate": (
            unique_containment_count / len(boxes) if boxes else 1.0
        ),
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
