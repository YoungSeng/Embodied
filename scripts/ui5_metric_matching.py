#!/usr/bin/env python3
"""Threshold-aware one-to-one IoU matching shared by UI5 scorers.

Maximizing raw IoU before applying an IoU threshold can under-count true
positives.  For example, raw-IoU Hungarian assignment chooses the diagonal of
``[[.90, .11], [.10, 0]]`` even though the off-diagonal is the only assignment
with two matches at threshold ``.1``.

The objective below is lexicographic:

1. maximize the number of assigned edges whose IoU is at least ``threshold``;
2. among those assignments, maximize the sum of their IoUs.

The returned arrays intentionally have the same shape and meaning as SciPy's
``linear_sum_assignment`` result.  Callers that retain below-threshold pairs
for diagnostics can therefore do so without changing their output schema.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _hungarian_fallback(cost_matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy rectangular Hungarian solver used when SciPy is unavailable.

    This is the shortest-augmenting-path form of the Hungarian algorithm.  The
    implementation solves ``rows <= columns`` directly and transposes the
    opposite rectangle, then restores SciPy's sorted-row return convention.
    """

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost_matrix must be two-dimensional; shape={cost.shape}")
    if not np.isfinite(cost).all():
        raise ValueError("cost_matrix contains a non-finite value")
    original_row_count, original_column_count = cost.shape
    if min(original_row_count, original_column_count) == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()

    transposed = original_row_count > original_column_count
    if transposed:
        cost = cost.T
    row_count, column_count = cost.shape

    # Arrays use the conventional one-based indexing of this Hungarian form;
    # p[j] is the row currently assigned to column j.
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    assigned_row = np.zeros(column_count + 1, dtype=np.intp)
    predecessor = np.zeros(column_count + 1, dtype=np.intp)

    for row in range(1, row_count + 1):
        assigned_row[0] = row
        minimum_reduced_cost = np.full(column_count + 1, np.inf)
        used_column = np.zeros(column_count + 1, dtype=bool)
        current_column = 0
        while True:
            used_column[current_column] = True
            current_row = int(assigned_row[current_column])
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used_column[column]:
                    continue
                reduced_cost = (
                    cost[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_reduced_cost[column]:
                    minimum_reduced_cost[column] = reduced_cost
                    predecessor[column] = current_column
                # Strict comparison gives stable lower-column tie breaking.
                if minimum_reduced_cost[column] < delta:
                    delta = minimum_reduced_cost[column]
                    next_column = column
            if not np.isfinite(delta) or next_column == 0:
                raise RuntimeError("Hungarian fallback could not augment assignment")
            for column in range(column_count + 1):
                if used_column[column]:
                    row_potential[assigned_row[column]] += delta
                    column_potential[column] -= delta
                elif column:
                    minimum_reduced_cost[column] -= delta
            current_column = next_column
            if assigned_row[current_column] == 0:
                break

        while True:
            previous_column = int(predecessor[current_column])
            assigned_row[current_column] = assigned_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    row_to_column = np.full(row_count, -1, dtype=np.intp)
    for column in range(1, column_count + 1):
        row = int(assigned_row[column])
        if row:
            row_to_column[row - 1] = column - 1
    if np.any(row_to_column < 0):
        raise RuntimeError("Hungarian fallback returned an incomplete assignment")

    rows = np.arange(row_count, dtype=np.intp)
    columns = row_to_column
    if transposed:
        rows, columns = columns, rows
        order = np.argsort(rows, kind="stable")
        rows, columns = rows[order], columns[order]
    return rows, columns


def threshold_aware_linear_sum_assignment(
    iou_matrix: Any,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cardinality-first, IoU-second complete assignment.

    Every valid IoU must be finite and in ``[0, 1]``.  SciPy is used when
    available; otherwise the module's pure NumPy rectangular Hungarian solver
    keeps CPU-only preflight and snapshot freezing functional.
    """

    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"iou_matrix must be two-dimensional; shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("iou_matrix contains a non-finite value")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("iou_matrix values must be in [0, 1]")

    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be finite and in [0, 1]; got {threshold!r}")

    row_count, column_count = matrix.shape
    maximum_pair_count = min(row_count, column_count)
    if maximum_pair_count == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()

    # One additional qualified edge must outweigh every possible change in
    # the secondary IoU sum.  Because each IoU is at most one and an assignment
    # has at most ``maximum_pair_count`` edges, ``maximum_pair_count + 1`` is a
    # strict cardinality bonus.  Ineligible edges carry zero weight, so they
    # cannot distort the secondary objective.
    qualified = matrix >= threshold
    cardinality_bonus = float(maximum_pair_count + 1)
    objective = np.where(qualified, cardinality_bonus + matrix, 0.0)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        row_indices, column_indices = _hungarian_fallback(-objective)
    else:
        row_indices, column_indices = linear_sum_assignment(-objective)
    return (
        np.asarray(row_indices, dtype=np.intp),
        np.asarray(column_indices, dtype=np.intp),
    )


def maximum_qualified_iou_matches(
    iou_matrix: Any,
    threshold: float,
) -> list[tuple[int, int]]:
    """Return only the qualifying pairs from the shared complete assignment."""

    matrix = np.asarray(iou_matrix, dtype=np.float64)
    row_indices, column_indices = threshold_aware_linear_sum_assignment(
        matrix, threshold
    )
    return [
        (int(row_index), int(column_index))
        for row_index, column_index in zip(row_indices, column_indices)
        if matrix[row_index, column_index] >= threshold
    ]


__all__ = [
    "maximum_qualified_iou_matches",
    "threshold_aware_linear_sum_assignment",
]
