"""Optimized distance matrix using math.hypot."""

from __future__ import annotations

from math import hypot


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for x1, y1 in points:
        row: list[float] = []
        for x2, y2 in points:
            dx = x1 - x2
            dy = y1 - y2
            row.append(hypot(dx, dy))
        result.append(row)
    return result
