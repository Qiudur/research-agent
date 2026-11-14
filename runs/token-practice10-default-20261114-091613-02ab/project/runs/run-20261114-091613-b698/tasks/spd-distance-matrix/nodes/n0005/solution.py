"""Intentionally slow reference implementation for the first local pilot."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    _sqrt = sqrt  # Cache as local variable to avoid repeated global lookup
    result: list[list[float]] = []
    for x1, y1 in points:
        row: list[float] = []
        for x2, y2 in points:
            dx = x1 - x2
            dy = y1 - y2
            row.append(_sqrt(dx * dx + dy * dy))
        result.append(row)
    return result
