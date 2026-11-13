"""Hidden pilot ceiling: compute each symmetric pair once."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    size = len(points)
    result = [[0.0] * size for _ in range(size)]
    for left in range(size):
        x1, y1 = points[left]
        row = result[left]
        for right in range(left + 1, size):
            x2, y2 = points[right]
            dx = x1 - x2
            dy = y1 - y2
            distance = sqrt(dx * dx + dy * dy)
            row[right] = distance
            result[right][left] = distance
    return result
