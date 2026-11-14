"""Optimized implementation with pre-cached coordinates."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Calculate pairwise Euclidean distances with cached coordinates."""
    if not points:
        return []
    
    # Pre-extract and cache coordinates to reduce tuple unpacking in inner loop
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    result: list[list[float]] = []
    for i in range(len(xs)):
        x1, y1 = xs[i], ys[i]
        row: list[float] = []
        for j in range(len(xs)):
            dx = x1 - xs[j]
            dy = y1 - ys[j]
            row.append(sqrt(dx * dx + dy * dy))
        result.append(row)
    return result
