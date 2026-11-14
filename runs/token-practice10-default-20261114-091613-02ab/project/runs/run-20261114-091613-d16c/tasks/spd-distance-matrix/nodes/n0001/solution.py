"""Optimized distance matrix with preallocation, symmetry, and list comprehension."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances.
    
    Optimizations applied:
    1. Preallocation - creates fixed-size result upfront
    2. Symmetry - only computes upper triangle, mirrors to lower
    3. Direct indexing - avoids append() overhead
    """
    n = len(points)
    if n == 0:
        return []
    
    # Preallocate result as list of fixed-length lists
    result: list[list[float]] = [[0.0] * n for _ in range(n)]
    
    # Compute upper triangle (including diagonal), exploit symmetry
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist = sqrt(dx * dx + dy * dy)
            result[i][j] = dist
            result[j][i] = dist  # Mirror for symmetry
    
    return result
