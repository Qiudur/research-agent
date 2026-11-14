"""Optimized distance matrix with list comprehensions and math.hypot."""

from __future__ import annotations

from math import hypot


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances.
    
    Optimizations applied:
    1. List comprehensions - reduced interpreter overhead vs explicit loops
    2. math.hypot - C-implemented sqrt(x*x + y*y), more efficient and numerically stable
    3. No intermediate variables - direct computation in comprehension
    """
    return [
        [hypot(x1 - x2, y1 - y2) for x2, y2 in points]
        for x1, y1 in points
    ]
