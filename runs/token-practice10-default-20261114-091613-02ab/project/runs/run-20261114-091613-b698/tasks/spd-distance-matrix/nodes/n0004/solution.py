"""Optimized distance matrix using symmetry exploitation."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    n = len(points)
    result: list[list[float]] = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        x1, y1 = points[i]
        # Only compute upper triangle (j >= i) and mirror
        for j in range(i, n):
            if i == j:
                result[i][j] = 0.0
            else:
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = sqrt(dx * dx + dy * dy)
                result[i][j] = dist
                result[j][i] = dist  # Mirror due to symmetry
    
    return result
