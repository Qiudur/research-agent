"""Optimized implementation using list comprehensions."""

from __future__ import annotations

from math import sqrt


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    return [
        [sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) for x2, y2 in points]
        for x1, y1 in points
    ]
