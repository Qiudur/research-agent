"""Vectorized candidate for the public pairwise-distance task."""

from __future__ import annotations

import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    if len(points) == 0:
        return []

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Each point must contain exactly two coordinates")

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    squared = x[:, None] - x[None, :]
    np.square(squared, out=squared)
    dy = y[:, None] - y[None, :]
    np.square(dy, out=dy)
    squared += dy
    np.sqrt(squared, out=squared)
    return squared.tolist()
