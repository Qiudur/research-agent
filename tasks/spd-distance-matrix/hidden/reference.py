"""Measured ceiling candidate: blockwise NumPy distances, returned as Python lists."""

from __future__ import annotations

import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    size = len(points)
    if not size:
        return []
    xy = np.asarray(points, dtype=np.float64)
    result: list[list[float]] = []
    for start in range(0, size, 128):
        block = xy[start:start + 128]
        dx = block[:, None, 0] - xy[None, :, 0]
        dy = block[:, None, 1] - xy[None, :, 1]
        result.extend(np.sqrt(dx * dx + dy * dy).tolist())
    return result
