"""Optimized implementation using NumPy broadcasting for distance matrix computation."""

from __future__ import annotations

import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances using vectorized operations.

    Args:
        points: List of (x, y) coordinate tuples

    Returns:
        Square list of lists containing Euclidean distances between all pairs
    """
    # Handle empty input
    if len(points) == 0:
        return []

    # Pre-compute X and Y coordinate arrays separately - avoids tuple indexing later
    x = np.array([p[0] for p in points])  # shape: (n,)
    y = np.array([p[1] for p in points])  # shape: (n,)
    
    n = len(x)
    
    # Use broadcasting to compute all pairwise differences at once
    # x[:, None] broadcasts to shape (n, 1)
    # x[None, :] broadcasts to shape (1, n)
    # Resulting dx/dy has shape (n, n)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    # Compute Euclidean distance using np.hypot(dx, dy) - vectorized
    distances = np.hypot(dx, dy)

    # Convert back to list of lists to match original interface
    return distances.tolist()
