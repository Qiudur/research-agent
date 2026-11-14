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

    # Convert to numpy array for vectorized computation
    arr = np.array(points)  # shape: (n, 2)
    n = len(arr)

    # Use broadcasting to compute all pairwise differences at once
    # arr[:, None, :] broadcasts to shape (n, 1, 2)
    # arr[None, :, :] broadcasts to shape (1, n, 2)
    # Resulting diff has shape (n, n, 2) containing (dx, dy) for each pair
    diff = arr[:, None, :] - arr[None, :, :]

    # Compute Euclidean distance using np.hypot(dx, dy) - vectorized
    distances = np.hypot(diff[:, :, 0], diff[:, :, 1])

    # Convert back to list of lists to match original interface
    return distances.tolist()
