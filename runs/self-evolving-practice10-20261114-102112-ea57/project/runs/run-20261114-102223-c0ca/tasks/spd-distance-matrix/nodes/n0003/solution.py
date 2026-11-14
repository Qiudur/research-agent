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
    n = len(points)
    
    # Handle empty input
    if n == 0:
        return []
    
    # Convert to numpy array with explicit dtype to avoid inference overhead
    arr = np.array(points, dtype=np.float64)
    
    # Use broadcasting to compute all pairwise differences at once
    # arr[:, None, :] broadcasts to shape (n, 1, 2)
    # arr[None, :, :] broadcasts to shape (1, n, 2)
    # Resulting diff has shape (n, n, 2) containing (dx, dy) for each pair
    diff = arr[:, None, :] - arr[None, :, :]
    
    # Compute Euclidean distance using np.hypot - fully vectorized, no Python loops
    distances = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    # Convert back to list of lists using efficient .tolist()
    return distances.tolist()
