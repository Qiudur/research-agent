"""Optimized distance matrix implementation using scipy."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    # Handle empty case
    if not points:
        return []
    
    # Convert to numpy array
    arr = np.array(points)
    
    # Compute distance matrix using scipy's optimized cdist
    dist_matrix = cdist(arr, arr, metric='euclidean')
    
    # Convert back to list of lists
    result = dist_matrix.tolist()
    
    return result
