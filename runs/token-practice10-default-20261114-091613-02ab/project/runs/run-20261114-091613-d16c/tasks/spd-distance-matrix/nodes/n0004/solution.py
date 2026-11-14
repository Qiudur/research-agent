"""Optimized implementation using scipy.spatial.distance.cdist."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    if not points:
        return []
    
    # Convert list of tuples to numpy array
    arr = np.array(points)
    
    # Use cdist with euclidean metric - highly optimized C implementation
    result = cdist(arr, arr, metric='euclidean')
    
    # Convert back to list of lists for API compatibility
    return result.tolist()
