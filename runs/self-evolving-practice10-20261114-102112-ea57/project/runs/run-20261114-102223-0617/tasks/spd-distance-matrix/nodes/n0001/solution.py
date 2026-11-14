"""Optimized implementation using scipy.spatial.distance.cdist."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    if len(points) == 0:
        return []
    
    # Convert list of tuples to numpy array
    arr = np.array(points)
    
    # Compute pairwise Euclidean distances using optimized BLAS-backed computation
    dist_array = cdist(arr, arr, metric='euclidean')
    
    # Convert back to list of lists to maintain original return type
    return dist_array.tolist()
