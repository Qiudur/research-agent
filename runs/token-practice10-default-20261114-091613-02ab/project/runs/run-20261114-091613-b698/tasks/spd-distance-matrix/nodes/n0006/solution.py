"""Optimized implementation using scipy.spatial.distance.cdist."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    if len(points) == 0:
        return []
    
    # Convert to numpy array for cdist
    arr = np.array(points, dtype=np.float64)
    
    # Compute pairwise Euclidean distances
    dist_matrix = cdist(arr, arr, metric='euclidean')
    
    # Convert back to list of lists
    return dist_matrix.tolist()
