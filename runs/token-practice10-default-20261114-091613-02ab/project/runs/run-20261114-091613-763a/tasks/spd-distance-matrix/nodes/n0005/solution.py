import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances using NumPy broadcasting."""
    n = len(points)
    if n == 0:
        return []
    
    # Convert to numpy array with shape (n, 2)
    arr = np.array(points)
    
    # Pre-allocate the distance matrix using np.empty
    dist = np.empty((n, n), dtype=float)
    
    # Use broadcasting to compute all pairwise differences at once
    # diff shape: (n, n, 2) - each element is [dx, dy] for a pair
    diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    
    # Compute Euclidean distances from the differences
    # Assign to pre-allocated array
    dist[:] = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Convert back to list of lists as required
    return dist.tolist()
