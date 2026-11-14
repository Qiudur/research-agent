import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances using NumPy broadcasting."""
    n = len(points)
    if n == 0:
        return []
    
    # Convert to numpy array with shape (n, 2)
    arr = np.asarray(points, dtype=np.float64)
    
    # Use broadcasting to compute all pairwise differences at once
    # diff shape: (n, n, 2) - each element is [dx, dy] for a pair
    diff = arr[:, None, :] - arr[None, :, :]
    
    # Compute Euclidean distances from the differences
    # sum over last axis gives squared distance per pair
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Convert back to list of lists as required
    return dist.tolist()
