import numpy as np


def distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    """Compute pairwise Euclidean distances exploiting symmetry."""
    n = len(points)
    if n == 0:
        return []
    
    # Convert to numpy array with shape (n, 2)
    arr = np.array(points)
    
    # Initialize result array
    dist = np.zeros((n, n))
    
    # Only compute upper triangle (including diagonal) - exploits d[i,j] == d[j,i]
    for i in range(n):
        for j in range(i, n):
            dx = arr[i, 0] - arr[j, 0]
            dy = arr[i, 1] - arr[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d  # Exploit symmetry
    
    # Convert back to list of lists as required
    return dist.tolist()
