"""NumPy-vectorized Levenshtein distance with scipy investigation.

Note: scipy.spatial.distance does not support edit/Levenshtein distance.
It provides metrics like euclidean, manhattan, cosine for numeric vectors.
We retain NumPy optimization which is well-suited for string edit distance.
"""

from __future__ import annotations
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(func):
        """Fallback decorator when numba is unavailable."""
        return func


@njit
def _edit_distance_jit(a_arr: np.ndarray, b_arr: np.ndarray) -> int:
    """Core DP computation with Numba JIT compilation.
    
    Args:
        a_arr: Character codes of first string (shorter)
        b_arr: Character codes of second string (longer)
        
    Returns:
        Edit distance as integer
    """
    m, n = len(a_arr), len(b_arr)
    
    if m == 0:
        return n
    if n == 0:
        return m
    
    prev = np.zeros(n + 1, dtype=np.int64)
    curr = np.zeros(n + 1, dtype=np.int64)
    
    for j in range(n + 1):
        prev[j] = j
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a_arr[i - 1] == b_arr[j - 1] else 1
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            substitute = prev[j - 1] + cost
            curr[j] = min(insert, delete, substitute)
        
        # Copy curr to prev
        for j in range(n + 1):
            prev[j] = curr[j]
    
    return int(prev[n])


def edit_distance(a: str, b: str) -> int:
    """Compute exact unit-cost Levenshtein distance between two strings.
    
    Args:
        a: First input string
        b: Second input string
        
    Returns:
        Non-negative integer representing minimum edit operations needed
    """
    # Early termination for identical strings
    if a == b:
        return 0
    
    # Ensure we iterate over shorter string
    if len(a) > len(b):
        a, b = b, a
    
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Convert strings to character codes
    a_arr = np.array([ord(c) for c in a], dtype=np.int64)
    b_arr = np.array([ord(c) for c in b], dtype=np.int64)
    
    return _edit_distance_jit(a_arr, b_arr)
