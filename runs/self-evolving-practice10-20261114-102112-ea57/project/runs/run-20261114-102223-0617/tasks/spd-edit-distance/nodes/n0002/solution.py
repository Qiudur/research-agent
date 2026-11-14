"""NumPy-vectorized Levenshtein distance with scipy investigation.

Note: scipy.spatial.distance does not support edit/Levenshtein distance.
It provides metrics like euclidean, manhattan, cosine for numeric vectors.
We retain NumPy optimization which is well-suited for string edit distance.
"""

from __future__ import annotations
import numpy as np


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
    
    # Ensure we iterate over shorter string, vectorize over longer
    if len(a) > len(b):
        a, b = b, a
    
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Convert string b to character codes for fast comparison
    b_arr = np.array([ord(c) for c in b], dtype=np.int64)
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    vals = np.empty(n + 1, dtype=np.int64)
    
    for i, ch in enumerate(a, 1):
        ch_code = ord(ch)
        # Substitution cost: 1 if characters differ, 0 otherwise
        sub = (b_arr != ch_code).astype(np.int64)
        # Best of insertion (prev[j]+1) vs substitution (prev[j-1]+sub)
        np.minimum(prev[1:] + 1, prev[:-1] + sub, out=vals[1:])
        vals[0] = i
        # Left dependency: curr[j] = min_k<=j (vals[k] + j - k)
        # Use prefix-min trick: subtract index, accumulate, add back
        vals -= idx
        np.minimum.accumulate(vals, out=vals)
        vals += idx
        # Swap buffers for next iteration
        prev, vals = vals, prev
    
    return int(prev[-1])
