"""Numpy-based edit distance implementation."""

from __future__ import annotations
import numpy as np


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance using numpy arrays for efficiency.
    
    Reduces Python-level overhead by using numpy arrays for storage and
    vectorized operations where possible (cost computation, array slicing).
    """
    n, m = len(a), len(b)
    
    if n == 0:
        return m
    if m == 0:
        return n
    
    # Use numpy int32 arrays for memory efficiency
    previous = np.arange(m + 1, dtype=np.int32)
    
    for i, char_a in enumerate(a, 1):
        current = np.empty(m + 1, dtype=np.int32)
        current[0] = i
        
        # Vectorized cost computation: 0 where chars match, 1 otherwise
        cost_row = (char_a != b).astype(np.int32)
        
        # Precompute up and diagonal values using numpy slicing
        diag = previous[:-1] + cost_row
        up = previous + 1
        
        # Fill current row with sequential dependency
        left_val = i
        for j in range(m):
            left_val += 1
            val = left_val
            if up[j + 1] < val:
                val = up[j + 1]
            if diag[j] < val:
                val = diag[j]
            current[j + 1] = val
        
        previous = current
    
    return int(previous[-1])
