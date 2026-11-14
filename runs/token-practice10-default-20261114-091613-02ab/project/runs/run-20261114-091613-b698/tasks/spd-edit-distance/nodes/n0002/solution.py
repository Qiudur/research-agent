"""Numpy-accelerated edit distance implementation."""

from __future__ import annotations
import numpy as np


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    
    # Handle edge cases
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Use numpy array for the previous row instead of Python list
    previous = np.arange(n + 1, dtype=np.int32)
    
    # Pre-convert b to integer codes for fast vectorized comparison
    b_codes = np.array([ord(c) for c in b], dtype=np.int32)
    
    for i in range(1, m + 1):
        current = np.empty(n + 1, dtype=np.int32)
        current[0] = i
        
        char_code = ord(a[i - 1])
        
        # Compute substitution costs vectorized for all j
        subst_costs = previous[:-1] + ((char_code != b_codes).astype(np.int32))
        
        # Compute for all positions j >= 1
        # current[j] = min(current[j-1] + 1, previous[j] + 1, subst_costs[j-1])
        for j in range(n):
            current[j + 1] = min(current[j] + 1, previous[j + 1] + 1, subst_costs[j])
        
        previous = current
    
    return int(previous[n])
