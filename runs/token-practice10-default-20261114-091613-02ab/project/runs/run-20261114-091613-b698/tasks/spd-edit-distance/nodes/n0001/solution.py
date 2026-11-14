"""Optimized edit distance with reused row buffers."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    
    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Reusable buffers - only allocate once
    prev = [i for i in range(n + 1)]
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        left = a[i - 1]
        for j in range(1, n + 1):
            # min of: deletion (prev[j]+1), insertion (curr[j-1]+1), substitution
            sub_cost = 0 if left == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub_cost)
        
        # Reuse buffer by copying values (avoid allocation)
        prev[:] = curr
    
    return prev[n]
