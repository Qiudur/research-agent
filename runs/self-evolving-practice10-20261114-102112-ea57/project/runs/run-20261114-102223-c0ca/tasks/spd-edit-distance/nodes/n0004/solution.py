"""Optimized edit distance with two-row buffer and range-based indexing."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    
    # Two-row DP with direct indexing (no enumerate tuples, no full matrix)
    previous = list(range(m + 1))
    current = [0] * (m + 1)
    
    for i in range(1, n + 1):
        current[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(current[j - 1] + 1,
                             previous[j] + 1,
                             previous[j - 1] + cost)
        # Swap row references for next iteration
        previous, current = current, previous
    
    return previous[m]
