"""Space-optimized edit distance."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    
    # Handle empty strings directly
    if n == 0:
        return m
    if m == 0:
        return n
    
    # Ensure b is shorter for fewer inner loop iterations
    if n < m:
        a, b = b, a
        n, m = m, n
    
    # Previous row starts as distances from empty string to b[:j]
    prev = list(range(m + 1))
    
    for i in range(n):
        # New row: deleting all characters from a[:i+1]
        curr = [i + 1] + [0] * m
        char_a = a[i]
        for j in range(m):
            cost = 0 if char_a == b[j] else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    
    return prev[m]
