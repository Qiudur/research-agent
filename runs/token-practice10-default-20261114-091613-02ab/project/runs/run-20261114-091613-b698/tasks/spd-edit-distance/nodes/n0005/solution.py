"""Optimized two-dimensional DP with reduced memory overhead."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    # Handle edge case where one or both strings are empty
    if not a:
        return len(b)
    if not b:
        return len(a)
    
    # Ensure b is longer for efficiency (less space needed)
    if len(a) > len(b):
        a, b = b, a
    
    n, m = len(a), len(b)
    
    # Single-row DP to reduce memory allocation overhead
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            # Inline min calculation for speed
            sub_cost = (ai != b[j - 1])
            del_cost = curr[j - 1] + 1
            ins_cost = prev[j] + 1
            sub_cost = prev[j - 1] + sub_cost
            
            # Manual min for three values
            if del_cost <= ins_cost:
                curr[j] = del_cost if del_cost <= sub_cost else sub_cost
            else:
                curr[j] = ins_cost if ins_cost <= sub_cost else sub_cost
        # Swap arrays instead of reallocating
        prev, curr = curr, prev
    
    return prev[m]
