"""Optimized edit distance with two-row buffer swapping."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    
    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Pre-allocate two fixed-size buffers
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i, left in enumerate(a, 1):
        curr_row[0] = i
        for j, right in enumerate(b, 1):
            # Compute three candidate costs
            ins_cost = curr_row[j - 1] + 1
            del_cost = prev_row[j] + 1
            sub_cost = prev_row[j - 1] + (left != right)
            
            # Inline min comparison to avoid function call overhead
            if ins_cost <= del_cost and ins_cost <= sub_cost:
                curr_row[j] = ins_cost
            elif del_cost <= sub_cost:
                curr_row[j] = del_cost
            else:
                curr_row[j] = sub_cost
        
        # Swap references instead of allocating new lists
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[-1]
