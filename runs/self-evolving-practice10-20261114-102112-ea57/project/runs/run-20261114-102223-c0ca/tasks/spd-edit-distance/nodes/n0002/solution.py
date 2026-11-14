"""Optimized edit distance with inlined min logic."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    
    # Edge cases
    if m == 0:
        return n
    if n == 0:
        return m
    
    previous = list(range(n + 1))
    for i, left in enumerate(a, 1):
        row = [i]
        for j, right in enumerate(b, 1):
            # Compute three candidate costs
            ins_cost = row[j - 1] + 1
            del_cost = previous[j] + 1
            sub_cost = previous[j - 1] + (left != right)
            
            # Inline min comparison to avoid function call overhead
            if ins_cost <= del_cost and ins_cost <= sub_cost:
                row.append(ins_cost)
            elif del_cost <= sub_cost:
                row.append(del_cost)
            else:
                row.append(sub_cost)
        previous = row
    
    return previous[-1]
