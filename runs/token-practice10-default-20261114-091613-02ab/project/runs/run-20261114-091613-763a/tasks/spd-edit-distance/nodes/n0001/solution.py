"""Clear two-dimensional dynamic-programming implementation."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    
    previous = list(range(len(b) + 1))
    matrix = [previous]
    for i, left in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, right in enumerate(b, 1):
            row[j] = min(row[j - 1] + 1, previous[j] + 1,
                         previous[j - 1] + (left != right))
        matrix.append(row)
        previous = row
    return matrix[-1][-1]
