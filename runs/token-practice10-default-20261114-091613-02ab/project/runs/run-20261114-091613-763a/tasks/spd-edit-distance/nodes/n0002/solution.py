"""Clear two-dimensional dynamic-programming implementation."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i] + [0] * len(b)
        for j, right in enumerate(b, 1):
            row[j] = min(row[j - 1] + 1, previous[j] + 1,
                         previous[j - 1] + (left != right))
        previous = row
    return previous[-1]
