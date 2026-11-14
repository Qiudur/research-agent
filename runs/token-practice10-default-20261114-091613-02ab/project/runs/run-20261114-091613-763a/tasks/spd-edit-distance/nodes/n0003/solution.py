"""Clear two-dimensional dynamic-programming implementation."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        row = [i] + [0] * len(b)
        diag = previous[0]  # Cache diagonal value (previous[j-1])
        for j, right in enumerate(b, 1):
            up = previous[j]       # Cached top value
            left_val = row[j - 1]  # Cached left value
            row[j] = min(left_val + 1, up + 1, diag + (left != right))
            diag = up              # Update diagonal for next iteration
        previous = row
    return previous[-1]
