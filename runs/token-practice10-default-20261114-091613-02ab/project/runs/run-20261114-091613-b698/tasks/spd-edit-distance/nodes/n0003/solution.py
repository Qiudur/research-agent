"""Clear two-dimensional dynamic-programming implementation."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    # Early termination: find longest common prefix
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    
    # Early termination: find longest common suffix (non-overlapping with prefix)
    end_a = len(a) - 1
    end_b = len(b) - 1
    skipped_suffix = 0
    while end_a >= start and end_b >= 0 and a[end_a] == b[end_b]:
        skipped_suffix += 1
        end_a -= 1
        end_b -= 1
    
    # Check if remaining substrings would give trivial result
    remaining_a = a[start:end_a + 1]
    remaining_b = b[start:end_b + 1]
    
    # If remaining parts are empty, distances come purely from length differences
    if not remaining_a or not remaining_b:
        return abs(len(a) - len(b))
    
    # Run DP on remaining portions only
    prev_len = len(remaining_b) + 1
    previous = list(range(prev_len))
    
    for i, left in enumerate(remaining_a, 1):
        row = [i] + [0] * len(remaining_b)
        for j, right in enumerate(remaining_b, 1):
            row[j] = min(row[j - 1] + 1, previous[j] + 1,
                         previous[j - 1] + (left != right))
        previous = row
    
    return previous[-1]
