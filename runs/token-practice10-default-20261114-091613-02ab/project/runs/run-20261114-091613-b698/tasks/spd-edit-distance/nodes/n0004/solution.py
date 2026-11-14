"""Two-row rolling window optimization for edit distance."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance using O(min(n,m)) space."""
    # Use shorter string for columns to minimize space usage
    if len(a) < len(b):
        a, b = b, a
    
    # Initialize previous row with deletion costs
    previous = list(range(len(b) + 1))
    
    for i, left in enumerate(a, 1):
        # Start new row with deletion cost from previous position
        current = [i]
        for j, right in enumerate(b, 1):
            # Minimum of delete, insert, or substitute
            current.append(
                min(current[-1] + 1,                    # insert
                    previous[j] + 1,                     # delete
                    previous[j - 1] + (left != right))   # substitute
            )
        previous = current
    
    return previous[-1]
