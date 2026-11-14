"""Clear two-dimensional dynamic-programming implementation."""

from __future__ import annotations
import array


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance using array storage.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Edit distance between a and b (unit cost for insert/delete/substitute)
    """
    n = len(b)
    # Use array('i') instead of list for more compact integer storage
    previous = array.array('i', range(n + 1))
    
    for i, left in enumerate(a, 1):
        # Build current row using array for potential allocation benefits
        current = array.array('i', [i] + [0] * n)
        for j, right in enumerate(b, 1):
            # min(insert, delete, substitute)
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left != right)
            )
        previous = current
    
    return previous[-1]
