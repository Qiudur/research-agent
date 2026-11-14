from __future__ import annotations

try:
    from Levenshtein import distance as _levenshtein_dist
except ImportError:
    _levenshtein_dist = None


def edit_distance(a: str, b: str) -> int:
    if _levenshtein_dist is not None:
        return _levenshtein_dist(a, b)
    
    # Fallback: original DP implementation
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1,
                              previous[j] + 1,
                              previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]
