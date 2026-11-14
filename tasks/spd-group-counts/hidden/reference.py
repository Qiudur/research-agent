from __future__ import annotations

def group_counts(keys: list[str]) -> dict[str, int]:
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts
