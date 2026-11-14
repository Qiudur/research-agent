from __future__ import annotations

def group_counts(keys: list[str]) -> dict[str, int]:
    counts = {}
    for key in set(keys):
        count = 0
        for other in keys:
            if other == key:
                count += 1
        counts[key] = count
    return counts
