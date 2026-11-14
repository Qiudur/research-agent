from __future__ import annotations

def weighted_groups(keys: list[str], values: list[float]) -> dict[str, float]:
    totals = {}
    for key in set(keys):
        total = 0.0
        for index, other in enumerate(keys):
            if other == key:
                total += values[index]
        totals[key] = total
    return totals
