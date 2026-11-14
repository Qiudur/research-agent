from __future__ import annotations

def weighted_groups(keys: list[str], values: list[float]) -> dict[str, float]:
    totals = {}
    for key, value in zip(keys, values):
        totals[key] = totals.get(key, 0.0) + value
    return totals
