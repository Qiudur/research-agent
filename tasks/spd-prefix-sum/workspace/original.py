from __future__ import annotations

def prefix_sum(values: list[float]) -> list[float]:
    return [sum(values[:index + 1]) for index in range(len(values))]
