"""Optimized trailing-window implementation with preallocated result list."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(values)
    result = [0.0] * n
    for index in range(n):
        start = max(0, index - window + 1)
        current = values[start:index + 1]
        result[index] = sum(current) / len(current)
    return result
