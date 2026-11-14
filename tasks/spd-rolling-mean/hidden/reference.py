"""Trusted linear-time rolling sum implementation."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result
