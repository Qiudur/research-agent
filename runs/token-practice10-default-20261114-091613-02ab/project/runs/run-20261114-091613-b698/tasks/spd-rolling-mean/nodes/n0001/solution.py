"""Optimized sliding-window rolling mean implementation."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute rolling mean using O(n) sliding window with accumulator.

    Args:
        values: List of finite float values
        window: Positive integer window size

    Returns:
        List of rolling averages, where each result is the mean of
        current and up to window-1 preceding values
    """
    if window <= 0:
        raise ValueError("window must be positive")

    result = []
    current_sum = 0.0
    left = 0

    for right, value in enumerate(values):
        current_sum += value

        # Maintain window size constraint
        if right - left + 1 > window:
            current_sum -= values[left]
            left += 1

        window_len = right - left + 1
        result.append(current_sum / window_len)

    return result
