"""Optimized trailing-window rolling mean implementation."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    
    n = len(values)
    result = []
    append_result = result.append
    current_sum = 0.0
    
    for i in range(n):
        # Add new value to running sum
        current_sum += values[i]
        
        # Remove oldest value if window is full
        if i >= window:
            current_sum -= values[i - window]
        
        # Actual window size at this position (smaller at start)
        actual_size = i + 1 if i < window else window
        append_result(current_sum / actual_size)
    
    return result
