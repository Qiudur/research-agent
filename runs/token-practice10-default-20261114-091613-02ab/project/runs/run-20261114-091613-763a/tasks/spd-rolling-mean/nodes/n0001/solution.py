"""Optimized trailing-window rolling mean implementation."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    
    result = []
    current_sum = 0.0
    
    for i in range(len(values)):
        # Add new value to running sum
        current_sum += values[i]
        
        # Remove oldest value if window is full
        if i >= window:
            current_sum -= values[i - window]
        
        # Actual window size at this position (smaller at start)
        actual_size = min(i + 1, window)
        result.append(current_sum / actual_size)
    
    return result
