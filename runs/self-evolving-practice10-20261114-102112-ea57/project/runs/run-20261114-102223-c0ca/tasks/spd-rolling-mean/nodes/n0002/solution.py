"""Optimized rolling mean using sliding window with running sum maintenance."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    
    n = len(values)
    if n == 0:
        return []
    
    result = []
    running_sum = 0.0
    
    for i in range(n):
        # Add current value to running sum
        running_sum += values[i]
        
        # If window is full, remove the oldest value that falls out
        if i >= window:
            running_sum -= values[i - window]
        
        # Count elements in current effective window
        count = min(i + 1, window)
        
        # Compute mean
        result.append(running_sum / count)
    
    return result
