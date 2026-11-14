"""Optimized trailing-window implementation using sliding window sum."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute rolling mean with O(1) update per position.
    
    Each result averages the current value and up to window-1 preceding values.
    Early positions use only available values.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    
    result = []
    current_sum = 0.0
    
    for index, val in enumerate(values):
        # Add new value to running sum
        current_sum += val
        
        # Remove value that falls out of window (if any)
        if index >= window:
            current_sum -= values[index - window]
        
        # Effective window size for this position
        effective_window = min(index + 1, window)
        
        # Append mean
        result.append(current_sum / effective_window)
    
    return result
