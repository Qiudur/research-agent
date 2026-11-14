"""Optimized trailing-window implementation using sliding window sum."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    
    result = []
    result_append = result.append
    window_sum = 0.0
    
    for index, value in enumerate(values):
        # Add new value to window sum
        window_sum += value
        
        # Remove oldest value when window is full-sized
        if index >= window:
            window_sum -= values[index - window]
        
        # Actual window size (grows during initial partial windows)
        current_window_size = min(index + 1, window)
        
        result_append(window_sum / current_window_size)
    
    return result
