"""Optimized rolling mean using numpy cumsum approach."""

from __future__ import annotations
import numpy as np


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    
    n = len(values)
    if n == 0:
        return []
    
    # Convert to numpy array for efficient cumsum
    arr = np.array(values, dtype=np.float64)
    
    # Compute cumulative sums with leading zero
    cumsum = np.zeros(n + 1, dtype=np.float64)
    cumsum[1:] = np.cumsum(arr)
    
    # Pre-compute reciprocal for full window to avoid repeated division
    window_reciprocal = 1.0 / float(window)
    
    # Build result using cumsum differences
    result = []
    
    # Prefix region: window not yet full, count varies
    for i in range(min(n, window)):
        total = cumsum[i + 1]
        count = i + 1
        result.append(total.item() / count)
    
    # Full window region: count always equals window, use reciprocal
    for i in range(window, n):
        start = i - window + 1
        total = cumsum[i + 1] - cumsum[start]
        result.append(total.item() * window_reciprocal)
    
    return result
