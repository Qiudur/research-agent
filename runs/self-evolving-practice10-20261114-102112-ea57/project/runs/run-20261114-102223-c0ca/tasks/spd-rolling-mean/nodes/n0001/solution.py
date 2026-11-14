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
    
    # Build result using cumsum differences
    # For each position i: sum from start to i is cumsum[i+1] - cumsum[start]
    result = []
    for i in range(n):
        start = max(0, i - window + 1)
        total = cumsum[i + 1] - cumsum[start]
        count = i + 1 - start
        result.append(total.item() / count)
    
    return result
