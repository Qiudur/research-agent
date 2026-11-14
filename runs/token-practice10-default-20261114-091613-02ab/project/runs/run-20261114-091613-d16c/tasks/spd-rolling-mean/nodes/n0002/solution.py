"""Optimized trailing-window implementation using NumPy vectorized operations."""

from __future__ import annotations


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute rolling mean with NumPy vectorized operations.
    
    Each result averages the current value and up to window-1 preceding values.
    Early positions use only available values.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    
    n = len(values)
    if n == 0:
        return []
    
    import numpy as np
    
    arr = np.asarray(values, dtype=np.float64)
    # Pad cumsum with 0 at start for easier window calculations
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    
    # For each position i (0-indexed), we want sum from max(0, i-window+1) to i inclusive
    # That's cumsum[i+1] - cumsum[max(0, i-window+1)]
    
    # End indices (i+1) for cumsum access
    end_indices = np.arange(1, n + 1)
    
    # Start indices for window beginning in cumsum
    start_indices = np.maximum(end_indices - window, 0)
    
    # Compute window sums via cumulative sum differencing
    window_sums = cumsum[end_indices] - cumsum[start_indices]
    
    # Effective window sizes (cannot exceed actual available elements)
    effective_window_sizes = np.minimum(np.arange(1, n + 1), window).astype(np.float64)
    
    return (window_sums / effective_window_sizes).tolist()
