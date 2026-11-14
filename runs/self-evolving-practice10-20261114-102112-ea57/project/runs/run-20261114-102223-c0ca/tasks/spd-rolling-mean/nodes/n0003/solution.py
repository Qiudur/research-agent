"""Optimized rolling mean using sliding window with running sum maintenance."""

from __future__ import annotations
from itertools import accumulate


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Compute trailing rolling mean with efficient cumulative sum approach."""
    if window <= 0:
        raise ValueError("window must be positive")
    
    if not values:
        return []
    
    # Use accumulate for O(1) window sum queries via cumulative sums
    # Then use list comprehension to compute means
    cumsum = [0.0] + list(accumulate(values))
    n = len(values)
    
    return [
        (cumsum[i + 1] - cumsum[max(0, i + 1 - window)]) / min(i + 1, window)
        for i in range(n)
    ]
