"""Linear-time maximum-drawdown implementation using single-pass peak tracking."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    """Calculate the maximum chronological peak-to-trough percentage.
    
    Uses O(n) single-pass algorithm tracking running peak.
    Returns 0.0 for empty lists or non-decreasing sequences.
    """
    if len(prices) < 2:
        return 0.0
    
    largest = 0.0
    peak = prices[0]
    
    for price in prices:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            if drawdown > largest:
                largest = drawdown
    
    return largest
