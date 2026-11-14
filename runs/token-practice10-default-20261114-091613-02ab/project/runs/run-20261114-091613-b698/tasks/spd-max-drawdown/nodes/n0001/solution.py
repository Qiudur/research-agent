"""Optimized linear-time maximum-drawdown implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    """Calculate maximum drawdown in O(n) time.
    
    Tracks running peak and computes drawdown in single pass.
    Returns largest peak-to-trough fraction where trough comes after peak.
    """
    if not prices:
        return 0.0
    
    peak = prices[0]
    largest = 0.0
    
    for price in prices:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            largest = max(largest, drawdown)
    
    return largest
