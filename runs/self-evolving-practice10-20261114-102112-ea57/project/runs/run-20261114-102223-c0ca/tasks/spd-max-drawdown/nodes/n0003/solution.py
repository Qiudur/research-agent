"""Optimized O(n) single-pass maximum-drawdown implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    n = len(prices)
    if n <= 1:
        return 0.0
    
    largest = 0.0
    peak = prices[0]
    _max = max  # Cache builtin to local variable
    
    # Use range indexing to avoid slice copying
    for i in range(1, n):
        price = prices[i]
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            if drawdown > largest:
                largest = drawdown
    
    return largest
