"""O(n) single-pass maximum-drawdown implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    
    largest = 0.0
    peak = prices[0]
    
    for price in prices[1:]:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            if drawdown > largest:
                largest = drawdown
    
    return largest
