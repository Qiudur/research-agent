"""O(n) maximum-drawdown implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    
    largest = 0.0
    peak = prices[0]
    
    for i in range(1, len(prices)):
        price = prices[i]
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            if drawdown > largest:
                largest = drawdown
    
    return largest
