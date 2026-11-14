"""O(n) single-pass maximum-drawdown implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    """Calculate maximum chronological peak-to-trough fraction in O(n)."""
    if len(prices) < 2:
        return 0.0
    
    max_price = prices[0]
    max_dd = 0.0
    
    for price in prices[1:]:
        if price > max_price:
            max_price = price
        # Drawdown from running peak to current price
        dd = (max_price - price) / max_price
        if dd > max_dd:
            max_dd = dd
    
    return max_dd
