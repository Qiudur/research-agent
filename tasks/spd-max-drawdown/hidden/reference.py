"""Trusted linear-time prefix-maximum implementation."""

from __future__ import annotations


def max_drawdown(prices: list[float]) -> float:
    peak = 0.0
    largest = 0.0
    for price in prices:
        if price > peak:
            peak = price
        else:
            largest = max(largest, (peak - price) / peak)
    return largest
