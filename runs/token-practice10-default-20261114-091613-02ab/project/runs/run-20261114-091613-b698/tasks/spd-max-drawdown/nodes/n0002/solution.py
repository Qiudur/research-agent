"""Optimized linear-time maximum-drawdown implementation."""

from __future__ import annotations
import numpy as np


def max_drawdown(prices: list[float]) -> float:
    """Calculate maximum drawdown in O(n) time.
    
    Tracks running peak and computes drawdown in single pass.
    Returns largest peak-to-trough fraction where trough comes after peak.
    """
    if not prices:
        return 0.0
    
    # Convert prices list to numpy array once at start
    prices_arr = np.array(prices)
    
    # Compute running peaks using cumulative maximum (numpy indexing)
    peaks = np.maximum.accumulate(prices_arr)
    
    # Compute all drawdowns at once using numpy operations
    drawdowns = (peaks - prices_arr) / peaks
    
    # Return maximum drawdown
    return float(np.max(drawdowns))
