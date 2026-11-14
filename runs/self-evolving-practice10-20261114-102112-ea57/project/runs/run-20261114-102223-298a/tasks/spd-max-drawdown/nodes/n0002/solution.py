"""Maximum-drawdown implementation with numpy vectorization."""

from __future__ import annotations

import numpy as np


def max_drawdown(prices: list[float]) -> float:
    """Compute maximum chronological peak-to-trough fraction.
    
    Args:
        prices: List of strictly positive finite float prices.
        
    Returns:
        Maximum drawdown as fraction, or 0.0 for empty/non-decreasing series.
    """
    if len(prices) <= 1:
        return 0.0
    
    arr = np.array(prices)
    
    # Use cummax to get running maximum (peak so far) at each position
    peak_arr = np.maximum.accumulate(arr)
    
    # Calculate drawdown at each position from the current peak
    drawdowns = (peak_arr - arr) / peak_arr
    
    # Return maximum drawdown across all positions
    return float(np.max(drawdowns))
