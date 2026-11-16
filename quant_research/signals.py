"""Transparent, deterministic comparison signals for the first research slice."""

from __future__ import annotations

from typing import Mapping


def trailing_momentum(
    history: Mapping[str, tuple[float, ...]], *, lookback: int = 5,
) -> dict[str, float]:
    """Rank each asset by its already-observed trailing close-price return."""
    if type(lookback) is not int or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    return {
        symbol: closes[-1] / closes[-lookback - 1] - 1.0
        if len(closes) > lookback else 0.0
        for symbol, closes in history.items()
    }
