"""Exact raw-token accounting checks for the internal TOKENS protocol."""

from __future__ import annotations

import math


BUCKETS = ("input", "cached_input", "output", "reasoning")


def is_exact_token_accounting(prices: dict, model: str) -> bool:
    """True only when one recorded unit equals one token in every bucket."""
    if not isinstance(prices, dict) or prices.get("currency") != "TOKENS":
        return False
    if type(prices.get("unit_tokens")) is not int or prices["unit_tokens"] != 1:
        return False
    models = prices.get("models")
    rates = models.get(model) if isinstance(models, dict) else None
    if not isinstance(rates, dict):
        return False
    return all(type(rates.get(bucket)) in (int, float)
               and math.isfinite(rates[bucket]) and rates[bucket] == 1
               for bucket in BUCKETS)


def require_exact_token_accounting(prices: dict, model: str) -> None:
    if isinstance(prices, dict) and prices.get("currency") == "TOKENS" and not is_exact_token_accounting(prices, model):
        raise ValueError("TOKENS accounting requires unit_tokens=1 and all four model rates=1")
