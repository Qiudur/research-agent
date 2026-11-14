from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    """Filter values to keep only those present in allowed, preserving order and duplicates."""
    allowed_set = set(allowed)
    return [value for value in values if value in allowed_set]
