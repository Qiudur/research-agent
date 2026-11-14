from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    """Keep values that appear in allowed, preserving order and duplicates from values."""
    allowed_set = set(allowed)
    return [value for value in values if value in allowed_set]
