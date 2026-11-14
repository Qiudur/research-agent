from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    allowed_frozenset = frozenset(allowed)
    return [value for value in values if value in allowed_frozenset]
