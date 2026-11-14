from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    allowed_set = set(allowed)
    return list(filter(lambda x: x in allowed_set, values))
