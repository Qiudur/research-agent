from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    allowed_set = set(allowed)
    contains = allowed_set.__contains__
    return [value for value in values if contains(value)]
