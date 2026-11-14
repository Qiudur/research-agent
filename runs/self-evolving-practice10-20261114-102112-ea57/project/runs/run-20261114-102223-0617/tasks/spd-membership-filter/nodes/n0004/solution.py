from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    if not allowed:
        return []
    elif len(allowed) == 1:
        target = allowed[0]
        return [value for value in values if value == target]
    else:
        return [value for value in values if value in allowed]
