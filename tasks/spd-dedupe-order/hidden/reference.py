from __future__ import annotations

def dedupe_order(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
