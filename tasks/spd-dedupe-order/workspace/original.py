from __future__ import annotations

def dedupe_order(values: list[int]) -> list[int]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
