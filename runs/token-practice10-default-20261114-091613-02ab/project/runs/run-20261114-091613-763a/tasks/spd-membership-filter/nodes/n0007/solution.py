from __future__ import annotations

def membership_filter(values: list[int], allowed: list[int]) -> list[int]:
    # Hybrid approach: use set for large allowed lists, direct check for small ones
    THRESHOLD = 50
    
    if len(allowed) >= THRESHOLD:
        # Large allowed list benefits from hash-based O(1) lookups
        allowed_set = set(allowed)
        return [value for value in values if value in allowed_set]
    else:
        # Small allowed list: avoid set conversion overhead
        return [value for value in values if value in allowed]
