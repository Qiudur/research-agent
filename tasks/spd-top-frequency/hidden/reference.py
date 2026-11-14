from __future__ import annotations

from collections import Counter


def top_frequency(values: list[int], k: int) -> list[tuple[int, int]]:
    return sorted(Counter(values).items(), key=lambda pair: (-pair[1], pair[0]))[:k]
