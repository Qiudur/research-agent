from __future__ import annotations

def top_frequency(values: list[int], k: int) -> list[tuple[int, int]]:
    counts = {}
    for value in set(values):
        count = 0
        for other in values:
            if other == value:
                count += 1
        counts[value] = count
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
