from __future__ import annotations

from math import fsum


def already_vectorized(values: list[float]) -> float:
    return fsum(values)
