"""NumPy-vectorized Levenshtein distance with 2-row space optimization."""

from __future__ import annotations
import numpy as np


def edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a  # distance is symmetric; vectorize over the longer string
    m, n = len(a), len(b)
    if m == 0:
        return n
    if a == b:
        return 0
    b_arr = np.fromiter((ord(c) for c in b), dtype=np.int64, count=n)
    idx = np.arange(n + 1, dtype=np.int64)
    prev = idx.copy()
    vals = np.empty(n + 1, dtype=np.int64)
    for i, ch in enumerate(a, 1):
        sub = (b_arr != ord(ch))
        # best of insertion/substitution, independent of current row
        np.minimum(prev[1:] + 1, prev[:-1] + sub, out=vals[1:])
        vals[0] = i
        # left dependency: curr[j] = min_k<=j (vals[k] + j - k)
        vals -= idx
        np.minimum.accumulate(vals, out=vals)
        vals += idx
        prev, vals = vals, prev
    return int(prev[-1])
