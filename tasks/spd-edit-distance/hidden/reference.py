"""Trusted bit-parallel Myers edit distance for Python big integers."""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    masks: dict[str, int] = {}
    for index, letter in enumerate(a):
        masks[letter] = masks.get(letter, 0) | (1 << index)
    positive = ~0
    negative = 0
    distance = len(a)
    final_bit = 1 << (len(a) - 1)
    for letter in b:
        matches = masks.get(letter, 0)
        vertical = matches | negative
        horizontal = (((matches & positive) + positive) ^ positive) | matches
        new_positive = negative | ~(horizontal | positive)
        new_negative = positive & horizontal
        if new_positive & final_bit:
            distance += 1
        elif new_negative & final_bit:
            distance -= 1
        new_positive = (new_positive << 1) | 1
        new_negative <<= 1
        positive = new_negative | ~(vertical | new_positive)
        negative = new_positive & vertical
    return distance
