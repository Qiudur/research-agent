from __future__ import annotations

def string_build(parts: list[str]) -> str:
    result = ""
    for part in parts:
        result = "".join((result, part))
    return result
