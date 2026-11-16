"""Strict, provenance-checked daily prices for the quant research demo.

The first version deliberately accepts only a complete fixed universe. It does
not infer a changing historical universe from today's surviving symbols.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


_CSV_COLUMNS = ["date", "symbol", "close"]
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class PricePanel:
    """Immutable complete date-by-symbol price panel, in deterministic order."""

    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    closes: Mapping[str, tuple[float, ...]]
    csv_sha256: str

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        symbols = tuple(self.symbols)
        if not dates or not symbols or dates != tuple(sorted(set(dates))):
            raise ValueError("PricePanel needs unique, increasing dates and symbols")
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("PricePanel symbols must be unique and sorted")
        if set(self.closes) != set(symbols):
            raise ValueError("PricePanel closes must include exactly the listed symbols")
        closes = {symbol: tuple(self.closes[symbol]) for symbol in symbols}
        if any(len(values) != len(dates) or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 for value in values)
                for values in closes.values()):
            raise ValueError("PricePanel prices must be finite, positive and complete")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "closes", MappingProxyType(closes))


def load_dataset(manifest_path: Path) -> PricePanel:
    """Load a daily CSV only when its declared provenance and digest match.

    Required manifest fields: schema, csv, csv_sha256, source, license,
    price_basis, universe_policy. The license is a source declaration, not an
    independent verification of downstream redistribution rights.
    """
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "quant_dataset/1":
        raise ValueError("Expected quant_dataset/1 manifest")
    for field in ("source", "license", "price_basis"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Dataset manifest requires nonempty {field}")
    if manifest.get("universe_policy") != "fixed_demo":
        raise ValueError("Only fixed_demo universe_policy is supported")
    csv_name = manifest.get("csv")
    if not isinstance(csv_name, str) or not csv_name or Path(csv_name).is_absolute():
        raise ValueError("Dataset csv must be a relative path")
    root = manifest_path.parent
    csv_path = (root / csv_name).resolve(strict=True)
    if not csv_path.is_relative_to(root) or not csv_path.is_file():
        raise ValueError("Dataset csv must stay within its manifest directory")
    expected = manifest.get("csv_sha256")
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise ValueError("Dataset manifest requires a SHA-256 digest")
    raw = csv_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected.lower():
        raise ValueError("Dataset CSV SHA-256 mismatch")
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, None) != _CSV_COLUMNS:
            raise ValueError("Dataset CSV columns must be date,symbol,close")
        by_date: dict[date, dict[str, float]] = {}
        previous: date | None = None
        for line_number, row in enumerate(reader, 2):
            if len(row) != 3:
                raise ValueError(f"Dataset CSV row {line_number} must have three values")
            date_text, symbol, close_text = row
            if not _DATE.fullmatch(date_text):
                raise ValueError(f"Dataset CSV row {line_number} has invalid date")
            try:
                day = date.fromisoformat(date_text)
                close = float(close_text)
            except ValueError as exc:
                raise ValueError(f"Dataset CSV row {line_number} has invalid date or close") from exc
            if previous is not None and day < previous:
                raise ValueError("Dataset CSV dates must be in increasing order")
            if not symbol or symbol != symbol.strip() or not math.isfinite(close) or close <= 0:
                raise ValueError(f"Dataset CSV row {line_number} has invalid symbol or close")
            day_prices = by_date.setdefault(day, {})
            if symbol in day_prices:
                raise ValueError(f"Dataset CSV row {line_number} duplicates a date/symbol")
            day_prices[symbol] = close
            previous = day
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Dataset CSV is not valid UTF-8 CSV") from exc
    if not by_date:
        raise ValueError("Dataset CSV has no price rows")
    dates = tuple(by_date)
    symbols = tuple(sorted(by_date[dates[0]]))
    if any(set(prices) != set(symbols) for prices in by_date.values()):
        raise ValueError("Dataset CSV must contain the same symbols on every date")
    closes = {symbol: tuple(by_date[day][symbol] for day in dates) for symbol in symbols}
    return PricePanel(dates, symbols, closes, digest)
