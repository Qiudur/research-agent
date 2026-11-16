"""Small, deterministic synthetic price panel for key-free demonstrations."""

from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path


def write_demo_dataset(directory: Path) -> Path:
    """Create a visibly synthetic fixed-universe dataset and its manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    csv_path = directory / "prices.csv"
    symbols = ("SYN-A", "SYN-B", "SYN-C", "SYN-D")
    dates: list[date] = []
    day = date(2020, 1, 1)
    while len(dates) < 120:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("date", "symbol", "close"))
        for index, trading_day in enumerate(dates):
            for offset, symbol in enumerate(symbols):
                # Deliberately artificial patterns; these have no market meaning.
                log_price = (math.log(80 + 15 * offset)
                             + (0.0004 + 0.0001 * offset) * index
                             + 0.03 * math.sin(index / (8 + offset) + offset)
                             + 0.008 * math.sin(index / 3 + offset * 2))
                writer.writerow((trading_day.isoformat(), symbol, f"{math.exp(log_price):.10f}"))
    manifest = {
        "schema": "quant_dataset/1", "csv": csv_path.name,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source": "deterministic synthetic generator in quant_research.demo_data",
        "license": "generated in this repository; no third-party market data",
        "price_basis": "synthetic unadjusted closes; no corporate actions",
        "universe_policy": "fixed_demo",
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
