"""Download a pinned, key-free historical ETF snapshot for the research case.

This deliberately fixes the universe and window before examining performance.
Downloaded Yahoo-derived market data belongs in ignored local run directories.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .data import load_dataset


SYMBOLS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD")
START_DATE = date(2008, 1, 1)
END_DATE = date(2019, 12, 31)
DATASET = "jacksoncrow/stock-market-dataset"
DATASET_VERSION = 2
SOURCE_URL = f"https://www.kaggle.com/datasets/{DATASET}/versions/{DATASET_VERSION}"
METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"
MAX_DOWNLOAD_BYTES = 5_000_000


def _download(url: str) -> bytes:
    """Bound response size, time and retries; never require credentials."""
    request = Request(url, headers={"User-Agent": "quant-research-reproducible-case/1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise ValueError("ETF source response exceeds the 5 MB per-file limit")
            if not raw:
                raise ValueError("ETF source returned an empty response")
            return raw
        except (TimeoutError, URLError) as exc:
            # Permanent HTTP errors should not consume repeated requests.
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _source_prices(raw: bytes, symbol: str) -> dict[date, float]:
    """Select source Adj Close, rejecting malformed rows inside the window."""
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if reader.fieldnames != ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            raise ValueError(f"{symbol}: unexpected source CSV columns")
        prices: dict[date, float] = {}
        seen: set[date] = set()
        previous: date | None = None
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{symbol}: malformed source CSV row")
            text = row["Date"]
            day = date.fromisoformat(text)
            if text != day.isoformat() or day in seen:
                raise ValueError(f"{symbol}: invalid or duplicate source date {text}")
            if previous is not None and day < previous:
                raise ValueError(f"{symbol}: source dates must increase")
            seen.add(day)
            previous = day
            if START_DATE <= day <= END_DATE:
                value = float(row["Adj Close"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{symbol}: adjusted close must be finite and positive")
                prices[day] = value
    except (UnicodeError, csv.Error, TypeError) as exc:
        raise ValueError(f"{symbol}: invalid source CSV") from exc
    if not prices:
        raise ValueError(f"{symbol}: no observed prices in the fixed research window")
    return prices


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch_etf_dataset(output_dir: Path) -> Path:
    """Download the fixed v2 snapshot and return a verified quant_dataset/1 manifest.

    Refuses an existing directory. Failure leaves raw evidence for diagnosis but
    never produces a valid manifest or substitutes generated market prices.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    metadata_raw = _download(METADATA_URL)
    (raw_dir / "kaggle-metadata.json").write_bytes(metadata_raw)
    metadata = json.loads(metadata_raw)
    if not isinstance(metadata, dict) or metadata.get("ref") != DATASET:
        raise ValueError("Kaggle metadata does not identify the pinned dataset")
    if metadata.get("currentVersionNumber") != DATASET_VERSION:
        raise ValueError("Source metadata version changed; review the pinned source before proceeding")
    if metadata.get("licenseName") != "CC0: Public Domain":
        raise ValueError("Source license declaration changed; review before proceeding")

    source_files: list[dict[str, object]] = []
    prices: dict[str, dict[date, float]] = {}
    for symbol in SYMBOLS:
        url = (f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}/"
               f"{quote(f'etfs/{symbol}.csv', safe='')}?datasetVersionNumber={DATASET_VERSION}")
        raw = _download(url)
        local_name = f"raw/{symbol}.csv"
        (output_dir / local_name).write_bytes(raw)
        prices[symbol] = _source_prices(raw, symbol)
        source_files.append({
            "symbol": symbol, "path": local_name, "source_file": f"etfs/{symbol}.csv",
            "url": url, "sha256": _sha256(raw), "bytes": len(raw),
            "in_window_observed_dates": len(prices[symbol]),
        })

    dates_by_symbol = {symbol: set(values) for symbol, values in prices.items()}
    observed_dates = set.union(*dates_by_symbol.values())
    dates = sorted(set.intersection(*dates_by_symbol.values()))
    if not dates:
        raise ValueError("The fixed ETF universe has no shared observed dates")
    kept = set(dates)
    dropped = sorted(observed_dates - kept)
    csv_path = output_dir / "prices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("date", "symbol", "close"))
        for day in dates:
            for symbol in sorted(SYMBOLS):
                writer.writerow((day.isoformat(), symbol, repr(prices[symbol][day])))

    manifest = {
        "schema": "quant_dataset/1", "csv": csv_path.name,
        "csv_sha256": _sha256(csv_path.read_bytes()),
        "source": f"Kaggle {DATASET} version {DATASET_VERSION}; uploader states Yahoo Finance via yfinance",
        "license": ("Kaggle uploader declares CC0: Public Domain; upstream Yahoo Finance rights "
                    "and redistribution permission are not independently verified"),
        "price_basis": ("source Adj Close: split- and dividend-adjusted, as described by uploader; "
                        "snapshot as of April 2020, not point-in-time corporate-action data"),
        "universe_policy": "fixed_demo", "data_kind": "market",
        "requested_universe": list(SYMBOLS),
        "requested_start_date": START_DATE.isoformat(),
        "requested_end_date": END_DATE.isoformat(),
        "actual_start_date": dates[0].isoformat(), "actual_end_date": dates[-1].isoformat(),
        "date_count": len(dates), "row_count": len(dates) * len(SYMBOLS),
        "source_snapshot": {
            "dataset": DATASET, "version": DATASET_VERSION, "url": SOURCE_URL,
            "last_updated": metadata.get("lastUpdated"),
            "metadata_url": METADATA_URL, "metadata_path": "raw/kaggle-metadata.json",
            "metadata_sha256": _sha256(metadata_raw),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": source_files,
        },
        "missing_date_policy": {
            "method": "intersection of observed dates across the fixed universe; no forward/backfill",
            "calendar_validation": "observed source dates only; no independent exchange calendar check",
            "observed_union_date_count": len(observed_dates),
            "discarded_date_count": len(dropped),
            "discarded_dates": [day.isoformat() for day in dropped],
            "missing_dates_by_symbol": {
                symbol: [day.isoformat() for day in sorted(observed_dates - days)]
                for symbol, days in dates_by_symbol.items()
            },
            "discarded_observations_by_symbol": {
                symbol: len(days - kept) for symbol, days in dates_by_symbol.items()
            },
        },
        "limitations": [
            "Fixed, retrospectively selected surviving ETF universe; not survivorship-bias-free.",
            "Adjusted close is a later snapshot and is not point-in-time corporate-action data.",
            "Source adjustments and completeness are declared by the uploader, not independently reconstructed.",
            "Keep raw and normalized market data local; uploader license does not establish upstream rights.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        load_dataset(manifest_path)
    except Exception:
        manifest_path.unlink()
        raise
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New local directory, normally under ignored runs/")
    args = parser.parse_args()
    path = fetch_etf_dataset(args.output_dir)
    print(path.resolve())


if __name__ == "__main__":
    main()
