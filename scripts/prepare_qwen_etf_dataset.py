"""Prepare the pre-registered Qwen ETF comparison dataset from the pinned raw snapshot.

The raw Kaggle files are kept under ignored ``runs/``. This command only
normalizes their adjusted-close columns through 2020-04-01; it never downloads
new data, computes a strategy score, or opens the comparison holdout.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import csv

from quant_research.data import load_dataset


SYMBOLS = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD")
START = date(2008, 1, 1)
END = date(2020, 4, 1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source_dir: Path, output_dir: Path, source_manifest: Path) -> Path:
    source_dir = Path(source_dir).expanduser().resolve(strict=True)
    source_manifest = Path(source_manifest).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory; evidence is never overwritten")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "quant_dataset/1" or source.get("source_snapshot", {}).get("version") != 2:
        raise ValueError("Expected the pinned Kaggle v2 ETF manifest")
    prices: dict[str, dict[date, float]] = {}
    source_files = []
    for symbol in SYMBOLS:
        path = source_dir / f"{symbol}.csv"
        raw = path.read_bytes()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if reader.fieldnames != expected:
                raise ValueError(f"{symbol}: unexpected raw CSV columns")
            values: dict[date, float] = {}
            for row in reader:
                day = date.fromisoformat(row["Date"])
                if START <= day <= END:
                    value = float(row["Adj Close"])
                    if value <= 0:
                        raise ValueError(f"{symbol}: adjusted close must be positive")
                    if day in values:
                        raise ValueError(f"{symbol}: duplicate date {day}")
                    values[day] = value
        if not values:
            raise ValueError(f"{symbol}: no prices in the extension window")
        prices[symbol] = values
        source_files.append({
            "symbol": symbol, "source_file": f"etfs/{symbol}.csv",
            "source_sha256": _sha(path), "source_path": f"raw/{symbol}.csv",
            "observed_dates": len(values),
        })

    dates = sorted(set.intersection(*(set(values) for values in prices.values())))
    if not dates or dates[-1] != END:
        raise ValueError("Pinned raw snapshot does not cover the registered end date")
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "prices.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("date", "symbol", "close"))
        for day in dates:
            for symbol in sorted(SYMBOLS):
                writer.writerow((day.isoformat(), symbol, repr(prices[symbol][day])))

    policy = dict(source.get("missing_date_policy", {}))
    policy.update({
        "method": "intersection of observed dates across the fixed universe; no forward/backfill",
        "calendar_validation": "observed source dates only; no independent exchange calendar check",
        "observed_union_date_count": len(set.union(*(set(values) for values in prices.values()))),
        "discarded_date_count": 0,
        "discarded_dates": [],
        "missing_dates_by_symbol": {symbol: [] for symbol in SYMBOLS},
        "discarded_observations_by_symbol": {symbol: 0 for symbol in SYMBOLS},
    })
    snapshot = dict(source["source_snapshot"])
    snapshot["source_files"] = source_files
    manifest = {
        **{key: source[key] for key in ("schema", "source", "license", "price_basis", "universe_policy", "data_kind", "limitations")},
        "csv": "prices.csv", "csv_sha256": _sha(csv_path),
        "requested_universe": list(SYMBOLS), "requested_start_date": START.isoformat(),
        "requested_end_date": END.isoformat(), "actual_start_date": dates[0].isoformat(),
        "actual_end_date": dates[-1].isoformat(), "date_count": len(dates),
        "row_count": len(dates) * len(SYMBOLS), "source_snapshot": snapshot,
        "missing_date_policy": policy,
        "derived_from_manifest_sha256": _sha(source_manifest),
        "derived_from_raw_snapshot": "runs/etf-case-data-v1/raw; no raw files are copied into this output",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_dataset(manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source_dir, args.output_dir, args.source_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
