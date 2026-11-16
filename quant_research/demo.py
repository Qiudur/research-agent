"""Key-free, synthetic development-window demonstration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets

from .backtest import equal_weight_signal, run_backtest
from .data import load_dataset
from .demo_data import write_demo_dataset
from .signals import trailing_momentum
from .splits import make_splits


def run_demo(output_dir: Path) -> Path:
    """Compare two fixed example signals on synthetic development dates only."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = write_demo_dataset(output_dir / "data")
    panel = load_dataset(manifest_path)
    split = make_splits(
        panel.dates, train_end=panel.dates[50], dev_start=panel.dates[52],
        dev_end=panel.dates[95], holdout_start=panel.dates[97], label_horizon=1,
    )
    window = {"first_result_date": split.dev[0], "last_result_date": split.dev[-1],
              "fee_bps": 2.0, "slippage_bps": 3.0}
    equal_weight = run_backtest(
        panel, equal_weight_signal, top_k=len(panel.symbols), **window)
    momentum = run_backtest(
        panel, trailing_momentum, top_k=1, **window)
    report = {
        "schema": "quant_demo_report/1",
        "label": "SYNTHETIC DATA / NO MODEL / DEVELOPMENT ONLY",
        "dataset_manifest": "data/manifest.json",
        "dataset_sha256": panel.csv_sha256,
        "universe_policy": "fixed_demo",
        "split": {
            "train_end": split.train[-1].isoformat(),
            "dev_start": split.dev[0].isoformat(),
            "dev_end": split.dev[-1].isoformat(),
            "holdout_start": split.holdout[0].isoformat(),
            "excluded_dates": [day.isoformat() for day in split.excluded],
            "label_horizon": split.label_horizon,
        },
        "assumptions": {
            "execution": "signal after t close, trade at t+1 close, realize through t+2 close",
            "fee_bps": window["fee_bps"],
            "slippage_bps": window["slippage_bps"],
        },
        "development": {
            "equal_weight": equal_weight.to_dict(),
            "trailing_momentum_5": momentum.to_dict(),
        },
        "selected_candidate": None,
        "holdout_scored": False,
        "model_calls": 0,
        "limitations": [
            "Synthetic fixed-universe prices cannot establish investment performance.",
            "No agent-generated candidate, independent finalizer, or sealed holdout has run.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"quant-demo-{stamp}-{secrets.token_hex(2)}"
    report = run_demo(output_dir)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
