"""Reproduce the registered ETF baseline case without model calls or API keys."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import secrets
import sys

from research_agent.io import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def run_case(*, output_dir: Path, dataset_manifest: Path | None = None,
             protocol_path: Path = ROOT / "config/etf_case_study.json") -> Path:
    """Download a pinned public snapshot, evaluate fixed signals, and render evidence."""
    # Import only when running: --help works without network or a model SDK.
    from .data import load_dataset
    from .etf_data import fetch_etf_dataset
    from .etf_report import generate_report
    from .etf_study import run_study

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Use a fresh case directory; existing evidence is retained")
    protocol_path = Path(protocol_path).expanduser().resolve(strict=True)
    if dataset_manifest is not None:
        dataset_manifest = Path(dataset_manifest).expanduser().resolve(strict=True)
        load_dataset(dataset_manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {"schema": "etf_case_pipeline/1", "status": "running", "phase": "data",
             "started_at": datetime.now(timezone.utc).isoformat(), "model_calls": 0}
    atomic_json(output_dir / "pipeline.json", state)
    try:
        print("ETF case: verifying the market-data snapshot", flush=True)
        if dataset_manifest is None:
            dataset_manifest = fetch_etf_dataset(output_dir / "private_data")
        state["phase"] = "study"
        atomic_json(output_dir / "pipeline.json", state)
        print("ETF case: evaluating the registered development windows and frozen holdout", flush=True)
        study = run_study(dataset_manifest, protocol_path, output_dir / "study")
        state["phase"] = "report"
        atomic_json(output_dir / "pipeline.json", state)
        report = generate_report(study, output_dir / "report")
        state.update(status="complete", phase="complete", report="report/index.html")
        atomic_json(output_dir / "pipeline.json", state)
        return report
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__,
                     error=str(exc),
                     message="ETF case did not complete; retained artifacts are not a complete result")
        atomic_json(output_dir / "pipeline.json", state)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-manifest", type=Path,
                        help="Reuse an already downloaded and hashed ETF snapshot")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/etf_case_study.json")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / "runs" / f"etf-case-{stamp}-{secrets.token_hex(2)}"
    try:
        report = run_case(output_dir=output, dataset_manifest=args.dataset_manifest,
                          protocol_path=args.protocol)
    except Exception as exc:
        print(f"ETF case failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        if (output / "pipeline.json").is_file():
            print(f"Retained state: {output / 'pipeline.json'}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
