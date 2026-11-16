"""Explicit one-shot sealed evaluation for a completed quant dev run."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from pathlib import Path
import sys

from research_agent.io import read_json

from .finalize import finalize_holdout
from .harness import QuantHarness


def harness_from_run(run_dir: Path) -> QuantHarness:
    """Rebuild the exact scorer configuration from the development manifest."""
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    if run_dir.parent.name != "runs":
        raise ValueError("Quant run must be directly under project runs/")
    project_root = run_dir.parent.parent
    manifest = read_json(run_dir / "manifest.json", "quant_run/1")
    if manifest.get("run_id") != run_dir.name or manifest.get("phase") != "dev":
        raise ValueError("Expected a completed quant development run")
    descriptor = manifest.get("dataset", {})
    relative = Path(descriptor.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset manifest path escapes project")
    dataset = (project_root / relative).resolve(strict=True)
    if not dataset.is_relative_to(project_root) or not dataset.is_file():
        raise ValueError("Dataset manifest path escapes project")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != descriptor.get("sha256"):
        raise ValueError("Dataset manifest changed since development")
    split = manifest["split"]
    execution = manifest["execution"]
    limits = manifest["limits"]
    return QuantHarness(
        run_dir / "task_package" / "quant-demo", dataset,
        train_end=date.fromisoformat(split["train_end"]),
        dev_start=date.fromisoformat(split["dev_start"]),
        dev_end=date.fromisoformat(split["dev_end"]),
        holdout_start=date.fromisoformat(split["holdout_start"]),
        label_horizon=split["label_horizon"],
        top_k=execution["top_k"], fee_bps=execution["fee_bps"],
        slippage_bps=execution["slippage_bps"],
        budget_money=limits["budget_tokens"],
        wall_seconds=limits["wall_seconds"],
        run_seconds=limits["run_seconds"],
        memory_mb=limits["memory_mb"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluate-sealed-once", action="store_true", required=True,
                        help="Consume the one permitted sealed evaluation attempt")
    args = parser.parse_args(argv)
    try:
        run_dir = args.run_dir.expanduser().resolve(strict=True)
        report = finalize_holdout(
            harness_from_run(run_dir), run_dir / "selection.json",
            run_dir / "final")
    except Exception as exc:
        print(f"sealed quant evaluation refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(run_dir / "final/report.json")
    return 0 if report["status"] == "scored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
