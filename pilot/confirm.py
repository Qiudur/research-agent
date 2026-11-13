"""Fresh-seed confirmation of the one-node local pilot (not formal holdout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from research_agent.io import atomic_json

from .harness import PilotHarness, TASK_ID


def confirm(run_dir: Path, seeds: list[int], *, size: int = 2500, repetitions: int = 7) -> dict:
    run_dir = Path(run_dir).resolve()
    node_dir = run_dir / "tasks" / TASK_ID / "nodes" / "n0001"
    if not (node_dir / "solution.py").is_file():
        raise FileNotFoundError(node_dir / "solution.py")
    harness = PilotHarness(size=size, repetitions=repetitions, measurements_path=run_dir / "measurements.jsonl")
    results = [harness.score(TASK_ID, node_dir, "dev", seed=seed) for seed in seeds]
    if not all(item["ok"] and item["correct"] for item in results):
        raise RuntimeError("A fresh-seed correctness or timing check failed")
    speeds = [float(item["raw"]) for item in results]
    report = {
        "schema": "pilot_confirmation/1", "run_id": run_dir.name,
        "task_id": TASK_ID, "seeds": seeds, "input_size": size,
        "repetitions_per_seed": repetitions, "correct": True,
        "raw_speedups": speeds, "mean_speedup": mean(speeds),
        "median_speedup": median(speeds), "min_speedup": min(speeds),
        "max_speedup": max(speeds), "model_cost": None,
        "note": "Fresh-seed correctness and speed measurements only; no live model calls or production holdout.",
    }
    atomic_json(run_dir / "fresh_seed_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2500)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261201, 20261202, 20261203])
    args = parser.parse_args()
    print(json.dumps(confirm(args.run_dir, args.seeds, size=args.size, repetitions=args.repetitions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
