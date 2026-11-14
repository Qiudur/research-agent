"""Measure five candidate tabular tasks on dev only, then activate metrics.

The dev-gap threshold of 0.01 was fixed before this calibration. This script
never reads or scores hidden/holdout.csv; it stages candidate code in a
temporary directory outside the task packages and uses the sandboxed B scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import tabular_harness

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261114
MIN_GAP = 0.01
BASELINE = ROOT / "scripts/tabular_v2_logistic.py"
REFERENCES = {
    "tab-spambase": ROOT / "scripts/tabular_v2_lightgbm.py",
    "tab-qsar-biodegradation": ROOT / "scripts/tabular_v2_lightgbm.py",
    "tab-wine-quality": ROOT / "scripts/tabular_v2_lightgbm.py",
    "tab-ionosphere": ROOT / "scripts/tabular_v2_lightgbm.py",
    "tab-monks-1": ROOT / "scripts/tabular_v2_lightgbm.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(task_id: str, source: Path) -> dict:
    with TemporaryDirectory(prefix="bank-v2-dev-") as temp:
        node = Path(temp)
        shutil.copyfile(source, node / "solution.py")
        result = tabular_harness.raw_score(task_id, node, "dev", seed=SEED)
    if not result["ok"] or not result["correct"] or not isinstance(result["raw"], (int, float)):
        raise RuntimeError(f"{task_id} {source.name} dev measurement failed: {result['error']}")
    return {"model": source.name, "auc": result["raw"],
            "sandbox_seconds": result["seconds"], "source_sha256": _sha(source)}


def measure_all() -> list[dict]:
    rows = []
    for task_id, reference_source in REFERENCES.items():
        root = ROOT / "tasks" / task_id
        baseline = _measure(task_id, BASELINE)
        reference = _measure(task_id, reference_source)
        gap = reference["auc"] - baseline["auc"]
        if gap < MIN_GAP:
            raise ValueError(f"{task_id}: dev reference gap {gap:.8f} < {MIN_GAP}")
        rows.append({
            "schema": "tabular_bank_v2_reference/1", "task_id": task_id,
            "split": "dev", "seed": SEED,
            "baseline": baseline, "reference": reference,
            "gap": gap, "min_gap": MIN_GAP, "gap_valid": True,
            "split_manifest_sha256": _sha(root / "split_manifest.json"),
            "scope": "train/dev only; hidden holdout unopened and unscored",
        })
    return rows


def activate(rows: list[dict]) -> None:
    for row in rows:
        root = ROOT / "tasks" / row["task_id"]
        task_path = root / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("scenario") != "tabular":
            raise ValueError(f"Not tabular: {root}")
        task["metric"]["baseline_raw"] = row["baseline"]["auc"]
        task["metric"]["ceiling_raw"] = row["reference"]["auc"]
        task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        shutil.copyfile(BASELINE, root / "workspace/original.py")
        measurement = root / "hidden/bank_v2_reference_measurements.json"
        measurement.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write task metrics and baseline code after all five pass")
    args = parser.parse_args()
    measured = measure_all()
    if args.apply:
        activate(measured)
    print(json.dumps({"applied": args.apply, "dev_gaps":
                      {row["task_id"]: row["gap"] for row in measured}}))
