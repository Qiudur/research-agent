"""Measure the pinned LightGBM baseline and AutoGluon reference on dev only.

Run with the project's Python 3.11 venv, after installing libomp and the
tabular dependencies. This script does not read the holdout split.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "tab-bank-marketing"
SEED = 20261114
sys.path.insert(0, str(ROOT))
import tabular_harness


def measure(time_limit: int) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    with TemporaryDirectory() as temp:
        node = Path(temp) / "node"
        node.mkdir()
        shutil.copyfile(TASK / "workspace/original.py", node / "solution.py")
        baseline = tabular_harness.raw_score("tab-bank-marketing", node, "dev", seed=SEED)
        if not baseline["ok"] or not baseline["correct"]:
            raise RuntimeError(f"LightGBM baseline failed: {baseline['error']}")

        forbidden = ["duration"]
        train = pd.read_csv(TASK / "workspace/train.csv").drop(columns=forbidden, errors="ignore")
        dev = pd.read_csv(TASK / "hidden/dev.csv").drop(columns=forbidden, errors="ignore")
        started = time.perf_counter()
        predictor = TabularPredictor(
            label="y", problem_type="binary", eval_metric="roc_auc",
            path=str(Path(temp) / "autogluon"),
        ).fit(train_data=train, time_limit=time_limit, presets="medium_quality", verbosity=1)
        fit_seconds = time.perf_counter() - started
        probabilities = predictor.predict_proba(dev.drop(columns=["y"]))[1].tolist()
        reference_auc = tabular_harness.roc_auc(dev["y"].astype(int).tolist(), probabilities)
        models = predictor.leaderboard(dev)["model"].tolist()

    result = {
        "schema": "tabular_reference/1", "task_id": "tab-bank-marketing",
        "split": "dev", "seed": SEED, "time_limit_seconds": time_limit,
        "presets": "medium_quality", "eval_metric": "roc_auc",
        "train_rows": len(train), "dev_rows": len(dev),
        "baseline": {"model": "workspace/original.py LGBMClassifier", "auc": baseline["raw"],
                     "sandbox_seconds": baseline["seconds"]},
        "reference": {"model": "AutoGluon TabularPredictor", "auc": reference_auc,
                      "best_model_by_internal_validation": predictor.model_best,
                      "trained_models": models, "fit_seconds": fit_seconds},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "pandas": version("pandas"), "scikit_learn": version("scikit-learn"),
                        "lightgbm": version("lightgbm"), "autogluon_tabular": version("autogluon.tabular"),
                        "catboost": version("catboost"), "xgboost": version("xgboost")},
        "scope": "dev only; holdout unopened; no model API tokens or monetary cost measured",
    }
    if reference_auc <= baseline["raw"]:
        raise RuntimeError(f"AutoGluon reference {reference_auc} did not beat baseline {baseline['raw']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=180)
    args = parser.parse_args()
    measurement = measure(args.time_limit)
    output = TASK / "hidden/reference_measurements.json"
    output.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measurement, indent=2))
