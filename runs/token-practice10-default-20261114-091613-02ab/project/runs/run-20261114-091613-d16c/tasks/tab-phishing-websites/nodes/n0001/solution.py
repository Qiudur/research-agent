"""Measured LightGBM baseline for this tabular task."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Convert all columns to numeric since they are all numeric features
    for name in train.columns:
        train[name] = pd.to_numeric(train[name], errors='coerce').fillna(0)
        test[name] = pd.to_numeric(test[name], errors='coerce').fillna(0)
    
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
