"""Measured LightGBM baseline for the Bank Marketing task."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    # duration is present in train.csv but is unavailable before a call.
    train = pd.read_csv(train_csv).drop(columns=["duration"], errors="ignore")
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
