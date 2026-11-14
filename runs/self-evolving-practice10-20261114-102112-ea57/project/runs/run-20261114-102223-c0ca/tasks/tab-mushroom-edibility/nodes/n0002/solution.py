"""Measured LightGBM baseline for this tabular task."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Normalize empty strings to a consistent placeholder to match train/test encoding
    NA_PLACEHOLDER = "__NA__"
    for col in train.columns:
        train[col] = train[col].fillna(NA_PLACEHOLDER).astype(str)
        test[col] = test[col].fillna(NA_PLACEHOLDER).astype(str)
        test[col] = test[col].replace('', NA_PLACEHOLDER)
    
    for name in train.columns:
        categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
        train[name] = pd.Categorical(train[name], categories=categories)
        test[name] = pd.Categorical(test[name], categories=categories)
    
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
