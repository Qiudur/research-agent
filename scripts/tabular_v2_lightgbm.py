"""Deterministic train-only LightGBM reference for candidate bank-v2 tasks."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv).drop(columns=["source_row_id"], errors="ignore")
    labels = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows).reindex(columns=train.columns)
    for column in train.columns:
        if pd.api.types.is_numeric_dtype(train[column]):
            test[column] = pd.to_numeric(test[column], errors="coerce")
        else:
            categories = pd.Categorical(pd.concat([train[column], test[column]], ignore_index=True)).categories
            train[column] = pd.Categorical(train[column], categories=categories)
            test[column] = pd.Categorical(test[column], categories=categories)
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, labels)
    return model.predict_proba(test)[:, 1].tolist()
