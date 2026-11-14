"""LightGBM with optimized hyperparameters from 5-fold CV."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
