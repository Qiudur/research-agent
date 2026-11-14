"""Simplified LightGBM to reduce interaction-induced overfitting."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Fill missing values with 'missing' token for consistent handling
    for name in train.columns:
        if not pd.api.types.is_numeric_dtype(train[name]):
            train[name] = train[name].fillna('missing')
            test[name] = test[name].fillna('missing')
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Shallow trees (stump-like) to avoid interaction overfitting
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        max_depth=1,
        num_leaves=2,
        min_child_samples=50
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
