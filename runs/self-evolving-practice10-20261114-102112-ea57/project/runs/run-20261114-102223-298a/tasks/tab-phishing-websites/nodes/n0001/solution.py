"""Logistic regression baseline with label transformation."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    
    # Transform y from -1/0/1 to binary: -1 or 0 → 0 (negative), 1 → 1 (positive)
    y = train.pop("y").apply(lambda x: 1 if x == 1 else 0).astype(int)
    
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Preprocess numeric and categorical columns consistently
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    model = LogisticRegression(random_state=20261114, max_iter=1000, solver="lbfgs")
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
