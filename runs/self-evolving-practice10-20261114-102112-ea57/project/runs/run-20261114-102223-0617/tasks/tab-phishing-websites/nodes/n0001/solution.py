"""Logistic regression with L2 regularization as sanity check."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y")
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Align types between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # L2 regularization is default (penalty='l2', solver='lbfgs')
    model = LogisticRegression(random_state=20261114, max_iter=1000)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
