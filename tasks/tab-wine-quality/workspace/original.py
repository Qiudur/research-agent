"""Deterministic, train-only logistic baseline for candidate bank-v2 tasks.

The same code can be copied to a task's workspace/original.py. It reads only
the supplied train CSV and test feature dictionaries. No hidden files or
holdout labels are accessed.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv).drop(columns=["source_row_id"], errors="ignore")
    labels = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows).reindex(columns=train.columns)
    for column in train.columns:
        if pd.api.types.is_numeric_dtype(train[column]):
            test[column] = pd.to_numeric(test[column], errors="coerce")
    x_train = pd.get_dummies(train, dummy_na=True)
    x_test = pd.get_dummies(test, dummy_na=True).reindex(columns=x_train.columns, fill_value=0)
    for column in x_train.columns:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=20261114),
    )
    model.fit(x_train, labels)
    return model.predict_proba(x_test)[:, 1].tolist()
