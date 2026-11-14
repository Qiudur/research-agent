"""LightGBM classifier for phishing website detection."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    X_train = train.copy()
    
    test = pd.DataFrame(test_rows)
    # Align column order with training data
    test = test[X_train.columns]
    
    # Convert numeric columns
    for name in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[name]):
            test[name] = pd.to_numeric(test[name], errors='coerce')
    
    # Fit LightGBM classifier
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.05,
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        force_col_wise=True
    )
    model.fit(X_train, y)
    return model.predict_proba(test)[:, 1].tolist()
