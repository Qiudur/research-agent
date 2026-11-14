"""Measured LightGBM baseline for this tabular task."""

from __future__ import annotations

import math
import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Build test dataframe preserving row order
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Standardize dtypes between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name], errors='coerce')
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Fit model
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    
    # Get probabilities and ensure format compliance
    probs = model.predict_proba(test)[:, 1]
    
    # Ensure exact row count and format: finite float in [0, 1]
    result = []
    expected_len = len(test_rows)
    
    for i in range(expected_len):
        p = float(probs[i]) if i < len(probs) else 0.5
        # Clamp to valid range and handle non-finite values
        if not math.isfinite(p):
            p = 0.5
        elif p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        result.append(p)
    
    return result
