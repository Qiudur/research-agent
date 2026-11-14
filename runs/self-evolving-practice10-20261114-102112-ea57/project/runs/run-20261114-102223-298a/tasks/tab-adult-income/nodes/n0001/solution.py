"""Measured LightGBM baseline for this tabular task."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    """Predict income >50K using LightGBM with target encoding for categorical features."""
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    global_mean = float(y.mean())
    
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Identify categorical vs numeric columns
    cat_cols = []
    num_cols = []
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            num_cols.append(name)
            test[name] = pd.to_numeric(test[name])
        else:
            cat_cols.append(name)
    
    # Create target encoding for categorical columns (mean target per category)
    cat_encodings = {}
    for col in cat_cols:
        # Compute mean target value for each category in training data
        cat_mean = train.groupby(col).apply(lambda g: y[g.index].mean()).to_dict()
        # Fill NA categories with 0 (they won't occur after groupby but safety)
        cat_mean = {k: v for k, v in cat_mean.items() if pd.notna(v)}
        cat_encodings[col] = cat_mean
    
    # Apply encoding: replace category with its mean target value
    for col in cat_cols:
        train[col] = train[col].map(cat_encodings[col]).fillna(global_mean)
        test[col] = test[col].map(cat_encodings[col]).fillna(global_mean)
        train[col] = train[col].astype(float)
        test[col] = test[col].astype(float)
    
    # Use all columns as features
    feature_cols = list(train.columns)
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    
    # Fit LightGBM classifier
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(X_train, y)
    return model.predict_proba(X_test)[:, 1].tolist()
