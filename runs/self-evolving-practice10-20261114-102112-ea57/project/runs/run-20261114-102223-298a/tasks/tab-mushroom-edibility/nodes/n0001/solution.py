"""Measured LightGBM baseline for this tabular task."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Get original columns before modification
    original_cols = [c for c in train.columns]
    test = pd.DataFrame(test_rows, columns=original_cols)
    
    # Create interaction feature between cap_shape and cap_color
    train["cap_shape_cap_color"] = train["cap_shape"].fillna("_NA_") + "_" + train["cap_color"].fillna("_NA_")
    test["cap_shape_cap_color"] = test["cap_shape"].fillna("_NA_") + "_" + test["cap_color"].fillna("_NA_")
    
    # Align columns for test dataframe
    test = test[train.columns]
    
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
