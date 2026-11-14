"""LightGBM classifier for phishing website prediction."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Ensure consistent dtypes between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name], errors='coerce')
        else:
            all_cats = sorted(set(train[name].dropna().unique()) | set(test[name].dropna().unique()))
            cat_type = pd.CategoricalDtype(categories=all_cats)
            train[name] = train[name].astype(cat_type)
            test[name] = test[name].astype(cat_type)
    
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
