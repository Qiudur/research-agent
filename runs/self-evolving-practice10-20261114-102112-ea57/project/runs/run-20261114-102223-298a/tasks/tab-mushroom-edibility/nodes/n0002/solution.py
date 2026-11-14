"""Fixed encoding consistency for Mushroom edibility prediction."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    # Read train and normalize empty strings to consistent marker
    train = pd.read_csv(train_csv)
    for col in train.columns:
        if col != "y" and train[col].dtype == object:
            train[col] = train[col].fillna("").replace("", "<missing>")
    
    y = train.pop("y").astype(int)
    
    # Create test DataFrame from test_rows
    test = pd.DataFrame(test_rows)
    
    # Reorder test columns to match train exactly
    test = test[train.columns]
    
    # Normalize empty strings in test to same marker
    for name in test.columns:
        if test[name].dtype == object:
            test[name] = test[name].fillna("").replace("", "<missing>")
    
    # Build consistent ordinal encodings from train categories
    encoders = {}
    for name in train.columns:
        cats = sorted(train[name].unique())
        encoders[name] = {c: i + 1 for i, c in enumerate(cats)}  # Start at 1, 0 for unknown
    
    # Apply encoding to both train and test
    for name in train.columns:
        train[name] = train[name].map(encoders[name]).fillna(0)
        test[name] = test[name].map(encoders[name]).fillna(0)
    
    # Train and predict
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
