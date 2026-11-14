"""LightGBM classifier with proper categorical handling."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Identify numeric vs categorical columns
    cat_cols = []
    num_cols = []
    
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            num_cols.append(name)
            test[name] = pd.to_numeric(test[name], errors='coerce')
        else:
            cat_cols.append(name)
            # Store categories from train+test combined, then convert to ordered integers
            combined_categories = sorted(set(
                set(train[name].astype(str).unique()) | 
                set(test[name].astype(str).unique())
            ))
            cat_mapping = {k: v for v, k in enumerate(combined_categories)}
            train[name] = train[name].astype(str).map(cat_mapping)
            test[name] = test[name].astype(str).map(cat_mapping)
    
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        class_weight='balanced',
        learning_rate=0.1,
        num_leaves=8,
        min_child_samples=20
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
