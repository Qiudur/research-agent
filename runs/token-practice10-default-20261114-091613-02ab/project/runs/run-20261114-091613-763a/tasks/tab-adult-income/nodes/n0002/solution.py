"""Target encoding implementation for high-cardinality categorical features."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(float)
    train_idx = train.index.copy()
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Handle numeric conversion for existing numeric columns in test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            # Categorical columns - convert to string for consistent encoding
            train[name] = train[name].astype(str)
            test[name] = test[name].astype(str)
    
    # Target encoding for categorical (now string) columns
    global_mean = y.mean()
    m = 20  # Additive smoothing parameter
    
    target_values = y.values
    for col in train.columns:
        train_cats = train[col].values
        test_cats = test[col].values
        
        # Compute category statistics from training data only
        uniq, counts = pd.unique(train_cats), None
        uniq = pd.unique(train_cats)
        
        # Get mean target per category
        group_means = {}
        for cat in uniq:
            mask = train_cats == cat
            if mask.any():
                group_means[cat] = target_values[mask].mean()
        
        # Build smoothed encoding map: (n*mean + m*global) / (n+m)
        enc_map = {}
        for cat in uniq:
            cat_mean = group_means.get(cat, global_mean)
            cat_count = sum(train_cats == cat)
            enc_map[cat] = (cat_count * cat_mean + m * global_mean) / (cat_count + m)
        
        # Apply encoding to both datasets
        train[col] = [enc_map.get(c, global_mean) for c in train_cats]
        test[col] = [enc_map.get(c, global_mean) for c in test_cats]
    
    # Train LightGBM on encoded features
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
