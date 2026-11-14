"""LightGBM model with cap_color-habitat interaction terms."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Handle missing values by treating them as explicit 'MISSING' category
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            continue
        else:
            train[name] = train[name].fillna('MISSING')
    
    # Create interaction term between cap_color and habitat
    train['cap_color_x_habitat'] = train['cap_color'].astype(str) + '_' + train['habitat'].astype(str)
    
    # Prepare test data - extract original columns (exclude target, exclude interaction we'll add)
    original_cols = [c for c in train.columns if c != 'cap_color_x_habitat']
    test = pd.DataFrame(test_rows, columns=original_cols)
    
    # Handle missing values in test data
    for name in test.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name], errors='coerce')
        else:
            test[name] = test[name].fillna('MISSING').replace('', 'MISSING')
    
    # Create interaction term in test data too
    test['cap_color_x_habitat'] = test['cap_color'].astype(str) + '_' + test['habitat'].astype(str)
    
    # For categorical columns, align categories between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            continue
        categories = sorted(set(train[name].unique()) | set(test[name].unique()))
        train[name] = pd.Categorical(train[name], categories=categories)
        test[name] = pd.Categorical(test[name], categories=categories)
    
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
