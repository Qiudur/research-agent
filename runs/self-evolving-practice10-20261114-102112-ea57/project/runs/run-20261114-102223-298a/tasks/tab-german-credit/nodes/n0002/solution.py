"""LightGBM with depth limiting, regularization, and class weighting."""

from __future__ import annotations

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Process features (handle numeric vs categorical)
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Compute balanced sample weights for imbalanced classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        max_depth=3,
        lambda_l1=0.5,
        lambda_l2=0.5
    )
    model.fit(train, y, sample_weight=sample_weights)
    
    return model.predict_proba(test)[:, 1].tolist()
