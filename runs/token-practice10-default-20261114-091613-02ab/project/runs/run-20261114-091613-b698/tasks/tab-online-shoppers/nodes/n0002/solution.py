"""Improved LightGBM baseline for tabular task with class imbalance handling."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Preprocess columns: numeric or categorical
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Compute scale_pos_weight for class imbalance (~15% positive rate)
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        num_leaves=63,
        min_child_samples=20,
        learning_rate=0.08,
        feature_fraction=0.9
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
