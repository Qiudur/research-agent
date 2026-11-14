"""Regularized LightGBM model to prevent overfitting."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    X_train = train.copy()
    test = pd.DataFrame(test_rows, columns=X_train.columns)
    
    # Align dtypes between train and test
    for name in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[name]):
            test[name] = pd.to_numeric(test[name], errors='coerce')
        else:
            all_cats = sorted(set(pd.concat([X_train[name], test[name]], ignore_index=True).dropna()))
            cat_type = pd.CategoricalDtype(categories=all_cats, ordered=False)
            X_train[name] = X_train[name].astype(cat_type)
            test[name] = test[name].astype(cat_type)
    
    # Regularized model with fewer leaves to prevent overfitting
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=30,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y)
    return model.predict_proba(test)[:, 1].tolist()
