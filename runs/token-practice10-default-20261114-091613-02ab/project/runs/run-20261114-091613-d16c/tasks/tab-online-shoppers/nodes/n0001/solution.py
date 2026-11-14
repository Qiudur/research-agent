"""LightGBM classifier with default hyperparameters on all available features."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Handle type consistency between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            # Combine train and test for consistent category encoding
            categories = pd.Categorical(
                pd.concat([train[name], test[name]], ignore_index=True)
            ).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Use default LightGBM hyperparameters with fixed random state
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    
    return model.predict_proba(test)[:, 1].tolist()
