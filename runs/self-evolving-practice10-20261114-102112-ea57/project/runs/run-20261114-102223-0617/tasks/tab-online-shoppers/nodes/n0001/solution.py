"""Measured LightGBM baseline with VisitorType-BounceRates interaction terms."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def _add_bounce_visitor_interactions(df: pd.DataFrame) -> None:
    """Add interaction terms between VisitorType and BounceRates."""
    if 'VisitorType' in df.columns and 'BounceRates' in df.columns:
        for vt in ['New_Visitor', 'Returning_Visitor']:
            is_type = (df['VisitorType'] == vt).astype(float)
            df[f'BounceRates_{vt}'] = df['BounceRates'] * is_type


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    original_cols = list(train.columns)
    
    # Add interaction features to train
    _add_bounce_visitor_interactions(train)
    
    test = pd.DataFrame(test_rows, columns=original_cols)
    
    # Align dtypes between train and test
    for name in original_cols:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(
                pd.concat([train[name], test[name]], ignore_index=True)
            ).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Add same interaction features to test
    _add_bounce_visitor_interactions(test)
    
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
