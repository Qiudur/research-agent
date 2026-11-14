"""Logistic regression with group-aware imputation for mushroom classification."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Replace empty strings with NaN for proper handling
    NA_PLACEHOLDER = ""
    feature_cols = train.columns.tolist()
    
    # Compute imputation mapping: mode of stalk_root per cap_shape group
    impute_mapping = {}
    for shape in train['cap_shape'].dropna().unique():
        vals = train.loc[train['cap_shape'] == shape, 'stalk_root'].dropna()
        if len(vals) > 0:
            impute_mapping[shape] = vals.mode()[0]
    
    # Default imputation value for unseen cap_shapes
    default_impute = train['stalk_root'].mode()[0] if len(train['stalk_root'].mode()) > 0 else ''
    
    def impute_stalk_root(df):
        df = df.copy()
        for idx, row in df.iterrows():
            val = row['stalk_root']
            if pd.isna(val) or str(val).strip() == '':
                shape = row['cap_shape']
                df.at[idx, 'stalk_root'] = impute_mapping.get(shape, default_impute)
        return df
    
    train = impute_stalk_root(train)
    test = impute_stalk_root(test)
    
    # Ensure all features are strings
    for col in feature_cols:
        train[col] = train[col].fillna('').astype(str)
        test[col] = test[col].fillna('').astype(str)
    
    # One-hot encode all categorical features
    train_encoded = pd.get_dummies(train, prefix_sep='__', columns=feature_cols)
    test_encoded = pd.get_dummies(test, prefix_sep='__', columns=feature_cols)
    
    # Align test columns with train (handle missing categories in test)
    for col in train_encoded.columns:
        if col not in test_encoded.columns:
            test_encoded[col] = 0
    test_encoded = test_encoded[train_encoded.columns]
    
    # Train logistic regression
    model = LogisticRegression(random_state=20261114, max_iter=1000)
    model.fit(train_encoded, y)
    
    # Predict probability of positive class (poisonous)
    probs = model.predict_proba(test_encoded)[:, 1]
    return probs.tolist()
