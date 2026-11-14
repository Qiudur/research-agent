"""Measured LightGBM baseline with robust hybrid preprocessing."""

from __future__ import annotations

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Define continuous columns
    continuous_cols = {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"}
    
    # Initialize feature lists
    train_features = []
    test_features = []
    feature_names = []
    
    # First pass: process test_rows into a proper DataFrame
    all_cols = list(train.columns)
    test = pd.DataFrame(test_rows, columns=all_cols)
    
    for col in all_cols:
        if col in continuous_cols:
            # Numeric: coerce and fill with train median
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")
            med = train[col].median()
            train[col] = train[col].fillna(med)
            test[col] = test[col].fillna(med)
            train_features.append(train[col].values.reshape(-1, 1))
            test_features.append(test[col].values.reshape(-1, 1))
            feature_names.append(col)
        else:
            # Categorical: compute cardinality from training data only
            valid_cats = set(train[col].dropna().unique())
            card = len(valid_cats)
            low_card_threshold = 6
            
            if card <= low_card_threshold:
                # Low cardinality: one-hot encoding
                dummies_train = pd.get_dummies(
                    train[col], prefix=col, drop_first=False, dtype=float
                )
                dummies_test = pd.get_dummies(
                    test[col], prefix=col, drop_first=False, dtype=float
                )
                # Align columns
                common_cols = dummies_train.columns
                for c in common_cols:
                    if c not in dummies_test.columns:
                        dummies_test[c] = 0.0
                dummies_test = dummies_test.reindex(columns=common_cols, fill_value=0.0)
                train_features.append(dummies_train.values)
                test_features.append(dummies_test.values)
                feature_names.extend(dummies_train.columns.tolist())
            else:
                # High cardinality: frequency + smoothed target encoding
                n_train = len(train)
                global_mean = y.mean()
                smoothing = 15.0
                
                # Compute stats from training only
                freq_map = train[col].value_counts(normalize=True).to_dict()
                group_counts = train[col].value_counts().to_dict()
                group_means = train.groupby(col)["y"].mean().to_dict()
                
                # Build smoothed target map
                smooth_target = {}
                for cat in valid_cats:
                    cnt = group_counts.get(cat, 0)
                    gm = group_means.get(cat, global_mean)
                    smooth_target[cat] = (cnt * gm + smoothing * global_mean) / (cnt + smoothing)
                
                def encode_freq(series):
                    return series.map(lambda x: freq_map.get(x, 0.0)).astype(float)
                
                def encode_target(series):
                    return series.map(lambda x: smooth_target.get(x, global_mean)).astype(float)
                
                train_feat = np.column_stack([
                    encode_freq(train[col]).values,
                    encode_target(train[col]).values
                ])
                test_feat = np.column_stack([
                    encode_freq(test[col]).values,
                    encode_target(test[col]).values
                ])
                
                feat_prefix = f"{col}_enc"
                train_features.append(train_feat)
                test_features.append(test_feat)
                feature_names.extend([f"{feat_prefix}_freq", f"{feat_prefix}_target"])
    
    X_train = np.hstack(train_features)
    X_test = np.hstack(test_features)
    
    # LightGBM with regularization and early stopping
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        n_estimators=200,
        max_depth=4,
        num_leaves=10,
        learning_rate=0.03,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5
    )
    model.fit(X_train, y)
    
    probs = model.predict_proba(X_test)[:, 1]
    return [float(p) for p in probs]
