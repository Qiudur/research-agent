"""Target Encoding with Smoothing for High Cardinality Features."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    # Load and prepare data
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Compute target encoding statistics from TRAIN ONLY
    global_mean = float(y.mean())
    
    # High-cardinality features to encode with target encoding
    cat_features_target_enc = ["native country", "occupation"]
    encoding_stats = {}
    
    for feat in cat_features_target_enc:
        if feat not in train.columns:
            continue
        # Count occurrences per category
        counts = train[feat].value_counts().to_dict()
        total_train_count = len(train)
        # Mean of y per category
        means = {}
        for cat in counts.keys():
            mask = train[feat] == cat
            means[cat] = float(y[mask].mean())
        encoding_stats[feat] = {
            "counts": counts,
            "means": means,
            "global_mean": global_mean,
            "total_count": total_train_count,
        }
    
    # Apply target encoding with smoothing
    def apply_target_encoding(df: pd.DataFrame, feat: str, stats: dict) -> None:
        """Encode feature with smoothed target encoding."""
        alpha = 10  # Smoothing parameter
        global_m = stats["global_mean"]
        counts = stats["counts"]
        means = stats["means"]
        
        def encode_val(val):
            if pd.isna(val):
                return global_m
            count = counts.get(val, 0)
            local_mean = means.get(val, global_m)
            # Smoothed encoding: weighted avg of local mean and global mean
            return (count * local_mean + alpha * global_m) / (count + alpha)
        
        df[feat] = df[feat].apply(encode_val)
    
    # Encode both train and test using stats computed from train only
    for feat in cat_features_target_enc:
        if feat in encoding_stats:
            apply_target_encoding(train, feat, encoding_stats[feat])
            apply_target_encoding(test, feat, encoding_stats[feat])
    
    # Handle remaining categorical columns with category codes
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            continue
        if name in cat_features_target_enc:
            continue  # Already encoded
        # Use category codes for remaining categorical features
        categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
        train[name] = pd.Categorical(train[name], categories=categories).codes.astype(float)
        test[name] = pd.Categorical(test[name], categories=categories).codes.astype(float)
    
    # Ensure test numeric columns match train types
    for name in train.columns:
        if name not in test.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name], errors="coerce")
    
    # Calculate scale_pos_weight from training data class distribution
    n_pos = int(sum(y))
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Fit LightGBM model with tuned hyperparameters
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        min_data_in_leaf=30,
    )
    model.fit(train, y)
    
    # Predict probabilities
    return model.predict_proba(test)[:, 1].tolist()
