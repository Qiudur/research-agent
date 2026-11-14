"""Measured LightGBM baseline with feature importance filtering."""

from __future__ import annotations

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier


def _identify_redundant_features(df: pd.DataFrame) -> set:
    """Identify features to drop based on high correlation (>0.9)."""
    drop_set = set()
    cols = df.columns.tolist()
    
    # Convert to numeric for correlation calculation
    numeric_df = pd.DataFrame()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_df[col] = df[col]
        else:
            # Use categorical codes for non-numeric
            numeric_df[col] = df[col].cat.codes if hasattr(df[col], 'cat') else pd.factorize(df[col])[0]
    
    # Compute pairwise correlations
    corr_matrix = numeric_df.corr().abs()
    
    # Mark one of each highly correlated pair for removal
    for i, col_i in enumerate(cols):
        if col_i not in numeric_df.columns or col_i in drop_set:
            continue
        for j, col_j in enumerate(cols):
            if j <= i or col_j in drop_set:
                continue
            if col_j not in numeric_df.columns:
                continue
            corr = corr_matrix.loc[col_i, col_j] if col_i in corr_matrix.columns and col_j in corr_matrix.columns else 0
            if corr > 0.9:
                # Drop the second one (arbitrary choice when correlation is high)
                drop_set.add(col_j)
    
    return drop_set


def _filter_low_importance(X: pd.DataFrame, y: pd.Series, importance_threshold: float = 0.01) -> list:
    """Train quick model and identify important features."""
    model = LGBMClassifier(random_state=20261114, n_estimators=50, n_jobs=1, verbosity=-1)
    model.fit(X, y)
    
    importances = dict(zip(X.columns, model.feature_importances_.tolist()))
    max_importance = max(importances.values()) if importances else 1
    
    keep_cols = [col for col, imp in importances.items() 
                 if imp / max(max_importance, 1e-8) >= importance_threshold]
    return keep_cols


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Store original columns
    all_cols = list(train.columns)
    
    test = pd.DataFrame(test_rows, columns=all_cols)
    
    # Process data types consistently
    for name in all_cols:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Step 1: Remove highly correlated/redundant features
    redundant_cols = _identify_redundant_features(train)
    remaining_cols = [c for c in all_cols if c not in redundant_cols]
    
    # Step 2: Filter by feature importance
    train_filtered = train[remaining_cols]
    important_cols = _filter_low_importance(train_filtered, y, importance_threshold=0.005)
    
    # Ensure we keep at least some features
    if len(important_cols) < 5:
        # Keep top N by importance
        sorted_by_imp = sorted(enumerate(LGBMClassifier(random_state=20261114, n_estimators=50, n_jobs=1, verbosity=-1).fit(train_filtered, y).feature_importances_), reverse=True)[:10]
        important_cols = [remaining_cols[i] for i, _ in sorted_by_imp]
    
    # Prepare final datasets
    X_train = train[important_cols]
    X_test = test[important_cols]
    
    # Final model
    model = LGBMClassifier(
        random_state=20261114,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=1,
        verbosity=-1
    )
    model.fit(X_train, y)
    
    return model.predict_proba(X_test)[:, 1].tolist()
