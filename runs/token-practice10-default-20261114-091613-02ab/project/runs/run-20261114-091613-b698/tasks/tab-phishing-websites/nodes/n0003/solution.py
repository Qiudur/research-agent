"""LightGBM baseline with hyperparameter tuning via random search."""

from __future__ import annotations

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


def _identify_redundant_features(df: pd.DataFrame) -> set:
    """Identify features to drop based on high correlation (>0.9)."""
    drop_set = set()
    cols = df.columns.tolist()
    
    numeric_df = pd.DataFrame()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_df[col] = df[col]
        else:
            numeric_df[col] = df[col].cat.codes if hasattr(df[col], 'cat') else pd.factorize(df[col])[0]
    
    corr_matrix = numeric_df.corr().abs()
    
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


def _tune_hyperparams(X: pd.DataFrame, y: pd.Series, n_iter: int = 20) -> dict:
    """Random search for optimal LightGBM hyperparameters using 5-fold CV."""
    param_grids = {
        'num_leaves': [15, 31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 5, 8, 10]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=20261114)
    best_score = -np.inf
    best_params = None
    
    np.random.seed(20261114)
    
    for _ in range(n_iter):
        params = {
            'n_estimators': 100,
            'random_state': 20261114,
            'n_jobs': 1,
            'verbosity': -1,
            'num_leaves': np.random.choice(param_grids['num_leaves']),
            'learning_rate': np.random.choice(param_grids['learning_rate']),
            'max_depth': np.random.choice(param_grids['max_depth'])
        }
        
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va, proba))
            except Exception:
                continue
        
        if len(scores) == 5:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
    
    return best_params if best_params else {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 20261114,
        'n_jobs': 1,
        'verbosity': -1
    }


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    all_cols = list(train.columns)
    test = pd.DataFrame(test_rows, columns=all_cols)
    
    for name in all_cols:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    redundant_cols = _identify_redundant_features(train)
    remaining_cols = [c for c in all_cols if c not in redundant_cols]
    
    train_filtered = train[remaining_cols]
    important_cols = _filter_low_importance(train_filtered, y, importance_threshold=0.005)
    
    if len(important_cols) < 5:
        sorted_by_imp = sorted(enumerate(LGBMClassifier(random_state=20261114, n_estimators=50, n_jobs=1, verbosity=-1).fit(train_filtered, y).feature_importances_), reverse=True)[:10]
        important_cols = [remaining_cols[i] for i, _ in sorted_by_imp]
    
    X_train = train[important_cols]
    X_test = test[important_cols]
    
    best_params = _tune_hyperparams(X_train, y)
    
    best_params['n_estimators'] = 300
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y)
    
    return model.predict_proba(X_test)[:, 1].tolist()
