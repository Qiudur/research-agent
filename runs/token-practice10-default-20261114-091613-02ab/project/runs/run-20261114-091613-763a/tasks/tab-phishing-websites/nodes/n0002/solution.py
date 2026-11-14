"""XGBoost with CV-tuned hyperparameters for this tabular task."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Align dtypes between train and test
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name])
        else:
            categories = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True)).categories
            train[name] = pd.Categorical(train[name], categories=categories)
            test[name] = pd.Categorical(test[name], categories=categories)
    
    # Convert categoricals to numeric codes for XGBoost
    for name in train.columns:
        if train[name].dtype.name == 'category':
            train[name] = train[name].cat.codes.astype(float)
            test[name] = test[name].cat.codes.astype(float)
    
    # Hyperparameters to tune via cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261114)
    
    # Search grid for XGBoost parameters
    param_candidates = [
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 300, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'max_depth': 5, 'learning_rate': 0.2, 'n_estimators': 150, 'subsample': 0.8, 'colsample_bytree': 0.8},
    ]
    
    best_score = 0
    best_params = param_candidates[0]
    
    for params in param_candidates:
        model = XGBClassifier(
            random_state=20261114,
            n_jobs=1,
            verbosity=-1,
            **params
        )
        scores = []
        for train_idx, val_idx in cv.split(train, y):
            X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            probas = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, probas))
        mean_score = float(sum(scores) / len(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    # Train final model with best hyperparameters
    model = XGBClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        **best_params
    )
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
