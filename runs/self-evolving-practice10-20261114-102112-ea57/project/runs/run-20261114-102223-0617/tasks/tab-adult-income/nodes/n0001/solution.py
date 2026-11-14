"""LightGBM with out-of-fold target encoding for high-cardinality categoricals and mode imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

TE_COLS = ["native_country", "workclass", "occupation"]
SMOOTH = 20.0


def _encode(keys: pd.Series, target: pd.Series, apply_to: pd.Series, prior: float) -> np.ndarray:
    stats = target.groupby(keys.values).agg(["sum", "count"])
    enc = (stats["sum"] + SMOOTH * prior) / (stats["count"] + SMOOTH)
    return apply_to.map(enc).fillna(prior).astype(float).values


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int).reset_index(drop=True)
    train = train.reset_index(drop=True)
    test = pd.DataFrame(test_rows, columns=train.columns)
    test = test.replace("", np.nan)

    te_cols = [c for c in TE_COLS if c in train.columns]
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test[name], errors="coerce")
        mode = train[name].mode(dropna=True)
        if len(mode):
            train[name] = train[name].fillna(mode.iloc[0])
            test[name] = test[name].fillna(mode.iloc[0])

    prior = float(y.mean())
    for name in te_cols:
        tr_keys = train[name].astype(str)
        te_keys = test[name].astype(str)
        oof = np.zeros(len(train))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261114)
        for fit_idx, val_idx in skf.split(train, y):
            oof[val_idx] = _encode(tr_keys.iloc[fit_idx], y.iloc[fit_idx], tr_keys.iloc[val_idx], prior)
        test[name] = _encode(tr_keys, y, te_keys, prior)
        train[name] = oof

    for name in train.columns:
        if name in te_cols or pd.api.types.is_numeric_dtype(train[name]):
            continue
        cats = pd.Categorical(pd.concat([train[name], test[name]], ignore_index=True).astype(str)).categories
        train[name] = pd.Categorical(train[name].astype(str), categories=cats)
        test[name] = pd.Categorical(test[name].astype(str), categories=cats)

    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(train, y)
    return model.predict_proba(test)[:, 1].tolist()
