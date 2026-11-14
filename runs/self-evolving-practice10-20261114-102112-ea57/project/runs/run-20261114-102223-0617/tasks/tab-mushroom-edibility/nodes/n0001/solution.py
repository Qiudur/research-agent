"""LightGBM with categorical feature encoding and missing-as-category handling."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

MISSING = "__missing__"


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    y = train.pop("y").astype(float).astype(int)
    test = pd.DataFrame(test_rows, columns=train.columns)
    for name in train.columns:
        tr = train[name].fillna("").astype(str).str.strip()
        te = test[name].fillna("").astype(str).str.strip()
        tr_num = pd.to_numeric(tr.replace("", None), errors="coerce")
        nonempty = tr != ""
        if nonempty.any() and tr_num[nonempty].notna().all():
            train[name] = tr_num.astype(float)
            test[name] = pd.to_numeric(te.replace("", None), errors="coerce").astype(float)
            continue
        tr = tr.replace("", MISSING).replace("?", MISSING)
        te = te.replace("", MISSING).replace("?", MISSING)
        categories = sorted(set(tr.unique()) | set(te.unique()))
        train[name] = pd.Categorical(tr, categories=categories)
        test[name] = pd.Categorical(te, categories=categories)
    model = LGBMClassifier(
        random_state=20261114,
        n_jobs=1,
        verbosity=-1,
        min_child_samples=5,
        min_data_per_group=5,
        cat_smooth=5,
    )
    model.fit(train, y)
    return [float(p) for p in model.predict_proba(test)[:, 1]]
