"""LightGBM with target encoding for high-cardinality categoricals."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    """Predict income > $50K using LightGBM with target encoding."""
    
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    test = pd.DataFrame(test_rows)
    
    # Align numeric columns
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            test[name] = pd.to_numeric(test.get(name, ""), errors="coerce").astype(float)
        else:
            train[name] = train[name].astype(str)
            test[name] = test.get(name, "").astype(str)
    
    # High cardinality categorical columns
    high_card_cols = ["native_country", "occupation", "workclass"]
    high_card_cols = [c for c in high_card_cols if c in train.columns and train[c].nunique() >= 5]
    
    # Global mean for smoothing
    global_mean = float(y.mean())
    
    def encode_column(df, col, stats=None):
        """Target-encode a categorical column with Laplace smoothing."""
        if stats is None:
            agg = df.groupby(col)["_target"].agg(["sum", "count"]).to_dict("index")
            result = {}
            for cat in df[col].unique():
                s = agg[cat]["sum"] if cat in agg else 0
                c = agg[cat]["count"] if cat in agg else 0
                result[cat] = (s + global_mean * 30) / (c + 30)
            return result
        else:
            return {cat: v for cat, v in stats.items()}
    
    # Compute encoders from training data
    train_temp = train.copy()
    train_temp["_target"] = y
    encoders = {}
    for col in high_card_cols:
        encoders[col] = encode_column(train_temp, col)
    del train_temp["_target"]
    
    # Apply encoding to train and test
    X_train = train.copy()
    X_test = test.copy()
    for col, encoder in encoders.items():
        X_train[f"{col}_enc"] = X_train[col].map(encoder).fillna(global_mean)
        X_test[f"{col}_enc"] = X_test[col].map(encoder).fillna(global_mean)
    
    # Prepare features
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    enc_cols = [c for c in num_cols if c.endswith("_enc")]
    feature_cols = [c for c in num_cols if not c.endswith("_enc")] + enc_cols
    
    X_train_f = X_train[feature_cols].fillna(0)
    X_test_f = X_test[feature_cols].fillna(0)
    
    # Fit LightGBM and predict
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(X_train_f, y)
    return model.predict_proba(X_test_f)[:, 1].tolist()
