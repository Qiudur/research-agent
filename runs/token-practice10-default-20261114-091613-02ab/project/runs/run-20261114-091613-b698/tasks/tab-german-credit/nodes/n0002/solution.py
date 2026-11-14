"""LightGBM with label encoding for all categorical features."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder


def fit_predict(train_csv: str, test_rows: list[dict[str, str]]) -> list[float]:
    train = pd.read_csv(train_csv)
    y = train.pop("y").astype(int)
    
    # Create test DataFrame preserving column order
    test = pd.DataFrame(test_rows, columns=train.columns)
    
    # Identify numeric vs categorical columns based on original dtype
    numeric_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    categorical_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    
    # Ensure numeric columns are properly typed (handles string-formatted numbers)
    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
    
    # Prepare numeric features
    X_train_num = train[numeric_cols].copy()
    X_test_num = test[numeric_cols].copy()
    
    # Process categorical features with label encoding
    X_train_cat_list = []
    X_test_cat_list = []
    
    for col in categorical_cols:
        le = LabelEncoder()
        train_col = train[col].astype(str)
        test_col = test[col].astype(str)
        
        # Fit on train, transform both
        train_encoded = le.fit_transform(train_col)
        
        # Transform test, assigning -1 for unseen categories
        test_encoded = test_col.apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        
        X_train_cat_list.append(pd.Series(train_encoded, name=col))
        X_test_cat_list.append(pd.Series(test_encoded, name=col))
    
    # Combine features
    X_train = X_train_num.copy()
    X_test = X_test_num.copy()
    
    if X_train_cat_list:
        X_train_cat = pd.concat(X_train_cat_list, axis=1)
        X_test_cat = pd.concat(X_test_cat_list, axis=1)
        X_train = pd.concat([X_train, X_train_cat], axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    
    # Fit and predict
    model = LGBMClassifier(random_state=20261114, n_jobs=1, verbosity=-1)
    model.fit(X_train, y)
    return model.predict_proba(X_test)[:, 1].tolist()
