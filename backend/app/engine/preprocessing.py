"""
preprocessing.py
=================
Generalized, leakage-safe preprocessing for an arbitrary uploaded dataset.

Preserves the original Q-REMED Phase 1 order exactly:

    raw data
      -> drop detected ID columns, keep numeric feature columns
      -> encode target to {0, 1}
      -> stratified train/test split (fixed seed)
      -> impute missing values using TRAIN statistics only
      -> fit StandardScaler on TRAIN only
      -> transform TRAIN and TEST with that fitted scaler

Nothing downstream (feature ranking, threshold selection, quantum scaling)
is allowed to look at X_test/y_test until final evaluation.
"""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset_analysis import classify_columns, detect_id_columns


def select_usable_features(df: pd.DataFrame, target_column: str) -> dict:
    """Drop ID-like columns and non-numeric columns (with an explicit
    explanation of what was dropped and why)."""
    id_cols = detect_id_columns(df.drop(columns=[target_column]))
    cols = classify_columns(df, target_column)
    dropped_categorical = cols["categorical"]
    usable = [c for c in cols["numeric"] if c not in id_cols]
    return {
        "usable_features": usable,
        "dropped_id_columns": id_cols,
        "dropped_categorical_columns": dropped_categorical,
    }


def encode_target(y_raw: pd.Series, positive_class: "str | int | None" = None):
    """Encode the target to {0, 1}. LabelEncoder sorts classes, so by
    default class index 1 (the class that sorts second) is treated as the
    'positive' class -- unless the caller explicitly names a positive
    class (e.g. from the dataset-analysis step)."""
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    classes = list(le.classes_)

    if positive_class is not None and str(positive_class) in [str(c) for c in classes]:
        pos_idx = [str(c) for c in classes].index(str(positive_class))
    else:
        pos_idx = 1 if len(classes) > 1 else 0

    label_map = {int(i): str(c) for i, c in enumerate(classes)}
    return y_enc, label_map, pos_idx


def leakage_safe_preprocess(df: pd.DataFrame, target_column: str, test_size: float,
                             random_state: int, positive_class=None):
    usable = select_usable_features(df, target_column)
    feature_cols = usable["usable_features"]
    if len(feature_cols) == 0:
        raise ValueError("No usable numeric feature columns remain after dropping ID/categorical columns.")

    X = df[feature_cols].copy()
    y_raw = df[target_column]
    y_enc, label_map, positive_label = encode_target(y_raw, positive_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc,
    )

    # Impute using TRAIN statistics only (median), applied to both splits.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_cols, index=X_test.index)

    negative_label = 1 - positive_label
    info = {
        "feature_columns": feature_cols,
        "dropped_id_columns": usable["dropped_id_columns"],
        "dropped_categorical_columns": usable["dropped_categorical_columns"],
        "label_map": label_map,
        "positive_label": int(positive_label),
        "positive_class_name": label_map.get(int(positive_label)),
        "negative_label": int(negative_label),
        "negative_class_name": label_map.get(int(negative_label)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": test_size,
        "random_state": random_state,
        "imputation": "median (fit on training data only)",
        "scaling": "StandardScaler (fit on training data only)",
        "message": "Leakage-safe preprocessing enabled",
    }

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "imputer": imputer, "info": info,
    }
