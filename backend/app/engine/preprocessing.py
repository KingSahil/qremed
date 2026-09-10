"""
preprocessing.py
=================
Generalized, leakage-safe preprocessing for arbitrary uploaded datasets.

Supports all datatypes:
- Binary targets: native label encoding.
- Continuous targets (e.g. age): binarized via median/mean/custom split.
- Multiclass targets: binarized via One-vs-Rest (OvR).
- Categorical features: encoded via OneHotEncoder/OrdinalEncoder (fit strictly on train).
- Boolean features: converted to 0/1 integers.
- Formatted numeric strings: parsed to clean floats.
- Datetime features: parsed to year/month/day/dayofweek components.

Preserves the Q-REMED Phase 1 order:
    raw data
      -> drop detected ID columns, retain all informative features
      -> clean booleans, parse numeric strings, extract date parts
      -> encode/binarize target to {0, 1}
      -> stratified train/test split (fixed seed)
      -> impute missing values using TRAIN statistics only
      -> encode categorical columns using TRAIN statistics only
      -> fit StandardScaler on TRAIN only
      -> transform TRAIN and TEST with that fitted scaler
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from .dataset_analysis import detect_id_columns, get_target_nature


def clean_feature_series(s: pd.Series) -> pd.Series:
    """Cleans an individual feature series: converts booleans, parses formatted
    numeric strings (e.g. '$1,200.50' or '95%'), leaving true categoricals intact."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        # Check if boolean-like strings
        lower_vals = s.dropna().astype(str).str.strip().str.lower()
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0, "t": 1, "f": 0, "y": 1, "n": 0}
        if len(lower_vals) > 0 and lower_vals.isin(bool_map.keys()).all():
            return s.astype(str).str.strip().str.lower().map(bool_map).astype(float)

        # Check if numeric string with commas/currency/percentages
        cleaned = s.astype(str).str.replace(r"[$,% ]", "", regex=True)
        numeric_attempt = pd.to_numeric(cleaned, errors="coerce")
        # If >=90% of non-null values convert to numbers, treat as numeric
        non_null_cnt = s.notna().sum()
        if non_null_cnt > 0 and (numeric_attempt.notna().sum() / non_null_cnt) >= 0.9:
            return numeric_attempt

    return s


def expand_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempts to parse string/object columns as datetimes and expand them."""
    expanded = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt = df[col]
            expanded[f"{col}_year"] = dt.dt.year
            expanded[f"{col}_month"] = dt.dt.month
            expanded[f"{col}_day"] = dt.dt.day
            expanded[f"{col}_dayofweek"] = dt.dt.dayofweek
            expanded.drop(columns=[col], inplace=True)
        elif pd.api.types.is_object_dtype(df[col]):
            # Try to see if this column is date-formatted
            sample = df[col].dropna().head(20).astype(str)
            date_like = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").all() or sample.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}").all()
            if date_like and len(sample) > 0:
                dt_attempt = pd.to_datetime(df[col], errors="coerce")
                if dt_attempt.notna().sum() / max(len(sample), 1) >= 0.8:
                    expanded[f"{col}_year"] = dt_attempt.dt.year
                    expanded[f"{col}_month"] = dt_attempt.dt.month
                    expanded[f"{col}_day"] = dt_attempt.dt.day
                    expanded[f"{col}_dayofweek"] = dt_attempt.dt.dayofweek
                    expanded.drop(columns=[col], inplace=True)
    return expanded


def select_usable_features(df: pd.DataFrame, target_column: str) -> dict:
    """Drop ID-like columns while keeping ALL informative feature columns
    (both numeric and categorical)."""
    feature_df = df.drop(columns=[target_column]) if target_column in df.columns else df
    id_cols = detect_id_columns(feature_df)
    remaining_cols = [c for c in feature_df.columns if c not in id_cols]

    numeric_cols = []
    categorical_cols = []

    for c in remaining_cols:
        s = clean_feature_series(feature_df[c])
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return {
        "usable_features": remaining_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dropped_id_columns": id_cols,
        "dropped_categorical_columns": [],  # We do not drop categorical columns anymore!
    }


def encode_target(
    y_raw: pd.Series,
    positive_class: "str | int | None" = None,
    binarize_strategy: Optional[str] = None,
    binarize_threshold: Optional[float] = None,
):
    """Encode any target datatype to {0, 1}:
    - Binary targets: standard LabelEncoder.
    - Continuous numeric targets (e.g. age with 47 unique values):
      Binarized using median split (default), mean split, or custom threshold.
    - Multiclass targets: One-vs-Rest (OvR) binarization.
    """
    nature = get_target_nature(y_raw)
    nunique = y_raw.nunique(dropna=True)

    if nunique <= 1:
        raise ValueError("Target column must have at least 2 distinct values.")

    # 1. Continuous numeric target -> Binarization
    if nature == "continuous":
        strategy = (binarize_strategy or "median").lower()
        if strategy == "mean":
            thresh = float(y_raw.mean())
        elif strategy == "custom" and binarize_threshold is not None:
            thresh = float(binarize_threshold)
        else:
            strategy = "median"
            thresh = float(y_raw.median())

        pos_mask = (y_raw >= thresh)
        # Ensure both classes have at least 1 sample
        if pos_mask.all() or (~pos_mask).all():
            thresh = float(y_raw.mean())
            pos_mask = (y_raw >= thresh)

        y_enc = pd.Series(pos_mask.astype(int), index=y_raw.index)
        pos_idx = 1
        label_map = {0: f"< {thresh:g}", 1: f">= {thresh:g}"}
        return y_enc, label_map, pos_idx

    # 2. Multiclass categorical target -> One-vs-Rest (OvR)
    if nature == "multiclass":
        vc = y_raw.value_counts(dropna=True)
        all_classes = [str(k) for k in vc.index]
        pos_choice = str(positive_class) if positive_class and str(positive_class) in all_classes else all_classes[0]

        pos_mask = (y_raw.astype(str) == pos_choice)
        y_enc = pd.Series(pos_mask.astype(int), index=y_raw.index)
        pos_idx = 1
        label_map = {0: f"Other (not {pos_choice})", 1: str(pos_choice)}
        return y_enc, label_map, pos_idx

    # 3. Native binary target
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    classes = list(le.classes_)

    if positive_class is not None and str(positive_class) in [str(c) for c in classes]:
        pos_idx = [str(c) for c in classes].index(str(positive_class))
    else:
        pos_idx = 1 if len(classes) > 1 else 0

    label_map = {int(i): str(c) for i, c in enumerate(classes)}
    return y_enc, label_map, pos_idx


def leakage_safe_preprocess(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    positive_class: Optional[str] = None,
    binarize_strategy: Optional[str] = None,
    binarize_threshold: Optional[float] = None,
):
    """Leakage-safe preprocessing supporting all target and feature datatypes.
    Features: cleaned, datetime-expanded, imputed on train, encoded on train, scaled on train.
    Target: encoded/binarized to {0, 1}.
    """
    usable = select_usable_features(df, target_column)
    raw_feature_cols = usable["usable_features"]
    if len(raw_feature_cols) == 0:
        raise ValueError("No usable feature columns remain after dropping row identifiers.")

    X_raw = df[raw_feature_cols].copy()

    # Clean individual series (booleans, numeric strings)
    for c in X_raw.columns:
        X_raw[c] = clean_feature_series(X_raw[c])

    # Expand any datetime columns
    X_raw = expand_datetime_columns(X_raw)

    y_raw = df[target_column]
    y_enc, label_map, positive_label = encode_target(
        y_raw, positive_class=positive_class,
        binarize_strategy=binarize_strategy,
        binarize_threshold=binarize_threshold,
    )

    # Stratified train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc,
    )

    # Separate numeric vs categorical features
    numeric_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    # Process Numeric Columns: Impute median on train, apply to train & test
    train_parts = []
    test_parts = []

    num_imputer = None
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X_train_num = pd.DataFrame(
            num_imputer.fit_transform(X_train_raw[numeric_cols]),
            columns=numeric_cols, index=X_train_raw.index,
        )
        X_test_num = pd.DataFrame(
            num_imputer.transform(X_test_raw[numeric_cols]),
            columns=numeric_cols, index=X_test_raw.index,
        )
        train_parts.append(X_train_num)
        test_parts.append(X_test_num)

    # Process Categorical Columns: Impute most frequent, encode with OneHot/Ordinal
    cat_imputer = None
    cat_encoder = None
    encoded_feature_names = []

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat_imp = pd.DataFrame(
            cat_imputer.fit_transform(X_train_raw[categorical_cols].astype(str)),
            columns=categorical_cols, index=X_train_raw.index,
        )
        X_test_cat_imp = pd.DataFrame(
            cat_imputer.transform(X_test_raw[categorical_cols].astype(str)),
            columns=categorical_cols, index=X_test_raw.index,
        )

        # Use OneHotEncoder for low cardinality (<=15 unique), OrdinalEncoder for high cardinality
        low_card_cols = [c for c in categorical_cols if X_train_raw[c].nunique(dropna=True) <= 15]
        high_card_cols = [c for c in categorical_cols if c not in low_card_cols]

        if len(low_card_cols) > 0:
            cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_enc = cat_encoder.fit_transform(X_train_cat_imp[low_card_cols])
            test_enc = cat_encoder.transform(X_test_cat_imp[low_card_cols])
            feat_names = list(cat_encoder.get_feature_names_out(low_card_cols))

            train_parts.append(pd.DataFrame(train_enc, columns=feat_names, index=X_train_raw.index))
            test_parts.append(pd.DataFrame(test_enc, columns=feat_names, index=X_test_raw.index))
            encoded_feature_names.extend(feat_names)

        if len(high_card_cols) > 0:
            ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            train_ord = ord_encoder.fit_transform(X_train_cat_imp[high_card_cols])
            test_ord = ord_encoder.transform(X_test_cat_imp[high_card_cols])

            train_parts.append(pd.DataFrame(train_ord, columns=high_card_cols, index=X_train_raw.index))
            test_parts.append(pd.DataFrame(test_ord, columns=high_card_cols, index=X_test_raw.index))
            encoded_feature_names.extend(high_card_cols)

    # Concatenate all parts
    X_train_combined = pd.concat(train_parts, axis=1)
    X_test_combined = pd.concat(test_parts, axis=1)

    # Scale with StandardScaler fitted strictly on train
    scaler = StandardScaler()
    final_feature_names = list(X_train_combined.columns)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_combined), columns=final_feature_names, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_combined), columns=final_feature_names, index=X_test_raw.index)

    negative_label = 1 - positive_label
    info = {
        "feature_columns": final_feature_names,
        "numeric_features": numeric_cols,
        "encoded_categorical_columns": categorical_cols,
        "dropped_id_columns": usable["dropped_id_columns"],
        "dropped_categorical_columns": [],  # zero dropped!
        "label_map": label_map,
        "positive_label": int(positive_label),
        "positive_class_name": label_map.get(int(positive_label)),
        "negative_label": int(negative_label),
        "negative_class_name": label_map.get(int(negative_label)),
        "n_train": int(len(X_train_scaled)),
        "n_test": int(len(X_test_scaled)),
        "test_size": test_size,
        "random_state": random_state,
        "imputation": "median (fit on training data only)",
        "scaling": "StandardScaler (fit on training data only)",
        "message": f"Leakage-safe preprocessing enabled ({len(final_feature_names)} features ready)",
    }

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "imputer": num_imputer, "info": info,
    }
