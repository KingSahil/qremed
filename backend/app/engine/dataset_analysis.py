"""
dataset_analysis.py
====================
Generalized dataset inspection for ANY uploaded CSV (not WDBC-specific).

Replaces the old data_loader.py, which only knew how to load the sklearn
breast-cancer bunch. This module inspects whatever CSV + target column the
user gives it and answers the same questions the original project asked
of WDBC, but computed rather than assumed.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

ID_COLUMN_PATTERN = re.compile(r"(^id$|_id$|^index$|^unnamed)", re.IGNORECASE)


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Flag columns that look like row identifiers rather than features:
    name matches an id-like pattern, OR the column is integer-valued AND
    (almost) every value is unique. Continuous float measurements are
    NEVER flagged by uniqueness alone -- real sensor/measurement data is
    very often >98% unique across rows without being an identifier."""
    candidates = []
    for col in df.columns:
        name_hit = bool(ID_COLUMN_PATTERN.search(str(col)))

        is_integer_like = False
        if pd.api.types.is_integer_dtype(df[col]):
            is_integer_like = True
        elif pd.api.types.is_float_dtype(df[col]):
            non_null = df[col].dropna()
            is_integer_like = len(non_null) > 0 and bool(np.all(np.equal(np.mod(non_null, 1), 0)))

        unique_ratio = df[col].nunique(dropna=False) / max(len(df), 1)
        looks_sequential = is_integer_like and unique_ratio > 0.98

        if name_hit or looks_sequential:
            candidates.append(col)
    return candidates


def classify_columns(df: pd.DataFrame, target_column: str) -> dict:
    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols, categorical_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def validate_dataset(df: pd.DataFrame, target_column: str) -> dict:
    """Checks whether this dataset is currently supported by Q-REMED.
    Returns {"supported": bool, "reasons": [...]} instead of crashing."""
    reasons = []
    if target_column not in df.columns:
        return {"supported": False, "reasons": [f"Target column '{target_column}' not found in the uploaded CSV."]}

    n_classes = df[target_column].nunique(dropna=True)
    if n_classes != 2:
        reasons.append(
            f"Q-REMED currently supports binary classification only. "
            f"The target column '{target_column}' has {n_classes} distinct classes."
        )

    cols = classify_columns(df, target_column)
    if len(cols["numeric"]) == 0:
        reasons.append("No numeric feature columns were found. Q-REMED currently requires primarily numerical features.")

    if len(df) < 30:
        reasons.append(f"Only {len(df)} rows found. At least ~30 rows are recommended for a stratified train/test split and any statistical feature ranking to be meaningful.")

    return {"supported": len(reasons) == 0, "reasons": reasons}


def analyze_dataset(df: pd.DataFrame, target_column: str) -> dict:
    """Full dataset overview, computed from the actual uploaded data."""
    n_samples, n_cols = df.shape
    id_cols = detect_id_columns(df.drop(columns=[target_column]) if target_column in df.columns else df)
    col_types = classify_columns(df, target_column) if target_column in df.columns else {"numeric": [], "categorical": []}

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    overview = {
        "n_samples": int(n_samples),
        "n_columns": int(n_cols),
        "feature_names": [c for c in df.columns if c != target_column],
        "numeric_columns": col_types["numeric"],
        "categorical_columns": col_types["categorical"],
        "possible_id_columns": id_cols,
        "missing_values_total": missing_values,
        "missing_values_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": duplicate_rows,
        "target_column": target_column,
    }

    if target_column in df.columns:
        vc = df[target_column].value_counts(dropna=True)
        n_classes = int(df[target_column].nunique(dropna=True))
        overview.update({
            "n_classes": n_classes,
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "class_balance_pct": {str(k): round(100 * v / n_samples, 2) for k, v in vc.items()},
        })

    overview["validation"] = validate_dataset(df, target_column)
    return overview


def csv_preview(df: pd.DataFrame, n_rows: int = 10) -> dict:
    preview = df.head(n_rows).copy()
    # Make JSON-safe
    for c in preview.columns:
        if pd.api.types.is_float_dtype(preview[c]):
            preview[c] = preview[c].round(4)
    return {"columns": list(preview.columns), "rows": preview.astype(object).where(pd.notnull(preview), None).values.tolist()}
