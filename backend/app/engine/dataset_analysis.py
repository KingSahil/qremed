"""
dataset_analysis.py
====================
Generalized dataset inspection for ANY uploaded CSV (not WDBC-specific).

Supports all datatypes:
- Binary targets: native 2-class classification.
- Continuous numeric targets (e.g. age with 47 unique values): binarized via
  median split, mean split, or custom threshold.
- Multiclass targets: binarized via One-vs-Rest (OvR).
- All feature datatypes: numeric, boolean, datetime, and categorical (encoded in preprocessing).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

ID_COLUMN_PATTERN = re.compile(r"(^id$|_id$|^index$|^unnamed)", re.IGNORECASE)

TARGET_NAME_PATTERNS = [
    re.compile(r"death[_\s-]?event", re.IGNORECASE),
    re.compile(r"(^target$|^label$|^class$|^diagnosis$|^outcome$|^status$|^condition$|^y$)", re.IGNORECASE),
    re.compile(r"(survived|mortality|disease|recurrence|cancer|relapse|event)", re.IGNORECASE),
]


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


def detect_target_candidate(df: pd.DataFrame) -> str:
    """Intelligently detects the most likely target column:
    1. Keyword match on target names (e.g. DEATH_EVENT, diagnosis, target, label).
    2. Any binary column (2 unique non-null values), checking from the last column forward.
    3. Last column of the CSV (standard machine learning CSV convention).
    """
    if len(df.columns) == 0:
        return ""

    cols = list(df.columns)

    # 1. Search for target name patterns
    for pat in TARGET_NAME_PATTERNS:
        for c in reversed(cols):
            if pat.search(str(c)):
                return str(c)

    # 2. Check if last column or any column is binary (exactly 2 unique values)
    if df[cols[-1]].nunique(dropna=True) == 2:
        return str(cols[-1])

    for c in reversed(cols):
        if df[c].nunique(dropna=True) == 2:
            return str(c)

    # 3. Default to last column (standard for datasets)
    return str(cols[-1])


def get_column_metadata(df: pd.DataFrame) -> list[dict]:
    """Generates descriptive metadata for all columns in the dataset."""
    id_cols = set(detect_id_columns(df))
    candidate_target = detect_target_candidate(df)
    meta = []

    for c in df.columns:
        series = df[c]
        nunique = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        if c in id_cols:
            cat = "id"
        elif nunique == 2:
            cat = "binary"
        elif pd.api.types.is_numeric_dtype(series):
            cat = "continuous" if nunique > 2 else "binary"
        elif pd.api.types.is_bool_dtype(series):
            cat = "binary"
        else:
            cat = "categorical"

        samples = [str(x) for x in series.dropna().unique()[:3].tolist()]

        meta.append({
            "name": str(c),
            "dtype": dtype_str,
            "category": cat,
            "n_unique": nunique,
            "sample_values": samples,
            "missing_count": int(series.isna().sum()),
            "is_candidate_target": bool(c == candidate_target),
        })

    return meta


def classify_columns(df: pd.DataFrame, target_column: str) -> dict:
    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols, categorical_cols, boolean_cols = [], [], []

    for c in feature_cols:
        series = df[c]
        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(c)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "boolean": boolean_cols,
        "all_features": feature_cols,
    }


def get_target_nature(series: pd.Series) -> str:
    """Returns 'binary', 'continuous', 'multiclass', or 'constant'."""
    nunique = series.nunique(dropna=True)
    if nunique <= 1:
        return "constant"
    if nunique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "multiclass"


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    binarize_strategy: Optional[str] = None,
    binarize_threshold: Optional[float] = None,
) -> dict:
    """Checks whether this dataset is supported by Q-REMED.
    Supports ALL datatypes:
    - Binary targets: native
    - Continuous numeric targets: supported via median/mean/custom binarization
    - Multiclass targets: supported via One-vs-Rest binarization
    - Categorical & numeric features: supported via leakage-safe encoding
    """
    reasons = []
    notices = []

    if target_column not in df.columns:
        return {"supported": False, "reasons": [f"Target column '{target_column}' not found in the uploaded CSV."]}

    series = df[target_column]
    nature = get_target_nature(series)
    nunique = int(series.nunique(dropna=True))

    if nature == "constant":
        reasons.append(
            f"The target column '{target_column}' has only 1 distinct value. "
            f"A classifier requires at least 2 distinct classes."
        )
    elif nature == "continuous":
        med = float(series.median())
        notices.append(
            f"Target column '{target_column}' is continuous numeric with {nunique} distinct values. "
            f"Q-REMED automatically binarizes this target (default: median split >= {med:g}) into high/low classes."
        )
    elif nature == "multiclass":
        notices.append(
            f"Target column '{target_column}' has {nunique} categorical classes. "
            f"Q-REMED supports this via One-vs-Rest (OvR) binarization."
        )

    cols = classify_columns(df, target_column)
    if len(cols["all_features"]) == 0:
        reasons.append("No feature columns were found in the uploaded CSV.")

    if len(df) < 15:
        reasons.append(f"Only {len(df)} rows found. At least 15 rows are required for train/test split.")

    return {
        "supported": len(reasons) == 0,
        "reasons": reasons,
        "notices": notices,
        "target_nature": nature,
    }


def analyze_dataset(
    df: pd.DataFrame,
    target_column: str,
    binarize_strategy: Optional[str] = None,
    binarize_threshold: Optional[float] = None,
    positive_class: Optional[str] = None,
) -> dict:
    """Full dataset overview computed from the actual uploaded data,
    supporting all target and feature datatypes."""
    n_samples, n_cols = df.shape
    id_cols = detect_id_columns(df.drop(columns=[target_column]) if target_column in df.columns else df)
    col_types = classify_columns(df, target_column) if target_column in df.columns else {
        "numeric": [], "categorical": [], "boolean": [], "all_features": []
    }

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    overview = {
        "n_samples": int(n_samples),
        "n_columns": int(n_cols),
        "feature_names": [c for c in df.columns if c != target_column],
        "numeric_columns": col_types["numeric"],
        "categorical_columns": col_types["categorical"],
        "boolean_columns": col_types.get("boolean", []),
        "possible_id_columns": id_cols,
        "missing_values_total": missing_values,
        "missing_values_by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": duplicate_rows,
        "target_column": target_column,
        "columns_metadata": get_column_metadata(df),
        "suggested_target": detect_target_candidate(df),
    }

    if target_column in df.columns:
        series = df[target_column]
        nature = get_target_nature(series)
        nunique = int(series.nunique(dropna=True))

        target_info = {
            "target_column": target_column,
            "dtype": str(series.dtype),
            "target_nature": nature,
            "original_n_classes": nunique,
            "is_binarized": False,
        }

        if nature == "continuous":
            med = float(series.median())
            mean_val = float(series.mean())
            min_val = float(series.min())
            max_val = float(series.max())

            strategy = (binarize_strategy or "median").lower()
            if strategy == "mean":
                threshold = mean_val
            elif strategy == "custom" and binarize_threshold is not None:
                threshold = float(binarize_threshold)
            else:
                strategy = "median"
                threshold = med

            pos_mask = (series >= threshold)
            pos_cnt = int(pos_mask.sum())
            neg_cnt = int((~pos_mask).sum())

            pos_name = f">= {threshold:g}"
            neg_name = f"< {threshold:g}"

            class_counts = {neg_name: neg_cnt, pos_name: pos_cnt}
            class_balance_pct = {
                neg_name: round(100 * neg_cnt / max(n_samples, 1), 2),
                pos_name: round(100 * pos_cnt / max(n_samples, 1), 2),
            }

            target_info.update({
                "is_binarized": True,
                "strategy": strategy,
                "threshold": round(threshold, 4),
                "median": round(med, 4),
                "mean": round(mean_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "positive_label_name": pos_name,
                "negative_label_name": neg_name,
            })

            overview.update({
                "n_classes": 2,
                "class_counts": class_counts,
                "class_balance_pct": class_balance_pct,
                "target_info": target_info,
            })

        elif nature == "multiclass":
            vc = series.value_counts(dropna=True)
            all_classes = [str(k) for k in vc.index]
            pos_choice = str(positive_class) if positive_class and str(positive_class) in all_classes else all_classes[0]

            pos_mask = (series.astype(str) == pos_choice)
            pos_cnt = int(pos_mask.sum())
            neg_cnt = int((~pos_mask).sum())

            pos_name = pos_choice
            neg_name = f"Other (not {pos_choice})"

            class_counts = {neg_name: neg_cnt, pos_name: pos_cnt}
            class_balance_pct = {
                neg_name: round(100 * neg_cnt / max(n_samples, 1), 2),
                pos_name: round(100 * pos_cnt / max(n_samples, 1), 2),
            }

            target_info.update({
                "is_binarized": True,
                "strategy": "ovr",
                "available_classes": all_classes,
                "positive_label_name": pos_name,
                "negative_label_name": neg_name,
            })

            overview.update({
                "n_classes": 2,
                "class_counts": class_counts,
                "class_balance_pct": class_balance_pct,
                "target_info": target_info,
            })

        else:  # binary
            vc = series.value_counts(dropna=True)
            overview.update({
                "n_classes": nunique,
                "class_counts": {str(k): int(v) for k, v in vc.items()},
                "class_balance_pct": {str(k): round(100 * v / max(n_samples, 1), 2) for k, v in vc.items()},
                "target_info": target_info,
            })

    overview["validation"] = validate_dataset(df, target_column, binarize_strategy, binarize_threshold)
    return overview


def csv_preview(df: pd.DataFrame, n_rows: int = 10) -> dict:
    preview = df.head(n_rows).copy()
    for c in preview.columns:
        if pd.api.types.is_float_dtype(preview[c]):
            preview[c] = preview[c].round(4)
    return {
        "columns": list(preview.columns),
        "rows": preview.astype(object).where(pd.notnull(preview), None).values.tolist(),
        "columns_metadata": get_column_metadata(df),
        "suggested_target": detect_target_candidate(df),
    }
