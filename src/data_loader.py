"""
data_loader.py
===============
Loads the Breast Cancer Wisconsin (Diagnostic) dataset via sklearn and
produces a verified dataset report. Nothing here alters the data --
this module only inspects and reports.
"""

import json
import os

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_dataset():
    """Load the dataset and return (X: DataFrame, y: Series, target_names, feature_names)."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    target_names = list(bunch.target_names)
    feature_names = list(bunch.feature_names)
    return X, y, target_names, feature_names


def build_dataset_report(X, y, target_names, feature_names):
    """
    Inspect the loaded data and answer the required dataset-report
    questions. Verifies assumptions rather than asserting them from memory.
    """
    n_samples, n_features = X.shape
    class_counts = y.value_counts().sort_index()
    missing_values = int(X.isna().sum().sum())
    duplicate_rows = int(X.duplicated().sum())

    # Verify malignant/benign encoding rather than assuming it.
    # sklearn's target_names is ordered by the integer label: index 0 -> label 0.
    label_to_name = {i: name for i, name in enumerate(target_names)}

    feature_ranges = {
        col: {"min": float(X[col].min()), "max": float(X[col].max())}
        for col in X.columns
    }

    report = {
        "row_meaning": (
            "Each row is one fine-needle-aspirate (FNA) breast mass biopsy, "
            "summarized into 30 numeric features describing cell-nucleus "
            "shape and texture."
        ),
        "target_meaning": "Binary diagnosis: malignant vs benign tumor.",
        "label_to_class_name": label_to_name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "feature_names": feature_names,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_balance_pct": {
            str(k): round(100 * v / n_samples, 2) for k, v in class_counts.items()
        },
        "missing_values_total": missing_values,
        "duplicate_rows": duplicate_rows,
        "feature_ranges": feature_ranges,
        "preprocessing_required": (
            "Standardization (features span very different numeric scales, "
            "e.g. 'mean area' in the hundreds vs 'mean smoothness' in the "
            "hundredths), stratified train/test split, and a duplicate-row "
            "check before any downstream step."
        ),
    }
    return report


def save_dataset_report(report, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "dataset_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_dataset_report(report):
    print("=" * 70)
    print("DATASET REPORT: Breast Cancer Wisconsin (Diagnostic)")
    print("=" * 70)
    print(f"Row meaning:        {report['row_meaning']}")
    print(f"Target meaning:     {report['target_meaning']}")
    print(f"Label -> class:     {report['label_to_class_name']}")
    print(f"Samples:            {report['n_samples']}")
    print(f"Features:           {report['n_features']}")
    print(f"Class counts:       {report['class_counts']}")
    print(f"Class balance (%):  {report['class_balance_pct']}")
    print(f"Missing values:     {report['missing_values_total']}")
    print(f"Duplicate rows:     {report['duplicate_rows']}")
    print(f"Preprocessing req.: {report['preprocessing_required']}")
    print("=" * 70)
