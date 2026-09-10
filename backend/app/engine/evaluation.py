"""
evaluation.py
==============
Full metric set for a fitted sklearn-style classifier (predict +
predict_proba) against a held-out test set, generalized from the
original module: the positive label is always taken from the
leakage-safe-preprocessing step's `positive_label`, never hard-coded.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    roc_auc_score, roc_curve,
)


def evaluate_predictions(y_true, y_pred, y_score_positive, positive_label: int, negative_label: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, negative_label])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true == positive_label, y_score_positive)) if len(set(y_true)) > 1 else float("nan"),
    }

    fpr, tpr, thresholds = roc_curve(y_true == positive_label, y_score_positive)

    return {
        "metrics": metrics,
        "confusion_matrix": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                              "positive_label": int(positive_label), "negative_label": int(negative_label)},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
    }


def evaluate_sklearn_model(model, X_test, y_test, positive_label: int, negative_label: int) -> dict:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    pos_idx = list(model.classes_).index(positive_label)
    y_score = proba[:, pos_idx]
    result = evaluate_predictions(y_test, y_pred, y_score, positive_label, negative_label)
    result["predictions"] = np.asarray(y_pred).tolist()
    result["probabilities"] = proba.tolist()
    return result
