"""
evaluation.py
==============
Computes the full metric set (not accuracy alone) for a fitted classifier
against a held-out test set, plus confusion matrix and ROC curve data.

"Positive" class = malignant, matching the clinical framing (sensitivity =
fraction of actual malignant cases caught; specificity = fraction of
actual benign cases correctly cleared). The malignant label is passed in
explicitly (verified from the dataset report) rather than hardcoded here,
so this module never silently assumes an encoding.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, positive_label, negative_label):
    """
    Returns a dict of metrics plus raw confusion-matrix counts and ROC
    curve data, all computed on the untouched test set exactly once.
    """
    y_pred = model.predict(X_test)

    # Probability of the positive (malignant) class, for ROC-AUC.
    proba = model.predict_proba(X_test)
    pos_idx = list(model.classes_).index(positive_label)
    y_score = proba[:, pos_idx]

    # Confusion matrix ordered [positive, negative] x [positive, negative]
    # so TP/FN/FP/TN map directly onto sensitivity/specificity.
    cm = confusion_matrix(y_test, y_pred, labels=[positive_label, negative_label])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_test, y_pred, pos_label=positive_label),
        "f1": f1_score(y_test, y_pred, pos_label=positive_label),
        "roc_auc": roc_auc_score(y_test == positive_label, y_score),
    }

    fpr, tpr, thresholds = roc_curve(y_test == positive_label, y_score)

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "positive_label": int(positive_label),
            "negative_label": int(negative_label),
        },
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }
