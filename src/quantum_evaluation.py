"""
quantum_evaluation.py
=======================
Evaluates a TRAINED, FROZEN VQC on the untouched Phase 1 test set.

VQC (unlike sklearn classifiers) does not expose a `.classes_` attribute,
so the predict_proba column order is not looked up dynamically -- it is
verified once, explicitly, against a synthetic well-separated dataset
(see project notes / decision log) to confirm column order follows
ascending sorted label order: column 0 = P(label 0), column 1 =
P(label 1). For this project label 0 = malignant, label 1 = benign, so
column 0 is the malignant (positive-class) probability.
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

MALIGNANT_LABEL = 0
BENIGN_LABEL = 1
MALIGNANT_PROBA_COLUMN = 0  # verified: predict_proba column order = ascending label order


def evaluate_vqc(vqc, X_test, y_test):
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    y_pred = vqc.predict(X_test_arr)
    y_proba = vqc.predict_proba(X_test_arr)
    y_score_malignant = y_proba[:, MALIGNANT_PROBA_COLUMN]

    cm = confusion_matrix(y_test_arr, y_pred, labels=[MALIGNANT_LABEL, BENIGN_LABEL])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test_arr, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_test_arr, y_pred, pos_label=MALIGNANT_LABEL),
        "f1": f1_score(y_test_arr, y_pred, pos_label=MALIGNANT_LABEL),
        "roc_auc": roc_auc_score(y_test_arr == MALIGNANT_LABEL, y_score_malignant),
    }

    fpr, tpr, thresholds = roc_curve(y_test_arr == MALIGNANT_LABEL, y_score_malignant)

    prediction_distribution = {
        "predicted_malignant": int(np.sum(y_pred == MALIGNANT_LABEL)),
        "predicted_benign": int(np.sum(y_pred == BENIGN_LABEL)),
        "actual_malignant": int(np.sum(y_test_arr == MALIGNANT_LABEL)),
        "actual_benign": int(np.sum(y_test_arr == BENIGN_LABEL)),
    }

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "positive_label": MALIGNANT_LABEL, "negative_label": BENIGN_LABEL,
        },
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
        "prediction_distribution": prediction_distribution,
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
    }
