"""
quantum_evaluation.py
=======================
Evaluates a trained, frozen VQC on the untouched test set. Generalized:
the positive label / column order is taken from preprocessing.py's
verified `positive_label`, never hard-coded to "malignant = 0".

VQC's predict_proba column order follows ascending sorted label order
(column i = P(label i)) -- this is a property of the underlying
qiskit-machine-learning classifier, verified against its documented
behavior, not assumed per-dataset.
"""
from __future__ import annotations

import numpy as np

from .evaluation import evaluate_predictions


def evaluate_vqc(vqc, X_test, y_test, positive_label: int, negative_label: int) -> dict:
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    y_pred = vqc.predict(X_test_arr)
    y_proba = vqc.predict_proba(X_test_arr)
    y_score_positive = y_proba[:, positive_label]

    result = evaluate_predictions(y_test_arr, y_pred, y_score_positive, positive_label, negative_label)
    result["predictions"] = np.asarray(y_pred).tolist()
    result["probabilities"] = y_proba.tolist()
    result["prediction_distribution"] = {
        "predicted_positive": int(np.sum(y_pred == positive_label)),
        "predicted_negative": int(np.sum(y_pred == negative_label)),
        "actual_positive": int(np.sum(y_test_arr == positive_label)),
        "actual_negative": int(np.sum(y_test_arr == negative_label)),
    }
    return result
