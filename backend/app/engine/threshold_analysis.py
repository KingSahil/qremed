"""
threshold_analysis.py
=======================
Validation-only decision threshold selection (new module -- generalizes
what the original Phase 2C notebook did ad hoc into a reusable engine
step for any classical or quantum model).

Method
------
1. Split the TRAINING split further into train2/val (stratified, fixed
   seed) -- the test set is never touched by threshold selection.
2. Re-fit the model on train2 only, score the val split.
3. Search thresholds in (0, 1) for the one that best satisfies the
   requested objective on the VALIDATION scores.
4. Report the metrics that threshold produces on the untouched TEST set
   (scored using the model refit on the full training split), for
   comparison against the default 0.5 threshold.

A decision threshold is not automatically a clinically calibrated
probability -- this is only a decision boundary tuned on held-out data.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .evaluation import evaluate_predictions

OBJECTIVES = ["maximize_f1", "prioritize_sensitivity", "prioritize_specificity"]


def _metrics_at_threshold(y_true, y_score, threshold, positive_label, negative_label):
    y_pred = np.where(y_score >= threshold, positive_label, negative_label)
    result = evaluate_predictions(y_true, y_pred, y_score, positive_label, negative_label)
    return result["metrics"]


def select_threshold(y_val, y_val_score, objective: str, positive_label: int, negative_label: int):
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_score = 0.5, -np.inf

    for t in candidate_thresholds:
        m = _metrics_at_threshold(y_val, y_val_score, t, positive_label, negative_label)
        if objective == "maximize_f1":
            score = m["f1"]
        elif objective == "prioritize_sensitivity":
            # maximize sensitivity, tie-broken by specificity, subject to specificity staying >= 0.5
            score = m["sensitivity"] + (0.001 * m["specificity"]) if m["specificity"] >= 0.5 else -1
        elif objective == "prioritize_specificity":
            score = m["specificity"] + (0.001 * m["sensitivity"]) if m["sensitivity"] >= 0.5 else -1
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if score > best_score:
            best_score, best_t = score, t

    return float(best_t)


def run_threshold_analysis(model_factory, X_train, y_train, X_test, y_test,
                            objective: str, positive_label: int, negative_label: int,
                            random_state: int = 42):
    """model_factory: zero-arg callable returning a fresh, unfit sklearn-style
    model (so it can be safely refit on train2 and again on full X_train)."""

    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train,
    )

    val_model = model_factory()
    val_model.fit(X_tr2, y_tr2)
    val_proba = val_model.predict_proba(X_val)
    pos_idx_val = list(val_model.classes_).index(positive_label)
    y_val_score = val_proba[:, pos_idx_val]

    selected_threshold = select_threshold(np.asarray(y_val), y_val_score, objective, positive_label, negative_label)

    # Refit on the full training split for the final, reportable test-set numbers.
    full_model = model_factory()
    full_model.fit(X_train, y_train)
    test_proba = full_model.predict_proba(X_test)
    pos_idx_test = list(full_model.classes_).index(positive_label)
    y_test_score = test_proba[:, pos_idx_test]

    default_metrics = _metrics_at_threshold(np.asarray(y_test), y_test_score, 0.5, positive_label, negative_label)
    selected_metrics = _metrics_at_threshold(np.asarray(y_test), y_test_score, selected_threshold, positive_label, negative_label)

    return {
        "objective": objective,
        "default_threshold": 0.5,
        "selected_threshold": round(selected_threshold, 3),
        "optimal_threshold": round(selected_threshold, 3),
        "default_threshold_metrics": default_metrics,
        "default_metrics": default_metrics,
        "selected_threshold_metrics": selected_metrics,
        "optimal_metrics": selected_metrics,
        "selected_on": "validation split (25% of training data, held out from both model fitting and test evaluation)",
        "warning": "A decision threshold is not automatically a clinically calibrated probability.",
    }
