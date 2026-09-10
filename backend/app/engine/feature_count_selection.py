"""
feature_count_selection.py
============================
Generalizes the original decision.py logic (which only ever compared a
fixed 4-vs-6-feature choice for WDBC) into "choose the smallest useful
feature representation" for ANY dataset and ANY candidate feature counts.

Method
------
For each candidate count k (from CANDIDATE_COUNTS, capped by the number
of available features and a practical quantum-circuit qubit ceiling):

  1. Take the top-k ANOVA-ranked features (training data only).
  2. Cross-validate a simple, fast classical model (Logistic Regression)
     on the TRAINING split only (never touches the test set) to get an
     unbiased estimate of validation performance at that feature count.

Then apply the project's declared bias: prefer the smallest k whose
cross-validated F1 is within DROP_TOLERANCE of the best F1 achieved by
any candidate count. This mirrors the original decision.py's
"don't blindly force 4 features, but bias toward smaller circuits unless
that costs meaningful performance" philosophy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .feature_selection import select_top_features

CANDIDATE_COUNTS = [4, 6, 8, 10]
MAX_QUBITS_PRACTICAL = 10  # ceiling for VQC/kernel simulation practicality
DROP_TOLERANCE_F1 = 0.02  # 2 percentage points


def evaluate_candidate_counts(X_train: pd.DataFrame, y_train: pd.Series,
                               anova_ranking_df: pd.DataFrame,
                               random_state: int = 42) -> dict:
    n_features_available = X_train.shape[1]
    candidates = [k for k in CANDIDATE_COUNTS if k <= n_features_available and k <= MAX_QUBITS_PRACTICAL]
    if not candidates:
        candidates = [min(n_features_available, MAX_QUBITS_PRACTICAL)]

    n_splits = min(5, int(y_train.value_counts().min()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    for k in candidates:
        feats = select_top_features(anova_ranking_df, k)
        Xk = X_train[feats]
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        f1_scores = cross_val_score(model, Xk, y_train, cv=cv, scoring="f1")
        acc_scores = cross_val_score(model, Xk, y_train, cv=cv, scoring="accuracy")
        results.append({
            "n_features": k,
            "features": feats,
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
            "cv_accuracy_mean": round(float(np.mean(acc_scores)), 4),
        })

    best_f1 = max(r["cv_f1_mean"] for r in results)
    within_tolerance = [r for r in results if (best_f1 - r["cv_f1_mean"]) <= DROP_TOLERANCE_F1]
    recommended = min(within_tolerance, key=lambda r: r["n_features"])

    best_overall = max(results, key=lambda r: r["cv_f1_mean"])
    if recommended["n_features"] == best_overall["n_features"]:
        reason = (
            f"The {recommended['n_features']}-feature representation achieves the best "
            f"cross-validated F1 ({recommended['cv_f1_mean']:.4f}) among the candidate counts tested."
        )
    else:
        reason = (
            f"The {recommended['n_features']}-feature representation preserves cross-validated "
            f"performance within {DROP_TOLERANCE_F1*100:.0f} percentage points of F1 of the best candidate "
            f"({best_overall['n_features']} features, F1={best_overall['cv_f1_mean']:.4f}), "
            f"while minimizing quantum circuit dimensionality."
        )

    return {
        "candidates_evaluated": results,
        "recommended_n_features": recommended["n_features"],
        "recommended_features": recommended["features"],
        "best_by_raw_performance": best_overall["n_features"],
        "drop_tolerance_f1": DROP_TOLERANCE_F1,
        "reason": reason,
        "cv_folds": n_splits,
    }
