"""
feature_selection.py
=====================
Feature ranking computed using TRAINING DATA ONLY.

Primary method: ANOVA F-score (sklearn.feature_selection.f_classif).
Cross-check method: Random Forest feature importance.

The Random Forest ranking is reported alongside the ANOVA ranking for
comparison, but it never drives the actual feature selection used
downstream -- only the ANOVA ranking does, per project decision (see
Phase 1 plan, Section 5).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif


def anova_ranking(X_train, y_train):
    """Primary ranking method: ANOVA F-score, computed on TRAIN only."""
    f_scores, p_values = f_classif(X_train, y_train)
    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "score": f_scores,
        "p_value": p_values,
        "method": "anova_f_score",
    }).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1  # 1-based rank
    ranking.index.name = "rank"
    return ranking


def random_forest_importance_ranking(X_train, y_train, rf_params):
    """Cross-check ranking method: Random Forest feature importance, TRAIN only."""
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "score": rf.feature_importances_,
        "method": "random_forest_importance",
    }).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "rank"
    return ranking


def compare_rankings(anova_df, rf_df, top_n=10):
    """
    Compare the top-N features from each ranking method and report whether
    they broadly agree (useful sanity check, flagged in the report rather
    than silently ignored).
    """
    anova_top = set(anova_df.head(top_n)["feature"])
    rf_top = set(rf_df.head(top_n)["feature"])
    overlap = anova_top & rf_top
    agreement_pct = 100 * len(overlap) / top_n
    return {
        "top_n": top_n,
        "anova_top_features": sorted(anova_top),
        "rf_top_features": sorted(rf_top),
        "overlap_features": sorted(overlap),
        "agreement_pct": round(agreement_pct, 1),
    }


def select_top_features(ranking_df, n_features):
    """Select the top-N feature names from a ranking DataFrame (already sorted)."""
    return list(ranking_df.head(n_features)["feature"])
