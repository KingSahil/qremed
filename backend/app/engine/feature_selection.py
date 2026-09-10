"""
feature_selection.py
=====================
Generalized version of the original Q-REMED feature ranking module.

Same methodology as the WDBC research (ANOVA F-score as the primary
ranking, Random Forest importance as a cross-check), but computed fresh
for whatever dataset/features are given -- nothing is hard-coded to
WDBC's "worst concave points / worst perimeter / mean concave points /
worst radius".

Both rankings are computed using TRAINING DATA ONLY.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif


def anova_ranking(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    f_scores, p_values = f_classif(X_train, y_train)
    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "score": f_scores,
        "p_value": p_values,
        "method": "anova_f_score",
    }).fillna({"score": 0.0, "p_value": 1.0}).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "rank"
    return ranking


def random_forest_importance_ranking(X_train: pd.DataFrame, y_train: pd.Series,
                                      random_state: int = 42, n_estimators: int = 200) -> pd.DataFrame:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "score": rf.feature_importances_,
        "method": "random_forest_importance",
    }).sort_values("score", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    ranking.index.name = "rank"
    return ranking


def compare_rankings(anova_df: pd.DataFrame, rf_df: pd.DataFrame, top_n: int = 10) -> dict:
    top_n = min(top_n, len(anova_df), len(rf_df))
    anova_top = set(anova_df.head(top_n)["feature"])
    rf_top = set(rf_df.head(top_n)["feature"])
    overlap = anova_top & rf_top
    agreement_pct = 100 * len(overlap) / top_n if top_n else 0.0
    return {
        "top_n": top_n,
        "anova_top_features": sorted(anova_top),
        "rf_top_features": sorted(rf_top),
        "overlap_features": sorted(overlap),
        "agreement_pct": round(agreement_pct, 1),
    }


def select_top_features(ranking_df: pd.DataFrame, n_features: int) -> list[str]:
    return list(ranking_df.head(n_features)["feature"])


def ranking_to_records(ranking_df: pd.DataFrame) -> list[dict]:
    out = ranking_df.reset_index().rename(columns={"index": "rank"})
    return out.round({"score": 6, "p_value": 6}).to_dict(orient="records")
