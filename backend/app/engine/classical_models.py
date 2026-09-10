"""
classical_models.py
=====================
Classical baselines: Logistic Regression, Random Forest.
Same as the original Q-REMED module, generalized to accept any
feature-selected, scaled training set (not just the WDBC 4/6-feature
subsets).
"""
from __future__ import annotations

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DEFAULT_LR_PARAMS = {"max_iter": 1000}
DEFAULT_RF_PARAMS = {"n_estimators": 200}


def train_all_baselines(X_train, y_train, random_state: int = 42):
    lr_params = {**DEFAULT_LR_PARAMS, "random_state": random_state}
    rf_params = {**DEFAULT_RF_PARAMS, "random_state": random_state}

    models = {}
    for name, cls, params in [
        ("Logistic Regression", LogisticRegression, lr_params),
        ("Random Forest", RandomForestClassifier, rf_params),
    ]:
        t0 = time.time()
        model = cls(**params)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        models[name] = {"model": model, "training_time_seconds": round(elapsed, 4)}
    return models
