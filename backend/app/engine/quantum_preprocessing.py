"""
quantum_preprocessing.py
==========================
Quantum-specific feature scaling, applied on top of (never instead of)
the leakage-safe StandardScaler output. Generalized to any selected
feature count -- same [0, pi] rationale as the original WDBC module
(keeps ZZFeatureMap rotation angles inside a single period so distinct
values stay distinguishable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_quantum_features(X_train_standardized: pd.DataFrame, X_test_standardized: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_train_standardized)
    X_train_q = pd.DataFrame(scaler.transform(X_train_standardized),
                              columns=X_train_standardized.columns, index=X_train_standardized.index)
    X_test_q = pd.DataFrame(scaler.transform(X_test_standardized),
                             columns=X_test_standardized.columns, index=X_test_standardized.index)
    return X_train_q, X_test_q, scaler
