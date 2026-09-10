"""
preprocessing.py
=================
Leakage-safe preprocessing: stratified train/test split, then a
StandardScaler fit ONLY on the training fold.

Order enforced by this module:

    raw data
      -> stratified train/test split (fixed seed)
      -> fit StandardScaler on TRAIN only
      -> transform TRAIN and TEST with that fitted scaler

Feature selection (a separate, later step) must also only look at the
returned TRAIN data -- see feature_selection.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def stratified_split(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_scaler_on_train(X_train):
    """Fit a StandardScaler using training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler, X):
    """Transform any split (train or test) using an already-fitted scaler."""
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


def leakage_safe_preprocess(X, y, test_size, random_state):
    """
    Full leakage-safe preprocessing entry point.
    Returns scaled train/test splits plus the fitted scaler (for reuse
    downstream, e.g. by the future quantum phase).
    """
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)

    scaler = fit_scaler_on_train(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled = apply_scaler(scaler, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
