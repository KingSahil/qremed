"""
quantum_preprocessing.py
==========================
Quantum-specific feature scaling, applied ON TOP OF (never instead of)
Phase 1's leakage-safe StandardScaler output. This module does not touch
or modify Phase 1's pipeline.

Why an additional transformation is needed
--------------------------------------------
Phase 1's StandardScaler produces features with mean 0, std 1 -- but no
fixed bound. A handful of samples can sit at +/-3 or beyond.

The chosen feature map (ZZFeatureMap) encodes each feature value as a
rotation angle via gates like RZ(2*x). Rotation gates are periodic with
period 2*pi: two feature values that differ by exactly 2*pi (or any
multiple of it) produce the IDENTICAL quantum state, even though the
original feature values are different. Left unbounded, standardized
values could in principle wrap around this periodicity and become
indistinguishable to the circuit -- silently destroying information the
classical baselines could see.

The fix: after Phase 1's StandardScaler, apply a MinMaxScaler -- fit on
TRAIN only, exactly like Phase 1's own scaler -- that maps every feature
into a fixed, sub-period range: [0, pi]. This keeps every encoded angle
well inside a single 2*pi period (with headroom, since pi < 2*pi), so
distinct feature values always produce distinguishable rotation angles.
[0, pi] (rather than the also-common [-pi, pi]) is chosen because it also
keeps every value non-negative, which slightly simplifies later circuit-
depth/parameter-range reasoning and matches common Qiskit ZZFeatureMap
usage examples.

Where this scaling happens
-----------------------------
This is a SEPARATE step from Phase 1's preprocessing.leakage_safe_preprocess,
applied immediately before circuit encoding, not folded into Phase 1's
saved scaler -- so Phase 1's own pipeline and its saved artifacts remain
completely unmodified, per the Phase 2 instruction not to touch Phase 1's
scientific results or preprocessing logic.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_quantum_scaler_on_train(X_train_standardized, feature_range=(0, np.pi)):
    """Fit a MinMaxScaler on the already-standardized TRAINING features only."""
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train_standardized)
    return scaler


def apply_quantum_scaler(scaler, X_standardized):
    """Transform any split (train or test) using an already-fitted quantum scaler."""
    scaled = scaler.transform(X_standardized)
    return pd.DataFrame(scaled, columns=X_standardized.columns, index=X_standardized.index)


def prepare_quantum_features(X_train_standardized, X_test_standardized):
    """
    Full quantum-feature-prep entry point. Takes Phase 1's already
    leakage-safe standardized train/test data (restricted to the selected
    feature columns) and returns [0, pi]-scaled versions ready for circuit
    encoding, plus the fitted quantum scaler for reproducibility records.
    """
    scaler = fit_quantum_scaler_on_train(X_train_standardized)
    X_train_q = apply_quantum_scaler(scaler, X_train_standardized)
    X_test_q = apply_quantum_scaler(scaler, X_test_standardized)
    return X_train_q, X_test_q, scaler
