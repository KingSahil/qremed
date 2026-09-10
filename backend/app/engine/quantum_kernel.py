"""
quantum_kernel.py
====================
Quantum-kernel SVM classifier. Generalized from the WDBC-only module:
feature map dimension follows the selected feature count, and the
subsampling message is generated dynamically instead of hard-coded.

A quantum kernel needs one circuit evaluation per PAIR of samples --
O(n^2) for the training kernel matrix. Training is therefore subsampled
(stratified, fixed seed) above MAX_KERNEL_TRAIN_SAMPLES and this is
surfaced to the user, never hidden. The test set is NOT subsampled.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

MAX_KERNEL_TRAIN_SAMPLES = 80


def subsample_training_set(X_train_q, y_train, n_samples: int, random_seed: int):
    if n_samples >= len(X_train_q):
        return X_train_q, y_train, None
    X_sub, _, y_sub, _ = train_test_split(
        X_train_q, y_train, train_size=n_samples, stratify=y_train, random_state=random_seed,
    )
    note = (
        f"Quantum Kernel training uses {n_samples} of {len(X_train_q)} training samples "
        f"because kernel construction requires pairwise quantum evaluations "
        f"(O(n^2) cost)."
    )
    return X_sub, y_sub, note


def build_kernel(feature_map, shots: int, random_seed: int):
    sampler = AerSamplerV2(default_shots=shots, seed=random_seed)
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


def train_and_evaluate(feature_map, X_train_sub_q, y_train_sub, X_test_q, y_test,
                        shots: int, random_seed: int, positive_label: int):
    kernel = build_kernel(feature_map, shots, random_seed)

    X_train_arr = np.asarray(X_train_sub_q)
    X_test_arr = np.asarray(X_test_q)
    y_train_arr = np.asarray(y_train_sub)
    y_test_arr = np.asarray(y_test)

    t0 = time.time()
    K_train = kernel.evaluate(x_vec=X_train_arr)
    kernel_train_time = time.time() - t0

    t0 = time.time()
    K_test = kernel.evaluate(x_vec=X_test_arr, y_vec=X_train_arr)
    kernel_test_time = time.time() - t0

    t0 = time.time()
    svm = SVC(kernel="precomputed", probability=True, random_state=random_seed)
    svm.fit(K_train, y_train_arr)
    svm_train_time = time.time() - t0

    pred = svm.predict(K_test)
    proba = svm.predict_proba(K_test)
    pos_col = list(svm.classes_).index(positive_label)
    y_score_positive = proba[:, pos_col]

    from .evaluation import evaluate_predictions
    negative_label = 1 - positive_label
    result = evaluate_predictions(y_test_arr, pred, y_score_positive, positive_label, negative_label)
    result["predictions"] = pred.tolist()
    result["probabilities"] = proba.tolist()
    result["timing"] = {
        "kernel_train_matrix_time_seconds": round(kernel_train_time, 3),
        "kernel_test_matrix_time_seconds": round(kernel_test_time, 3),
        "svm_train_time_seconds": round(svm_train_time, 3),
    }
    result["n_train_subsampled"] = len(X_train_arr)
    result["n_test"] = len(X_test_arr)
    return result
