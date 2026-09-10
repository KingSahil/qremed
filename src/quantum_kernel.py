"""
quantum_kernel.py
====================
Phase 2C Experiment 3: one quantum-kernel classifier.

Feature map: ZZFeatureMap(reps=1) -- the SAME feature map already used
for the VQC experiments (not a new circuit family), applied to the same
4 primary features with the same [0, pi] quantum scaling.

Kernel: state-fidelity kernel (|<phi(x_i)|phi(x_j)>|^2) via
FidelityQuantumKernel + ComputeUncompute, evaluated on ideal Aer
(1024 shots), matching the "ideal Aer" execution environment used
throughout this project.

Computational-feasibility note (documented, not hidden)
------------------------------------------------------------
A quantum kernel requires one circuit evaluation per PAIR of samples --
O(n^2) for the training kernel matrix. At the measured ~6-8ms per pair
on this hardware, the full 455-sample Phase 1 training set would need
roughly 103,000 training pairs alone (~15+ minutes) plus another ~52,000
train-test pairs -- infeasible within this environment's per-command
execution limits. The training set is therefore SUBSAMPLED (stratified,
fixed seed) to a smaller size before kernel computation. The test set
is NOT subsampled -- it remains the full, untouched Phase 1 test set.
This means the quantum kernel SVM is trained on less data than the VQC
or classical baselines, which is a real limitation of this comparison,
not swept under the rug -- see the Phase 2C interpretation for how this
is treated in the final comparison.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def subsample_training_set(X_train_q, y_train, n_samples, random_seed):
    """Stratified subsample of the training set for kernel-matrix feasibility."""
    if n_samples >= len(X_train_q):
        return X_train_q, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train_q, y_train,
        train_size=n_samples, stratify=y_train, random_state=random_seed,
    )
    return X_sub, y_sub


def build_kernel(feature_map, shots, random_seed):
    sampler = AerSamplerV2(default_shots=shots, seed=random_seed)
    fidelity = ComputeUncompute(sampler=sampler)
    return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)


def train_and_evaluate(feature_map, X_train_sub_q, y_train_sub, X_test_q, y_test,
                        shots, random_seed):
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
    malignant_col = list(svm.classes_).index(0)  # 0 = malignant

    return {
        "predictions": pred.tolist(),
        "probabilities": proba.tolist(),
        "malignant_proba_column": malignant_col,
        "kernel_train_matrix_time_seconds": round(kernel_train_time, 3),
        "kernel_test_matrix_time_seconds": round(kernel_test_time, 3),
        "svm_train_time_seconds": round(svm_train_time, 3),
        "n_train_subsampled": len(X_train_arr),
        "n_test": len(X_test_arr),
        "svm_classes": svm.classes_.tolist(),
    }
