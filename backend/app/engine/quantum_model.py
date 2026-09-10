"""
quantum_model.py
==================
Builds and trains a VQC on the ideal Aer simulator. Generalized from the
original WDBC-only module: qubit count, reps, shots, optimizer, seed, and
maxiter are all configurable instead of fixed.

VQC training cost scales with (training samples) x (circuit evaluations
per optimizer iteration) x (maxiter). For an interactive platform running
on arbitrary uploaded datasets, training is subsampled above
MAX_VQC_TRAIN_SAMPLES -- exactly the same "explain, don't hide" approach
the original project used for the quantum kernel's O(n^2) cost.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.model_selection import train_test_split

MAX_VQC_TRAIN_SAMPLES = 150


def subsample_for_vqc(X_train_q, y_train, max_samples: int, random_state: int):
    if len(X_train_q) <= max_samples:
        return X_train_q, y_train, None
    X_sub, _, y_sub, _ = train_test_split(
        X_train_q, y_train, train_size=max_samples, stratify=y_train, random_state=random_state,
    )
    note = (
        f"VQC training uses {max_samples} of {len(X_train_q)} training samples "
        f"because each optimizer iteration re-evaluates the circuit on every "
        f"training sample; training was subsampled (stratified, fixed seed) to "
        f"keep training time practical."
    )
    return X_sub, y_sub, note


def build_vqc(feature_map, ansatz, random_seed: int, shots: int = 1024, maxiter: int = 100, optimizer_name: str = "COBYLA"):
    algorithm_globals.random_seed = random_seed
    sampler = AerSamplerV2(default_shots=shots, seed=random_seed)
    optimizer = COBYLA(maxiter=maxiter)  # COBYLA: gradient-free, standard default for shot-based VQC loss

    loss_history: list[float] = []

    def callback(weights, loss):
        loss_history.append(float(loss))

    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler, callback=callback)
    return vqc, loss_history


def train_vqc(vqc, loss_history, X_train, y_train):
    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    start = time.time()
    vqc.fit(X_train_arr, y_train_arr)
    elapsed = time.time() - start

    training_info = {
        "optimizer": "COBYLA",
        "num_loss_evaluations": len(loss_history),
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "training_time_seconds": round(elapsed, 3),
        "converged": _check_convergence(loss_history),
    }
    return training_info


def _check_convergence(loss_history, window: int = 5, rel_tol: float = 0.02):
    if len(loss_history) < window + 1:
        return None
    recent = loss_history[-window:]
    rel_change = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-9)
    return bool(rel_change < rel_tol)
