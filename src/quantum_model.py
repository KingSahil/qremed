"""
quantum_model.py
==================
Builds and trains the VQC on the ideal Aer simulator ONLY (no noise, no
fake backend, no IBM hardware -- training happens purely in simulation,
per Phase 2 scope).

Optimizer choice: COBYLA, maxiter=100
    - Gradient-free: VQC's loss is evaluated via sampled measurement
      counts (shot noise), and a gradient-free optimizer avoids the extra
      circuit evaluations a gradient-based method would need (e.g.
      parameter-shift), which matters once real hardware time enters the
      picture in a later phase.
    - Standard first choice for small VQC problems in the Qiskit Machine
      Learning documentation/tutorials -- picking one reasonable default
      rather than benchmarking several, per Phase 2 scope.
    - maxiter=100 was chosen after timing a smaller run (10 iterations on
      the full ~455-sample training set took ~12s), giving an estimated
      full-training budget of roughly 2 minutes -- reasonable for a
      shallow, 4-qubit, 8-parameter circuit.

Reproducibility: qiskit_machine_learning.utils.algorithm_globals.random_seed
is fixed, and the AerSamplerV2's own seed is fixed separately (shot
sampling is itself stochastic and needs its own seed).
"""

import time

import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


def build_vqc(feature_map, ansatz, random_seed, shots=1024, maxiter=100):
    algorithm_globals.random_seed = random_seed

    sampler = AerSamplerV2(default_shots=shots, seed=random_seed)
    optimizer = COBYLA(maxiter=maxiter)

    loss_history = []

    def callback(weights, loss):
        loss_history.append(float(loss))

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        callback=callback,
    )
    return vqc, loss_history


def train_vqc(vqc, loss_history, X_train, y_train):
    """
    Train on the ideal Aer simulator (via the sampler bound into `vqc`).
    Returns training metadata: elapsed time, iteration count, final loss,
    and the frozen (post-training) parameters.
    """
    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    start = time.time()
    vqc.fit(X_train_arr, y_train_arr)
    elapsed = time.time() - start

    final_params = np.asarray(vqc.weights).tolist()

    training_info = {
        "optimizer": "COBYLA",
        "maxiter": vqc.optimizer.settings.get("maxiter") if hasattr(vqc.optimizer, "settings") else None,
        "num_loss_evaluations": len(loss_history),
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
        "training_time_seconds": round(elapsed, 3),
        "final_parameters": final_params,
        "converged": _check_convergence(loss_history),
    }
    return training_info


def _check_convergence(loss_history, window=5, rel_tol=0.02):
    """
    Lightweight convergence check: did the loss stabilize (small relative
    change) over the final `window` evaluations? This is a descriptive
    signal for the training report, not a stopping criterion (COBYLA
    already ran to its own maxiter/tolerance).
    """
    if len(loss_history) < window + 1:
        return None
    recent = loss_history[-window:]
    rel_change = (max(recent) - min(recent)) / (abs(min(recent)) + 1e-9)
    return bool(rel_change < rel_tol)
