"""
exact_training.py
====================
Phase 2B Experiment A: exact/statevector-compatible training evaluation,
using the SAME feature map + ansatz circuit as the primary VQC, with the
shot-sampled Sampler-based evaluation replaced by an Estimator-based
EXACT expectation value.

Why this is the correct "exact" pathway, and not an invented one
--------------------------------------------------------------------
Qiskit's V2 Sampler primitives (AerSamplerV2, StatevectorSampler) always
draw a finite number of shots -- there is no "shots=None -> exact
probabilities" mode in this installed Qiskit version. VQC is hard-wired
to a Sampler (parity-of-counts) interpretation, so VQC itself cannot be
made exact.

qiskit.primitives.StatevectorEstimator, however, IS exact for any
circuit containing only unitary operations (per its own documentation):
it computes expectation values via linear algebra on the statevector,
not by sampling. Our feature-map+ansatz circuit is fully unitary (no
resets, no mid-circuit measurement), so this route gives a genuinely
exact, zero-shot-noise training signal using the identical circuit
structure.

The observable is Z^{\\otimes 4} ("ZZZZ") -- the exact expectation-value
analog of the ORIGINAL VQC's parity-of-all-qubits interpretation, so
this experiment isolates "shot noise vs. no shot noise" as cleanly as
possible without changing what is being measured conceptually.

Necessary consequence of this route (documented, not hidden): switching
from Sampler-based multi-outcome classification to Estimator-based
single-observable regression requires (a) squared_error loss instead of
cross_entropy -- QNN outputs in [-1, 1] are not a probability
distribution, so cross-entropy does not apply -- and (b) labels encoded
as {-1, +1} rather than {0, 1}. Qiskit Machine Learning's
NeuralNetworkClassifier does NOT auto-convert plain 0/1 integer labels
for its 1-dimensional binary-output case (verified directly from its
source: labels are only re-encoded when they are strings, or when
one_hot=True); passing raw {0,1} labels here silently trains against
the wrong target scale. This module handles that conversion explicitly
and predict() converts back to {0, 1} before returning, so callers
never have to think about the internal {-1,+1} encoding.
"""

import time

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

MALIGNANT_LABEL = 0
BENIGN_LABEL = 1


def build_exact_qnn(feature_map, ansatz, full_circuit):
    observable = SparsePauliOp("Z" * full_circuit.num_qubits)  # parity analog
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=full_circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def train_and_evaluate_exact(feature_map, ansatz, full_circuit,
                              X_train_q, y_train, X_test_q, y_test,
                              random_seed, maxiter=100):
    qnn = build_exact_qnn(feature_map, ansatz, full_circuit)

    y_train_signed = 2 * np.asarray(y_train) - 1  # {0,1} -> {-1,+1}
    y_test_arr = np.asarray(y_test)

    algorithm_globals.random_seed = random_seed
    loss_history = []

    def callback(weights, loss):
        loss_history.append(float(loss))

    clf = NeuralNetworkClassifier(
        neural_network=qnn, loss="squared_error",
        optimizer=COBYLA(maxiter=maxiter), callback=callback,
    )

    start = time.time()
    clf.fit(np.asarray(X_train_q), y_train_signed)
    elapsed = time.time() - start

    pred_signed = clf.predict(np.asarray(X_test_q))
    pred = ((pred_signed + 1) / 2).astype(int)  # back to {0, 1}

    raw = qnn.forward(np.asarray(X_test_q), clf._fit_result.x)  # in [-1, +1]
    # observable = ZZZZ; +1 <-> label 1 (benign) side, -1 <-> label 0 (malignant) side
    # (matches the {-1,+1} target encoding used for training)
    proba_malignant = ((1 - raw.flatten()) / 2)

    return {
        "predictions": pred.tolist(),
        "proba_malignant": proba_malignant.tolist(),
        "loss_history": loss_history,
        "num_loss_evaluations": len(loss_history),
        "final_loss": loss_history[-1] if loss_history else None,
        "training_time_seconds": round(elapsed, 3),
        "final_parameters": clf._fit_result.x.tolist(),
        "true_labels": y_test_arr.tolist(),
    }
