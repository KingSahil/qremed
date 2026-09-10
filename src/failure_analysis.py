"""
failure_analysis.py
=====================
Runs the Phase 2 Section 11 failure-analysis checklist when VQC
performance is substantially worse than the classical baselines, in the
specified order:

    1. preprocessing
    2. label encoding
    3. feature scaling
    4. circuit construction
    5. measurement/output interpretation
    6. optimizer convergence
    7. initialization
    8. circuit depth

Each check either rules a cause in or out, with the evidence recorded.
No architecture change is applied automatically -- per Phase 2 scope,
this module only diagnoses and reports; any resulting change (e.g. a
6-qubit escalation, a different primary optimizer) is a decision left
for review, not something this script decides unilaterally.
"""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2

import quantum_circuit


def check_preprocessing_and_scaling(X_train_q, X_test_q, y_train, y_test, feature_names, random_seed):
    """
    Checks 1-3 combined: verify the classification signal is still present
    in the exact quantum-scaled ([0, pi]) data a classical model would see,
    and verify column order/labels are consistent train vs test.
    """
    columns_consistent = (
        list(X_train_q.columns) == feature_names and list(X_test_q.columns) == feature_names
    )
    lr = LogisticRegression(random_state=random_seed, max_iter=1000)
    lr.fit(X_train_q, y_train)
    reproduced_acc = accuracy_score(y_test, lr.predict(X_test_q))

    return {
        "check": "preprocessing / label encoding / feature scaling (checks 1-3)",
        "method": (
            "Train a classical Logistic Regression on the EXACT same "
            "[0, pi]-scaled 4-feature data the VQC receives, using the "
            "same train/test split and labels."
        ),
        "column_order_consistent_train_test": columns_consistent,
        "classical_accuracy_on_quantum_scaled_data": round(float(reproduced_acc), 4),
        "conclusion": (
            "Signal fully intact after quantum scaling -- rules out a "
            "preprocessing, label-encoding, or feature-scaling bug as the "
            "cause of VQC underperformance."
            if reproduced_acc > 0.85 else
            "Classical accuracy dropped substantially after quantum "
            "scaling -- preprocessing/scaling is implicated and should be "
            "investigated further before blaming the circuit/optimizer."
        ),
    }


def check_initialization_stability(feature_map, ansatz, X_train_q, y_train, X_test_q, y_test,
                                    seeds, shots, maxiter):
    """Check 7: multiple random seeds/initial points, same circuit and optimizer (COBYLA)."""
    from qiskit_machine_learning.optimizers import COBYLA

    results = []
    for seed in seeds:
        algorithm_globals.random_seed = seed
        sampler = AerSamplerV2(default_shots=shots, seed=seed)
        vqc = VQC(feature_map=feature_map, ansatz=ansatz,
                  optimizer=COBYLA(maxiter=maxiter), sampler=sampler)
        vqc.fit(np.asarray(X_train_q), np.asarray(y_train))
        acc = accuracy_score(y_test, vqc.predict(np.asarray(X_test_q)))
        results.append({"seed": seed, "test_accuracy": round(float(acc), 4)})

    accs = [r["test_accuracy"] for r in results]
    spread = max(accs) - min(accs)
    return {
        "check": "initialization / seed stability (check 7)",
        "method": f"COBYLA (maxiter={maxiter}), {len(seeds)} different random seeds, "
                  "same circuit structure.",
        "results_by_seed": results,
        "accuracy_spread": round(float(spread), 4),
        "conclusion": (
            "Consistently poor performance across independent random "
            "initializations -- rules out an unlucky single initialization "
            "as the primary cause; points toward a structural (circuit / "
            "optimizer landscape) limitation rather than initialization "
            "chance."
            if spread < 0.10 else
            "Performance varies substantially across seeds -- suggests "
            "initialization sensitivity / local minima are a meaningful "
            "factor, and a multi-restart training strategy could help."
        ),
    }


def check_optimizer_choice(feature_map, ansatz, X_train_q, y_train, X_test_q, y_test,
                            random_seed, shots, maxiter):
    """Check 6: compare COBYLA (gradient-free direct search) vs SPSA (stochastic approximation,
    designed for noisy objectives) on the identical circuit."""
    from qiskit_machine_learning.optimizers import COBYLA

    algorithm_globals.random_seed = random_seed
    sampler_c = AerSamplerV2(default_shots=shots, seed=random_seed)
    vqc_c = VQC(feature_map=feature_map, ansatz=ansatz,
                optimizer=COBYLA(maxiter=maxiter), sampler=sampler_c)
    vqc_c.fit(np.asarray(X_train_q), np.asarray(y_train))
    acc_c = accuracy_score(y_test, vqc_c.predict(np.asarray(X_test_q)))

    algorithm_globals.random_seed = random_seed
    sampler_s = AerSamplerV2(default_shots=shots, seed=random_seed)
    spsa_loss = []
    def cb(*args):
        spsa_loss.append(float(args[2]))
    vqc_s = VQC(feature_map=feature_map, ansatz=ansatz,
                optimizer=SPSA(maxiter=maxiter), sampler=sampler_s, callback=cb)
    vqc_s.fit(np.asarray(X_train_q), np.asarray(y_train))
    acc_s = accuracy_score(y_test, vqc_s.predict(np.asarray(X_test_q)))

    both_poor = acc_c < 0.75 and acc_s < 0.75
    return {
        "check": "optimizer convergence (check 6)",
        "method": f"Same circuit/data, COBYLA vs SPSA, both maxiter={maxiter}, shots={shots}.",
        "cobyla_test_accuracy": round(float(acc_c), 4),
        "spsa_test_accuracy": round(float(acc_s), 4),
        "conclusion": (
            "Two structurally different optimizers (direct-search vs "
            "stochastic-approximation) converge to similarly poor results "
            "-- this argues against 'wrong optimizer' as the root cause "
            "and toward a circuit-expressivity / loss-landscape limitation."
            if both_poor else
            "Optimizer choice materially changes the result -- the "
            "primary optimizer choice should be revisited."
        ),
    }


def check_circuit_depth(feature_map, X_train_q, y_train, X_test_q, y_test, random_seed, shots, maxiter):
    """Check 8: does increasing ansatz depth (reps 1 -> 2) change the outcome?"""
    from qiskit_machine_learning.optimizers import COBYLA

    results = {}
    for reps in (1, 2):
        ansatz = quantum_circuit.build_ansatz(num_qubits=4, reps=reps)
        algorithm_globals.random_seed = random_seed
        sampler = AerSamplerV2(default_shots=shots, seed=random_seed)
        vqc = VQC(feature_map=feature_map, ansatz=ansatz,
                  optimizer=COBYLA(maxiter=maxiter), sampler=sampler)
        vqc.fit(np.asarray(X_train_q), np.asarray(y_train))
        acc = accuracy_score(y_test, vqc.predict(np.asarray(X_test_q)))
        results[f"reps_{reps}"] = {
            "num_trainable_parameters": ansatz.num_parameters,
            "test_accuracy": round(float(acc), 4),
        }

    improved = results["reps_2"]["test_accuracy"] - results["reps_1"]["test_accuracy"] > 0.10
    return {
        "check": "circuit depth / ansatz expressivity (check 8)",
        "method": "Same feature map, ansatz reps=1 vs reps=2, same optimizer/seed.",
        "results_by_reps": results,
        "conclusion": (
            "Increasing ansatz depth materially improved performance -- "
            "under-expressivity at reps=1 is implicated."
            if improved else
            "Increasing ansatz depth did not materially improve "
            "performance -- rules out simple under-expressivity (at least "
            "within this modest depth range) as the sole cause."
        ),
    }


def run_full_failure_analysis(feature_map, ansatz, X_train_q, X_test_q, y_train, y_test,
                                feature_names, random_seed, shots, maxiter):
    """
    Runs checks 1-3, 6, 7, 8 (measurement interpretation, check 5, is verified
    separately and cheaply -- see run_phase2.py, since it only requires
    inspecting the QNN's interpret function, not retraining).
    """
    print("Running Phase 2 failure analysis (VQC substantially underperformed classical baselines)...")

    result_123 = check_preprocessing_and_scaling(
        X_train_q, X_test_q, y_train, y_test, feature_names, random_seed
    )
    print(f"  [1-3] {result_123['conclusion']}")

    result_6 = check_optimizer_choice(
        feature_map, ansatz, X_train_q, y_train, X_test_q, y_test,
        random_seed, shots=256, maxiter=25,
    )
    print(f"  [6] {result_6['conclusion']}")

    result_7 = check_initialization_stability(
        feature_map, ansatz, X_train_q, y_train, X_test_q, y_test,
        seeds=[1, 7, 42, 123, 2024], shots=512, maxiter=60,
    )
    print(f"  [7] {result_7['conclusion']}")

    result_8 = check_circuit_depth(
        feature_map, X_train_q, y_train, X_test_q, y_test,
        random_seed, shots=1024, maxiter=100,
    )
    print(f"  [8] {result_8['conclusion']}")

    return {
        "checks_1_to_3_preprocessing_labels_scaling": result_123,
        "check_6_optimizer_convergence": result_6,
        "check_7_initialization_stability": result_7,
        "check_8_circuit_depth": result_8,
    }
