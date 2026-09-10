"""
finalize_phase2.py
=====================
Assembles results/phase2_failure_analysis.json from the diagnostic checks
already run (Section 11 order: preprocessing/labels/scaling, optimizer
convergence, initialization stability, circuit depth -- circuit
construction and measurement interpretation verified statically), and
rewrites phase2_scientific_interpretation.json to honestly reflect the
finding that the primary VQC configuration substantially underperformed
the classical baselines and did not clearly beat the trivial majority-
class baseline.

Does not retrain the primary VQC (already trained and frozen, results in
vqc_training_info.json / vqc_test_predictions.json from the successful
Phase 2 run). Does not change the primary configuration -- Phase 2 scope
prohibits unilateral architecture changes; this stays a diagnostic report.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

RESULTS_DIR = config.RESULTS_DIR


def main():
    training_info = json.load(open(os.path.join(RESULTS_DIR, "vqc_training_info.json")))
    circuit_analysis = json.load(open(os.path.join(RESULTS_DIR, "circuit_analysis.json")))
    hw_check = json.load(open(os.path.join(RESULTS_DIR, "hardware_compatibility_check.json")))

    import pandas as pd
    comparison_df = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_classical_vs_quantum.csv"))
    lr_row = comparison_df[comparison_df["model"] == "Logistic Regression"].iloc[0]
    rf_row = comparison_df[comparison_df["model"] == "Random Forest"].iloc[0]
    vqc_row = comparison_df[comparison_df["model"] == "VQC"].iloc[0]

    # Majority-class baseline, recomputed from the exact Phase 1 split.
    import data_loader
    import preprocessing
    X, y, tn, fn = data_loader.load_dataset()
    X_train_full, X_test_full, y_train, y_test, scaler = preprocessing.leakage_safe_preprocess(
        X, y, config.TEST_SIZE, config.RANDOM_SEED
    )
    majority_class = int(y_train.value_counts().idxmax())
    majority_baseline_accuracy = float((y_test == majority_class).mean())

    performance_gap = float(lr_row["accuracy"] - vqc_row["accuracy"])

    # ------------------------------------------------------------------
    # Checks 1-3: preprocessing / label encoding / feature scaling
    # (already verified: classical LR on the identical [0, pi]-scaled
    # 4-feature data reaches 0.9386 accuracy -- see run log)
    # ------------------------------------------------------------------
    check_123 = {
        "check": "preprocessing / label encoding / feature scaling (checks 1-3)",
        "method": (
            "Train a classical Logistic Regression on the EXACT same "
            "[0, pi]-scaled 4-feature data the VQC receives, using the "
            "same train/test split and labels."
        ),
        "column_order_consistent_train_test": True,
        "classical_accuracy_on_quantum_scaled_data": 0.9386,
        "conclusion": (
            "Signal fully intact after quantum scaling (0.939 accuracy, "
            "essentially matching the standardized-feature classical "
            "result) -- rules out a preprocessing, label-encoding, or "
            "feature-scaling bug as the cause of VQC underperformance."
        ),
    }

    # ------------------------------------------------------------------
    # Check 4: circuit construction (static review)
    # ------------------------------------------------------------------
    check_4 = {
        "check": "circuit construction (check 4)",
        "method": "Static review of feature-map/ansatz composition and parameter counts.",
        "feature_map_parameters": circuit_analysis["feature_map"]["num_parameters"],
        "ansatz_parameters": circuit_analysis["ansatz"]["num_parameters"],
        "full_circuit_depth": circuit_analysis["full_circuit"]["depth"],
        "conclusion": (
            "Circuit composes correctly (feature map then ansatz, 4 "
            "qubits, standard ZZFeatureMap + RealAmplitudes structure) -- "
            "no construction error found; expressivity tested separately "
            "under check 8."
        ),
    }

    # ------------------------------------------------------------------
    # Check 5: measurement / output interpretation (static review)
    # ------------------------------------------------------------------
    check_5 = {
        "check": "measurement / output interpretation (check 5)",
        "method": "Inspect the underlying SamplerQNN's interpret function and output shape.",
        "output_shape": [2],
        "interpret_function": "standard VQC parity-based binary interpretation (Qiskit ML default)",
        "predict_proba_column_order_verified": (
            "Verified on a separate, well-separated synthetic dataset: "
            "column 0 = P(label 0), column 1 = P(label 1), matching "
            "ascending sorted label order."
        ),
        "conclusion": (
            "Standard, unmodified Qiskit Machine Learning default parity "
            "interpretation for binary VQC classification -- no custom or "
            "misconfigured output mapping found."
        ),
    }

    # ------------------------------------------------------------------
    # Check 6: optimizer convergence (COBYLA vs SPSA, matched settings)
    # ------------------------------------------------------------------
    check_6 = {
        "check": "optimizer convergence (check 6)",
        "method": "Same circuit/data/seed (42), COBYLA vs SPSA, both maxiter=25, shots=256.",
        "cobyla_test_accuracy": 0.5965,
        "spsa_test_accuracy": 0.6140,
        "conclusion": (
            "Two structurally different optimizers (COBYLA: gradient-free "
            "direct search; SPSA: stochastic approximation, designed for "
            "noisy objectives) converge to similarly poor results (0.597 "
            "vs 0.614 accuracy, both far below the 0.930 classical "
            "baseline) -- this argues against 'wrong optimizer' as the "
            "root cause and toward a circuit-expressivity / loss-"
            "landscape limitation."
        ),
    }

    # ------------------------------------------------------------------
    # Check 7: initialization / seed stability
    # ------------------------------------------------------------------
    seed_results = [
        {"seed": 1, "test_accuracy": 0.6404},
        {"seed": 7, "test_accuracy": 0.6316},
        {"seed": 42, "test_accuracy": 0.5877},
        {"seed": 123, "test_accuracy": 0.6842},
        {"seed": 2024, "test_accuracy": 0.6228},
    ]
    accs = [r["test_accuracy"] for r in seed_results]
    spread = round(max(accs) - min(accs), 4)
    check_7 = {
        "check": "initialization / seed stability (check 7)",
        "method": "COBYLA (maxiter=60, shots=512), 5 different random seeds, same circuit structure.",
        "results_by_seed": seed_results,
        "accuracy_spread": spread,
        "conclusion": (
            f"Consistently poor performance across 5 independent random "
            f"initializations (accuracy range {min(accs):.3f}-{max(accs):.3f}, "
            f"spread={spread}) -- rules out an unlucky single initialization "
            f"as the primary cause; points toward a structural (circuit / "
            f"optimizer landscape) limitation rather than initialization "
            f"chance."
        ),
    }

    # ------------------------------------------------------------------
    # Check 8: circuit depth / ansatz expressivity
    # ------------------------------------------------------------------
    check_8 = {
        "check": "circuit depth / ansatz expressivity (check 8)",
        "method": "Same feature map, ansatz reps=1 vs reps=2, same optimizer (COBYLA, "
                  "maxiter=100, shots=1024, seed=42).",
        "results_by_reps": {
            "reps_1": {"num_trainable_parameters": 8, "test_accuracy": 0.5965},
            "reps_2": {"num_trainable_parameters": 12, "test_accuracy": 0.6053},
        },
        "conclusion": (
            "Increasing ansatz depth (reps 1 -> 2, 8 -> 12 trainable "
            "parameters) improved accuracy by only 0.009 -- rules out "
            "simple under-expressivity (at least within this modest depth "
            "range) as the sole cause."
        ),
    }

    failure_analysis_results = {
        "trigger": f"VQC accuracy ({vqc_row['accuracy']:.4f}) underperforms Logistic "
                   f"Regression ({lr_row['accuracy']:.4f}) by {performance_gap:.4f}, "
                   f"exceeding the 0.10 investigation threshold.",
        "checks_1_to_3_preprocessing_labels_scaling": check_123,
        "check_4_circuit_construction": check_4,
        "check_5_measurement_interpretation": check_5,
        "check_6_optimizer_convergence": check_6,
        "check_7_initialization_stability": check_7,
        "check_8_circuit_depth": check_8,
        "performance_gap_vs_logistic_regression": round(performance_gap, 4),
        "majority_class_baseline_accuracy": round(majority_baseline_accuracy, 4),
        "overall_conclusion": (
            "All eight failure-analysis checks were run in the specified "
            "order. None found a data, preprocessing, labeling, "
            "construction, or measurement-interpretation bug. Two "
            "independent optimizers and five independent random "
            "initializations all converge to similarly poor results, and "
            "doubling ansatz depth barely moved the needle. This is most "
            "consistent with a genuine trainability / optimization-"
            "landscape limitation of this shallow, gradient-free-trained, "
            "shot-sampled VQC configuration on this problem at this "
            "circuit size -- not a pipeline error. No architecture change "
            "was applied automatically; this is reported as a finding for "
            "review before Phase 3."
        ),
    }

    with open(os.path.join(RESULTS_DIR, "phase2_failure_analysis.json"), "w") as f:
        json.dump(failure_analysis_results, f, indent=2)

    # ------------------------------------------------------------------
    # Rewrite the scientific interpretation honestly
    # ------------------------------------------------------------------
    vqc_m = {
        "accuracy": float(vqc_row["accuracy"]),
        "sensitivity": float(vqc_row["sensitivity"]),
        "specificity": float(vqc_row["specificity"]),
        "f1": float(vqc_row["f1"]),
        "roc_auc": float(vqc_row["roc_auc"]),
    }
    beats_majority = vqc_m["accuracy"] > majority_baseline_accuracy
    depth_note = "shallow" if circuit_analysis["full_circuit"]["depth"] <= 15 else "moderate/deep"

    interpretation = {
        "1_did_vqc_learn_the_task": (
            f"Partially, and not well at the default decision threshold. "
            f"Test accuracy ({vqc_m['accuracy']:.3f}) is "
            f"{'above' if beats_majority else 'BELOW'} the trivial "
            f"majority-class baseline ({majority_baseline_accuracy:.3f}) -- "
            f"meaning the default-threshold predictions are "
            f"{'' if beats_majority else 'not '}competitive with simply "
            f"always predicting the majority class. However, ROC-AUC "
            f"({vqc_m['roc_auc']:.3f}) is meaningfully above chance (0.5), "
            f"indicating the model's underlying probability ranking does "
            f"carry some real class-discriminative signal even though its "
            f"default 0.5-threshold decisions are poorly calibrated. The "
            f"required failure-analysis checks (see "
            f"phase2_failure_analysis.json) rule out a preprocessing bug "
            f"(classical LR on the identical quantum-scaled data reaches "
            f"0.939 accuracy), rule out an unlucky single initialization "
            f"(consistently poor across 5 seeds, spread=0.097), and rule "
            f"out 'wrong optimizer' as the sole cause (COBYLA and SPSA "
            f"converge to similarly poor results). This points to a "
            f"genuine optimization-landscape / trainability limitation of "
            f"this shallow circuit configuration on this problem, not a "
            f"pipeline error."
        ),
        "2_comparison_with_logistic_regression": (
            f"VQC accuracy {vqc_m['accuracy']:.3f} vs Logistic Regression "
            f"{lr_row['accuracy']:.3f} (gap of {performance_gap:.3f}); VQC "
            f"F1 {vqc_m['f1']:.3f} vs LR F1 {lr_row['f1']:.3f}. VQC "
            f"substantially underperforms the linear classical baseline "
            f"on this run."
        ),
        "3_comparison_with_random_forest": (
            f"VQC accuracy {vqc_m['accuracy']:.3f} vs Random Forest "
            f"{rf_row['accuracy']:.3f}; VQC F1 {vqc_m['f1']:.3f} vs RF F1 "
            f"{rf_row['f1']:.3f}. Same substantial gap as against "
            f"Logistic Regression."
        ),
        "4_is_circuit_shallow_enough_for_hardware": (
            f"Full circuit depth is {circuit_analysis['full_circuit']['depth']} "
            f"({depth_note}) with {circuit_analysis['full_circuit']['two_qubit_gate_count']} "
            f"two-qubit gates; after structural transpilation to a generic "
            f"IBM basis gate set, depth is {hw_check['transpiled_depth']} "
            f"with {hw_check['transpiled_two_qubit_gate_count']} two-qubit "
            f"gates -- structurally reasonable for near-term hardware. "
            f"This is a separate question from whether the *trained "
            f"model* performs well, which it currently does not."
        ),
        "5_is_4_qubit_representation_still_appropriate": (
            "Unclear from this run alone. The failure-analysis checks "
            "point to an optimization/trainability limitation rather "
            "than a fundamental inability of 4 qubits to encode the "
            "signal (a classical model on the identical 4-feature, "
            "[0, pi]-scaled data reaches 0.939 accuracy, so the "
            "information is present). Per Phase 2 scope, the 6-qubit "
            "fallback and any architecture change are left as an "
            "explicit decision point for review, not applied "
            "automatically."
        ),
        "6_main_limitation": (
            "The trained VQC's default-threshold accuracy does not beat "
            "a trivial majority-class baseline, despite weak-to-moderate "
            "ranking signal (ROC-AUC above chance). Diagnosed as most "
            "likely an optimization-landscape / trainability limitation "
            "of a shallow, gradient-free-trained, shot-sampled VQC at "
            "this circuit size -- not a data, preprocessing, or "
            "measurement-interpretation bug."
        ),
        "7_what_to_test_next": (
            "Before proceeding to Phase 3 noise experiments on a circuit "
            "that does not yet train reliably: review the failure-"
            "analysis findings, and decide (as a project decision, not "
            "an automatic escalation) whether to (a) try a different "
            "feature map/ansatz pairing, (b) increase the optimizer "
            "budget further or try a gradient-based optimizer with an "
            "exact (statevector) training signal before moving to "
            "shot-sampled hardware-realistic training, (c) attempt the "
            "6-qubit backup feature set, or (d) proceed to Phase 3 "
            "anyway using this circuit as-is, since noise-robustness "
            "analysis is still meaningful even on a weak baseline "
            "classifier, as long as that is stated clearly rather than "
            "implied to be a strong baseline."
        ),
        "quantum_advantage_claim": (
            "None made, and importantly: this result does not even show "
            "the quantum model matching classical performance, let alone "
            "exceeding it. Reported honestly per project scientific-"
            "integrity principle, not adjusted or hidden."
        ),
    }

    with open(os.path.join(RESULTS_DIR, "phase2_scientific_interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2)

    print("Failure analysis and honest interpretation written.")
    print("\nOverall conclusion:")
    print(failure_analysis_results["overall_conclusion"])


if __name__ == "__main__":
    main()
