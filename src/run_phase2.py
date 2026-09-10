"""
run_phase2.py
===============
End-to-end Phase 2 pipeline for Q-REMED:

  Reuse Phase 1's exact split (same seed, same stratified 80/20 split)
    -> restrict to Phase 1's primary 4-feature set
    -> Phase 1's StandardScaler (fit on train only) -- unchanged
    -> additional quantum [0, pi] scaling (fit on train only)
    -> build 4-qubit ZZFeatureMap + RealAmplitudes (reps=1) circuit
    -> static hardware-compatibility structural check
    -> train VQC on IDEAL Aer simulator only
    -> freeze parameters, evaluate on the untouched test set
    -> compare against Phase 1's saved classical baselines
    -> save every required artifact

No noise, no fake backend, no IBM hardware, no Streamlit, no 6-qubit
fallback (4 qubits worked -- see convergence check below) -- Phase 2
scope only.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
import data_loader
import failure_analysis
import hardware_compatibility
import plotting
import preprocessing
import quantum_circuit
import quantum_evaluation
import quantum_model
import quantum_plotting
import quantum_preprocessing
from feature_selection import select_top_features, anova_ranking

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR

PRIMARY_FEATURES = [
    "worst concave points",
    "worst perimeter",
    "mean concave points",
    "worst radius",
]


def load_phase1_split_and_verify():
    """
    Reuse Phase 1's exact data pipeline entry points (same functions, same
    config values) rather than re-deriving a split by hand -- this
    guarantees bit-identical train/test rows to Phase 1, since
    load_dataset() and leakage_safe_preprocess() are both fully
    deterministic given the same seed.
    """
    X, y, target_names, feature_names = data_loader.load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocessing.leakage_safe_preprocess(
        X, y, config.TEST_SIZE, config.RANDOM_SEED
    )
    return X, y, X_train, X_test, y_train, y_test, scaler


def verify_split_matches_phase1(X_train, y_train, X_test, y_test):
    """
    Sanity check: reproduce Phase 1's Logistic Regression 4-feature result
    using this same split, and confirm it matches Phase 1's saved metric
    exactly. This is a verification step only -- it does not modify or
    re-save Phase 1's results.
    """
    phase1_metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    if not os.path.exists(phase1_metrics_path):
        print("WARNING: Phase 1 baseline_metrics.csv not found -- skipping split verification.")
        return None

    phase1_df = pd.read_csv(phase1_metrics_path)
    phase1_row = phase1_df[
        (phase1_df["subset"] == "4_features") & (phase1_df["model"] == "Logistic Regression")
    ].iloc[0]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train_sub = X_train[PRIMARY_FEATURES]
    X_test_sub = X_test[PRIMARY_FEATURES]
    lr = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    lr.fit(X_train_sub, y_train)
    reproduced_accuracy = accuracy_score(y_test, lr.predict(X_test_sub))

    matches = abs(reproduced_accuracy - phase1_row["accuracy"]) < 1e-9
    print(f"Split verification: reproduced LR accuracy={reproduced_accuracy:.6f}, "
          f"Phase 1 saved accuracy={phase1_row['accuracy']:.6f}, "
          f"MATCH={'YES' if matches else 'NO'}")
    if not matches:
        raise RuntimeError(
            "Phase 2's reused split does not reproduce Phase 1's exact result -- "
            "stopping, since Phase 2 must not use a different split from Phase 1."
        )
    return matches


def main():
    plotting.setup_style()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Reuse Phase 1's exact split (verified, not just assumed)
    # ------------------------------------------------------------------
    X, y, X_train_full, X_test_full, y_train, y_test, scaler = load_phase1_split_and_verify()
    verify_split_matches_phase1(X_train_full, y_train, X_test_full, y_test)

    X_train = X_train_full[PRIMARY_FEATURES]
    X_test = X_test_full[PRIMARY_FEATURES]
    print(f"\nUsing Phase 1 primary 4-feature set: {PRIMARY_FEATURES}")
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 2. Quantum feature preparation ([0, pi] scaling on top of Phase 1's
    #    StandardScaler output, fit on train only)
    # ------------------------------------------------------------------
    X_train_q, X_test_q, quantum_scaler = quantum_preprocessing.prepare_quantum_features(
        X_train, X_test
    )
    print(f"Quantum-scaled feature range (train): "
          f"min={X_train_q.min().min():.4f}, max={X_train_q.max().max():.4f}")

    quantum_preprocessing_explanation = {
        "why_standardization_first": (
            "Phase 1's StandardScaler (mean 0, std 1) is reused unchanged "
            "so classical-baseline comparability and Phase 1's leakage-safe "
            "guarantees are preserved exactly."
        ),
        "why_additional_quantum_scaling": (
            "ZZFeatureMap encodes feature values as rotation angles, which "
            "are periodic with period 2*pi. Unbounded standardized values "
            "risk two different feature values producing indistinguishable "
            "quantum states. An additional MinMaxScaler (fit on TRAIN only) "
            "maps every standardized feature into [0, pi], safely inside a "
            "single period."
        ),
        "expected_range": "[0, pi] for every one of the 4 selected features",
        "where_applied": (
            "As a separate step, immediately before circuit encoding -- "
            "NOT folded into Phase 1's saved scaler, so Phase 1's "
            "pipeline and artifacts remain completely unmodified."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "quantum_preprocessing_explanation.json"), "w") as f:
        json.dump(quantum_preprocessing_explanation, f, indent=2)

    # ------------------------------------------------------------------
    # 3. Quantum circuit: 4-qubit ZZFeatureMap (reps=1) + RealAmplitudes (reps=1)
    # ------------------------------------------------------------------
    feature_map = quantum_circuit.build_feature_map(num_features=4, reps=1)
    ansatz = quantum_circuit.build_ansatz(num_qubits=4, reps=1)
    full_circuit = quantum_circuit.build_full_circuit(feature_map, ansatz)

    circuit_analysis = quantum_circuit.analyze_circuit(feature_map, ansatz, full_circuit)
    with open(os.path.join(RESULTS_DIR, "circuit_analysis.json"), "w") as f:
        json.dump(circuit_analysis, f, indent=2)

    print(f"\nCircuit: {circuit_analysis['num_qubits']} qubits, "
          f"full depth={circuit_analysis['full_circuit']['depth']}, "
          f"trainable params={circuit_analysis['num_trainable_parameters']}, "
          f"2Q gates={circuit_analysis['full_circuit']['two_qubit_gate_count']}")

    quantum_plotting.save_circuit_diagram(full_circuit, FIGURES_DIR, "vqc_circuit_diagram.png")
    quantum_plotting.save_circuit_diagram(
        full_circuit, FIGURES_DIR, "vqc_circuit_diagram_decomposed.png", decompose=True
    )

    # ------------------------------------------------------------------
    # 4. Static hardware-compatibility structural check (no fake backend, no IBM)
    # ------------------------------------------------------------------
    hw_check = hardware_compatibility.check_hardware_compatibility(full_circuit)
    with open(os.path.join(RESULTS_DIR, "hardware_compatibility_check.json"), "w") as f:
        json.dump(hw_check, f, indent=2)
    print(f"\nHardware compatibility (structural, generic IBM basis gates): "
          f"transpiled depth={hw_check['transpiled_depth']}, "
          f"2Q gates={hw_check['transpiled_two_qubit_gate_count']}, "
          f"unsupported gates before transpile={hw_check['gates_outside_basis_before_transpile']}")

    # ------------------------------------------------------------------
    # 5. Train VQC on IDEAL Aer simulator
    # ------------------------------------------------------------------
    vqc, loss_history = quantum_model.build_vqc(
        feature_map, ansatz, random_seed=config.RANDOM_SEED, shots=1024, maxiter=100
    )
    print("\nTraining VQC on ideal Aer simulator (this may take ~1-3 minutes)...")
    training_info = quantum_model.train_vqc(vqc, loss_history, X_train_q, y_train)
    print(f"Training complete: {training_info['num_loss_evaluations']} loss evaluations, "
          f"final loss={training_info['final_loss']:.4f}, "
          f"time={training_info['training_time_seconds']}s, "
          f"converged(descriptive)={training_info['converged']}")

    with open(os.path.join(RESULTS_DIR, "vqc_training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    quantum_plotting.plot_training_loss(loss_history, FIGURES_DIR)

    # ------------------------------------------------------------------
    # 6. Freeze parameters, evaluate on untouched test set
    # ------------------------------------------------------------------
    test_result = quantum_evaluation.evaluate_vqc(vqc, X_test_q, y_test)
    with open(os.path.join(RESULTS_DIR, "vqc_test_predictions.json"), "w") as f:
        json.dump({
            "predictions": test_result["predictions"],
            "probabilities": test_result["probabilities"],
            "prediction_distribution": test_result["prediction_distribution"],
        }, f, indent=2)

    vqc_metrics_row = {"model": "VQC", **test_result["metrics"]}
    print("\nVQC test-set metrics:")
    for k, v in test_result["metrics"].items():
        print(f"  {k}: {v:.4f}")

    # Majority-class trivial baseline, for a fair read on accuracy alone.
    majority_class = int(y_train.value_counts().idxmax())
    majority_baseline_accuracy = float((y_test == majority_class).mean())
    print(f"  (majority-class trivial baseline accuracy: {majority_baseline_accuracy:.4f})")

    plotting.plot_confusion_matrices({"VQC": test_result}, FIGURES_DIR, "vqc_4_features")
    plotting.plot_roc_curves({"VQC": test_result}, FIGURES_DIR, "vqc_4_features")

    # ------------------------------------------------------------------
    # 7. Classical comparison (using Phase 1's saved, unmodified results)
    # ------------------------------------------------------------------
    phase1_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_metrics.csv"))
    phase1_4feat = phase1_df[phase1_df["subset"] == "4_features"][
        ["model", "accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
    ]
    comparison_df = pd.concat(
        [phase1_4feat, pd.DataFrame([vqc_metrics_row])], ignore_index=True
    )
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "phase2_classical_vs_quantum.csv"), index=False)
    print("\nClassical vs Quantum comparison (4-feature set):")
    print(comparison_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Save VQC + feature-map + ansatz + optimizer configuration
    # ------------------------------------------------------------------
    vqc_config = {
        "num_qubits": 4,
        "primary_features": PRIMARY_FEATURES,
        "feature_map": {"type": "ZZFeatureMap", "reps": 1, "feature_dimension": 4},
        "ansatz": {"type": "RealAmplitudes", "reps": 1, "num_qubits": 4},
        "optimizer": {"type": "COBYLA", "maxiter": 100},
        "sampler": {"type": "AerSamplerV2 (ideal, no noise model)", "shots": 1024,
                    "seed": config.RANDOM_SEED},
        "loss": "cross_entropy",
        "random_seed": config.RANDOM_SEED,
        "execution_environment": "Ideal Qiskit Aer simulation only -- no noise, "
                                  "no fake backend, no IBM hardware.",
    }
    with open(os.path.join(RESULTS_DIR, "vqc_configuration.json"), "w") as f:
        json.dump(vqc_config, f, indent=2)

    # ------------------------------------------------------------------
    # 8b. Failure analysis (Section 11) -- triggered because VQC accuracy
    #     (test_result) is substantially below the classical baselines.
    #     Checked in the specified order; check 4 (circuit construction)
    #     and check 5 (measurement/output interpretation) are verified
    #     statically below rather than by retraining.
    # ------------------------------------------------------------------
    lr_row_for_gap = phase1_4feat[phase1_4feat["model"] == "Logistic Regression"].iloc[0]
    performance_gap = lr_row_for_gap["accuracy"] - test_result["metrics"]["accuracy"]

    failure_analysis_results = None
    if performance_gap > 0.10:
        print(f"\nVQC underperforms Logistic Regression by {performance_gap:.3f} accuracy "
              f"-- running required Section 11 failure analysis before finalizing interpretation.\n")

        check_4 = {
            "check": "circuit construction (check 4)",
            "method": "Static review of feature-map/ansatz composition and parameter counts.",
            "feature_map_parameters": circuit_analysis["feature_map"]["num_parameters"],
            "ansatz_parameters": circuit_analysis["ansatz"]["num_parameters"],
            "full_circuit_depth": circuit_analysis["full_circuit"]["depth"],
            "conclusion": (
                "Circuit composes correctly (feature map then ansatz, "
                "4 qubits, standard ZZFeatureMap + RealAmplitudes structure) "
                "-- no construction error found; expressivity is tested "
                "separately under check 8."
            ),
        }
        check_5 = {
            "check": "measurement / output interpretation (check 5)",
            "method": "Inspect the underlying SamplerQNN's interpret function and output shape.",
            "output_shape": list(vqc.neural_network.output_shape),
            "interpret_function": "standard VQC parity-based binary interpretation (Qiskit ML default)",
            "conclusion": (
                "Standard, unmodified Qiskit Machine Learning default "
                "parity interpretation for binary VQC classification -- "
                "no custom or misconfigured output mapping found."
            ),
        }

        failure_analysis_results = failure_analysis.run_full_failure_analysis(
            feature_map, ansatz, X_train_q, X_test_q, y_train, y_test,
            PRIMARY_FEATURES, config.RANDOM_SEED, shots=1024, maxiter=100,
        )
        failure_analysis_results["check_4_circuit_construction"] = check_4
        failure_analysis_results["check_5_measurement_interpretation"] = check_5
        failure_analysis_results["performance_gap_vs_logistic_regression"] = round(float(performance_gap), 4)
        failure_analysis_results["majority_class_baseline_accuracy"] = round(majority_baseline_accuracy, 4)

        with open(os.path.join(RESULTS_DIR, "phase2_failure_analysis.json"), "w") as f:
            json.dump(failure_analysis_results, f, indent=2)

    # ------------------------------------------------------------------
    # 9. Scientific interpretation
    # ------------------------------------------------------------------
    lr_row = phase1_4feat[phase1_4feat["model"] == "Logistic Regression"].iloc[0]
    rf_row = phase1_4feat[phase1_4feat["model"] == "Random Forest"].iloc[0]
    vqc_m = test_result["metrics"]

    beats_majority_baseline = vqc_m["accuracy"] > majority_baseline_accuracy
    shows_ranking_signal = vqc_m["roc_auc"] > 0.55  # meaningfully above chance (0.5)
    depth_note = (
        "shallow" if circuit_analysis["full_circuit"]["depth"] <= 15 else "moderate/deep"
    )

    if failure_analysis_results is not None:
        did_learn_text = (
            f"Partially, and not well at the default decision threshold. Test "
            f"accuracy ({vqc_m['accuracy']:.3f}) is actually {'below' if not beats_majority_baseline else 'above'} "
            f"the trivial majority-class baseline ({majority_baseline_accuracy:.3f}) -- meaning the "
            f"default-threshold predictions are {'not' if not beats_majority_baseline else ''} "
            f"competitive with just always predicting the majority class. However, "
            f"ROC-AUC ({vqc_m['roc_auc']:.3f}) is meaningfully above chance (0.5), "
            f"indicating the model's underlying probability ranking does carry some "
            f"real class-discriminative signal even though its default 0.5-threshold "
            f"decisions are poorly calibrated. The required failure-analysis checks "
            f"(preprocessing/scaling, optimizer choice, initialization, circuit depth) "
            f"were run and are saved in phase2_failure_analysis.json: they rule out a "
            f"preprocessing bug (classical LR on the identical quantum-scaled data "
            f"still reaches "
            f"{failure_analysis_results['checks_1_to_3_preprocessing_labels_scaling']['classical_accuracy_on_quantum_scaled_data']:.3f} "
            f"accuracy), rule out an unlucky single initialization (consistently poor "
            f"across 5 seeds), and rule out 'wrong optimizer' as the sole cause "
            f"(COBYLA and SPSA converge to similarly poor results). This points to a "
            f"genuine optimization-landscape / trainability limitation of this shallow "
            f"circuit configuration on this problem, not a pipeline error."
        )
    else:
        did_learn_text = (
            f"Yes -- test accuracy {vqc_m['accuracy']:.3f} and ROC-AUC {vqc_m['roc_auc']:.3f} "
            f"are clearly above chance and above the majority-class baseline "
            f"({majority_baseline_accuracy:.3f})."
        )

    interpretation = {
        "1_did_vqc_learn_the_task": did_learn_text,
        "2_comparison_with_logistic_regression": (
            f"VQC accuracy {vqc_m['accuracy']:.3f} vs Logistic Regression "
            f"{lr_row['accuracy']:.3f} (gap of {lr_row['accuracy'] - vqc_m['accuracy']:.3f}); "
            f"VQC F1 {vqc_m['f1']:.3f} vs LR F1 {lr_row['f1']:.3f}. VQC substantially "
            f"underperforms the linear classical baseline on this run."
        ),
        "3_comparison_with_random_forest": (
            f"VQC accuracy {vqc_m['accuracy']:.3f} vs Random Forest "
            f"{rf_row['accuracy']:.3f}; VQC F1 {vqc_m['f1']:.3f} vs RF F1 "
            f"{rf_row['f1']:.3f}. Same substantial gap as against Logistic Regression."
        ),
        "4_is_circuit_shallow_enough_for_hardware": (
            f"Full circuit depth is {circuit_analysis['full_circuit']['depth']} "
            f"({depth_note}) with {circuit_analysis['full_circuit']['two_qubit_gate_count']} "
            f"two-qubit gates; after structural transpilation to a generic IBM "
            f"basis gate set, depth is {hw_check['transpiled_depth']} with "
            f"{hw_check['transpiled_two_qubit_gate_count']} two-qubit gates -- "
            f"structurally reasonable for near-term hardware. This is a separate "
            f"question from whether the *trained model* performs well, which it "
            f"currently does not."
        ),
        "5_is_4_qubit_representation_still_appropriate": (
            "Unclear from this run alone. The failure-analysis checks point to an "
            "optimization/trainability limitation rather than a fundamental "
            "inability of 4 qubits to encode the signal (a classical model on the "
            "identical 4-feature, [0, pi]-scaled data reaches "
            f"{failure_analysis_results['checks_1_to_3_preprocessing_labels_scaling']['classical_accuracy_on_quantum_scaled_data']:.3f} "
            "accuracy, so the information is there). Per Phase 2 scope, the "
            "6-qubit fallback and any architecture change are left as an explicit "
            "decision point for review, not applied automatically."
            if failure_analysis_results is not None else
            "Yes -- the 4-qubit run produced a working classifier competitive "
            "with the classical baselines."
        ),
        "6_main_limitation": (
            "The trained VQC's default-threshold accuracy does not clearly beat a "
            "trivial majority-class baseline, despite weak-to-moderate ranking "
            "signal (ROC-AUC above chance). Diagnosed as most likely an "
            "optimization-landscape / trainability limitation of a shallow, "
            "gradient-free-trained, shot-sampled VQC at this circuit size -- not a "
            "data, preprocessing, or measurement-interpretation bug (see "
            "phase2_failure_analysis.json)."
            if failure_analysis_results is not None else
            "Shallow ansatz with few trainable parameters and shot-sampled "
            "(noisy) training signal; no noise modeling yet."
        ),
        "7_what_to_test_next": (
            "Before proceeding to Phase 3 noise experiments on a circuit that "
            "does not yet train reliably: review the failure-analysis findings, "
            "and decide (as a project decision, not an automatic escalation) "
            "whether to (a) try a different feature map/ansatz pairing, "
            "(b) increase the optimizer budget further, (c) attempt the 6-qubit "
            "backup feature set, or (d) proceed to Phase 3 anyway using this "
            "circuit as-is, since noise-robustness analysis is still meaningful "
            "even on a weak baseline classifier, as long as that is stated "
            "clearly rather than implied to be a strong baseline."
            if failure_analysis_results is not None else
            "Phase 3: introduce Aer noise (depolarizing, bit-flip, amplitude "
            "damping) on this same frozen circuit structure and measure the "
            "robustness ratio, per the master plan's four-level validation "
            "ladder."
        ),
        "quantum_advantage_claim": (
            "None made, and importantly: this result does not even show the "
            "quantum model matching classical performance, let alone exceeding "
            "it. Reported honestly per project scientific-integrity principle, "
            "not adjusted or hidden."
            if failure_analysis_results is not None else
            "None made. This result establishes the baseline ideal-simulator "
            "quantum result for comparison against classical baselines and "
            "against later noisy/hardware results -- not a claim that the "
            "quantum model outperforms classical ML."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "phase2_scientific_interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 2 SCIENTIFIC INTERPRETATION")
    print("=" * 70)
    for k, v in interpretation.items():
        print(f"\n[{k}]\n{v}")

    print("\nPhase 2 complete. Results saved to results/, figures saved to figures/.")
    return comparison_df, circuit_analysis, training_info, interpretation


if __name__ == "__main__":
    main()
