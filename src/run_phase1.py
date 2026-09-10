"""
run_phase1.py
==============
End-to-end Phase 1 pipeline for Q-REMED:

  dataset load + verification
    -> leakage-safe preprocessing (split first, scale on train only)
    -> feature ranking (ANOVA primary, Random Forest cross-check)
    -> 4-feature and 6-feature subset experiments
    -> classical baselines (Logistic Regression, Random Forest) per subset
    -> full metric set + confusion matrices + ROC curves
    -> save everything the quantum phase will need

No Qiskit / quantum / frontend / backend / database code -- Phase 1 scope
only, per project instruction.
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
import data_loader
import decision
import evaluation
import feature_selection
import plotting
import preprocessing
from classical_models import train_logistic_regression, train_random_forest


def main():
    plotting.setup_style()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load + verify dataset
    # ------------------------------------------------------------------
    X, y, target_names, feature_names = data_loader.load_dataset()
    report = data_loader.build_dataset_report(X, y, target_names, feature_names)
    data_loader.print_dataset_report(report)
    data_loader.save_dataset_report(report, config.RESULTS_DIR)

    # Verify malignant/benign encoding from the data itself (never assumed).
    malignant_label = [k for k, v in report["label_to_class_name"].items() if v == "malignant"][0]
    benign_label = [k for k, v in report["label_to_class_name"].items() if v == "benign"][0]
    print(f"\nVerified encoding: malignant={malignant_label}, benign={benign_label}\n")

    plotting.plot_class_distribution(y, target_names, config.FIGURES_DIR)
    plotting.plot_correlation_heatmap(X, config.FIGURES_DIR)

    # ------------------------------------------------------------------
    # 2. Leakage-safe split + scaling
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler = preprocessing.leakage_safe_preprocess(
        X, y, config.TEST_SIZE, config.RANDOM_SEED
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}\n")

    # ------------------------------------------------------------------
    # 3. Feature ranking (TRAIN only) -- ANOVA primary, RF cross-check
    # ------------------------------------------------------------------
    anova_df = feature_selection.anova_ranking(X_train, y_train)
    rf_df = feature_selection.random_forest_importance_ranking(
        X_train, y_train, config.RF_IMPORTANCE_PARAMS
    )
    agreement = feature_selection.compare_rankings(anova_df, rf_df, top_n=10)

    anova_df.to_csv(os.path.join(config.RESULTS_DIR, "feature_ranking_anova.csv"))
    rf_df.to_csv(os.path.join(config.RESULTS_DIR, "feature_ranking_rf_crosscheck.csv"))
    with open(os.path.join(config.RESULTS_DIR, "feature_ranking_agreement.json"), "w") as f:
        json.dump(agreement, f, indent=2)

    print("Top 10 features (ANOVA F-score, primary):")
    print(anova_df.head(10)[["feature", "score", "p_value"]].to_string())
    print(f"\nTop-10 agreement with Random Forest cross-check: {agreement['agreement_pct']}%\n")

    # ------------------------------------------------------------------
    # 4. Subset experiments: 4 features vs 6 features
    # ------------------------------------------------------------------
    all_results_rows = []
    selected_feature_sets = {}

    for n_features in config.N_FEATURES_OPTIONS:
        subset_label = f"{n_features}_features"
        selected = feature_selection.select_top_features(anova_df, n_features)
        selected_feature_sets[subset_label] = selected
        print(f"Selected {n_features}-feature subset: {selected}")

        X_train_sub = X_train[selected]
        X_test_sub = X_test[selected]

        results_by_model = {}
        lr_model = train_logistic_regression(X_train_sub, y_train, config.LOGISTIC_REGRESSION_PARAMS)
        rf_model = train_random_forest(X_train_sub, y_train, config.RANDOM_FOREST_PARAMS)
        models = {"Logistic Regression": lr_model, "Random Forest": rf_model}

        for model_name, model in models.items():
            result = evaluation.evaluate_model(
                model, X_test_sub, y_test,
                positive_label=malignant_label, negative_label=benign_label,
            )
            results_by_model[model_name] = result

            row = {"subset": subset_label, "model": model_name, **result["metrics"]}
            all_results_rows.append(row)

        plotting.plot_confusion_matrices(results_by_model, config.FIGURES_DIR, subset_label)
        plotting.plot_roc_curves(results_by_model, config.FIGURES_DIR, subset_label)

    results_df = pd.DataFrame(all_results_rows)
    results_df.to_csv(os.path.join(config.RESULTS_DIR, "baseline_metrics.csv"), index=False)
    print("\nBaseline results (all subsets):")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Save selected feature sets + full experiment config
    # ------------------------------------------------------------------
    with open(os.path.join(config.RESULTS_DIR, "selected_feature_sets.json"), "w") as f:
        json.dump(selected_feature_sets, f, indent=2)

    experiment_config = {
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "n_features_options": config.N_FEATURES_OPTIONS,
        "feature_selection_method_primary": config.FEATURE_SELECTION_METHOD,
        "feature_selection_method_crosscheck": config.FEATURE_SELECTION_CROSSCHECK_METHOD,
        "logistic_regression_params": config.LOGISTIC_REGRESSION_PARAMS,
        "random_forest_params": config.RANDOM_FOREST_PARAMS,
        "malignant_label": int(malignant_label),
        "benign_label": int(benign_label),
        "preprocessing": "StandardScaler fit on TRAIN only, applied to TRAIN and TEST",
        "split": "stratified 80/20 train/test, fixed random_state",
    }
    with open(os.path.join(config.RESULTS_DIR, "experiment_config.json"), "w") as f:
        json.dump(experiment_config, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Primary / backup feature-set decision (plan Section 10-11)
    # ------------------------------------------------------------------
    feature_decision = decision.recommend_feature_set(results_df, selected_feature_sets)
    with open(os.path.join(config.RESULTS_DIR, "feature_set_decision.json"), "w") as f:
        json.dump(feature_decision, f, indent=2)
    decision.write_decision_markdown(
        feature_decision, os.path.join(config.RESULTS_DIR, "feature_set_decision.md")
    )
    print(f"\nDecision: PRIMARY = {feature_decision['primary_feature_set_label']}, "
          f"BACKUP = {feature_decision['backup_feature_set_label']}")
    print(feature_decision["rationale"])

    print("\nPhase 1 complete. Results saved to results/, figures saved to figures/.")
    return results_df, selected_feature_sets, agreement, feature_decision


if __name__ == "__main__":
    main()
