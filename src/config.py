"""
config.py
=========
Centralized experiment configuration for Q-REMED Phase 1
(dataset + leakage-safe preprocessing + classical baseline).

No magic numbers should appear elsewhere in the pipeline -- everything
that controls reproducibility or experiment behavior lives here.
"""

RANDOM_SEED = 42

TEST_SIZE = 0.2  # 80% train / 20% test, stratified

# Candidate quantum-circuit sizes to evaluate (4 qubits vs 6 qubits)
N_FEATURES_OPTIONS = [4, 6]

# Primary feature-selection method. Random Forest importance is always
# computed too, but only as a cross-check -- it never drives the actual
# feature selection used downstream.
FEATURE_SELECTION_METHOD = "anova_f_score"
FEATURE_SELECTION_CROSSCHECK_METHOD = "random_forest_importance"

LOGISTIC_REGRESSION_PARAMS = {
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
}

# Random Forest used purely for the feature-importance cross-check
# (kept separate from the baseline classifier's own Random Forest so the
# two roles never get conflated, even though the hyperparameters happen
# to match).
RF_IMPORTANCE_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
}

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
