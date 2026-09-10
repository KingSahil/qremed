"""
classical_models.py
=====================
Trains the two classical baselines (Logistic Regression, Random Forest)
on a given (already feature-selected, already scaled) training set, and
evaluates once on the corresponding test set.

No hyperparameter tuning against the test set -- parameters come straight
from config.py.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, params):
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, params):
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_all_baselines(X_train, y_train, lr_params, rf_params):
    """Returns a dict of {model_name: fitted_model}."""
    return {
        "Logistic Regression": train_logistic_regression(X_train, y_train, lr_params),
        "Random Forest": train_random_forest(X_train, y_train, rf_params),
    }
