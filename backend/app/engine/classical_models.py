"""
classical_models.py
=====================
Classical baselines: Logistic Regression, Random Forest, and GPU-accelerated XGBoost.
Automatically selects GPU (CUDA) if an NVIDIA GPU is present on the machine,
or falls back to multi-core CPU execution.
"""
from __future__ import annotations

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .device_detection import detect_gpu

DEFAULT_LR_PARAMS = {"max_iter": 1000}
DEFAULT_RF_PARAMS = {"n_estimators": 200, "n_jobs": 1}


def train_all_baselines(X_train, y_train, random_state: int = 42, prefer_device: str = "auto"):
    lr_params = {**DEFAULT_LR_PARAMS, "random_state": random_state}
    rf_params = {**DEFAULT_RF_PARAMS, "random_state": random_state}

    gpu_info = detect_gpu()
    use_gpu = gpu_info["has_gpu"] and (prefer_device in ("auto", "gpu"))

    models = {}

    # 1. XGBoost (GPU if available, else CPU)
    try:
        import xgboost as xgb
        xgb_device = "cuda" if use_gpu else "cpu"
        xgb_name = f"XGBoost (GPU - {gpu_info['gpu_name']})" if use_gpu else "XGBoost (CPU)"
        
        xgb_model = xgb.XGBClassifier(
            device=xgb_device,
            tree_method="hist",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
        )
        t0 = time.time()
        xgb_model.fit(X_train, y_train)
        elapsed = time.time() - t0
        models[xgb_name] = {
            "model": xgb_model,
            "training_time_seconds": round(elapsed, 4),
            "device": "GPU (CUDA)" if use_gpu else "CPU",
            "accelerator": gpu_info["gpu_name"] if use_gpu else "CPU",
        }
    except Exception as e:
        print(f"XGBoost training skipped or failed: {e}")

    # 2. Random Forest (Multi-core CPU)
    t0 = time.time()
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    elapsed = time.time() - t0
    models["Random Forest"] = {
        "model": rf_model,
        "training_time_seconds": round(elapsed, 4),
        "device": "CPU (Multi-threaded)",
        "accelerator": "CPU Multi-Core",
    }

    # 3. Logistic Regression
    t0 = time.time()
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train, y_train)
    elapsed = time.time() - t0
    models["Logistic Regression"] = {
        "model": lr_model,
        "training_time_seconds": round(elapsed, 4),
        "device": "CPU",
        "accelerator": "CPU",
    }

    return models
