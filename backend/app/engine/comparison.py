"""
comparison.py
==============
Builds the central Model Comparison table and identifies best
classical / best quantum / overall best model from ACTUAL measured
results. Never assumes quantum wins; never hard-codes a winner.
"""
from __future__ import annotations

import math

CLASSICAL_TYPES = {"Logistic Regression", "Random Forest"}
QUANTUM_TYPES = {"EfficientSU2 VQC", "RealAmplitudes VQC", "Quantum Kernel SVM"}


def build_comparison_table(model_results: dict) -> dict:
    """model_results: {model_name: {"type": "classical"|"quantum", "metrics": {...}, "training_time_seconds": float}}"""
    rows = []
    for name, r in model_results.items():
        m = r["metrics"]
        rows.append({
            "model": name,
            "type": r["type"],
            "accuracy": round(m["accuracy"], 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4) if math.isfinite(m["roc_auc"]) else None,
            "training_time_seconds": r.get("training_time_seconds"),
        })

    classical_rows = [r for r in rows if r["type"] == "classical"]
    quantum_rows = [r for r in rows if r["type"] == "quantum"]

    def _best(rows_subset):
        valid = [r for r in rows_subset if r["roc_auc"] is not None]
        pool = valid or rows_subset
        return max(pool, key=lambda r: (r["f1"], r["accuracy"])) if pool else None

    best_classical = _best(classical_rows)
    best_quantum = _best(quantum_rows)
    best_overall = _best(rows)

    if best_classical and best_quantum:
        if best_classical["f1"] >= best_quantum["f1"]:
            verdict = "Classical model currently provides the strongest performance on this dataset."
        else:
            verdict = (
                f"The quantum model '{best_quantum['model']}' measured a higher F1 score "
                f"({best_quantum['f1']:.4f}) than the best classical model on this dataset and split. "
                f"This is a measured result on this specific dataset, not a general claim of quantum advantage."
            )
    else:
        verdict = "Insufficient results to compare classical and quantum models yet."

    return {
        "rows": rows,
        "best_classical_model": best_classical["model"] if best_classical else None,
        "best_quantum_model": best_quantum["model"] if best_quantum else None,
        "best_overall_model": best_overall["model"] if best_overall else None,
        "verdict": verdict,
    }
