"""
robustness.py
==============
Multi-seed robustness evaluation. Generalizes the original Phase 2C
seed-robustness script into a reusable engine step for any model.

`run_fn(seed) -> dict(metrics)` re-runs the split -> train -> evaluate
pipeline end to end for a given random seed (so the seed changes the
train/test split, not just model initialization -- matching the original
project's definition of "robustness").
"""
from __future__ import annotations

import statistics

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_robustness_analysis(run_fn, seeds=None) -> dict:
    seeds = seeds or DEFAULT_SEEDS
    per_seed = []
    for seed in seeds:
        metrics = run_fn(seed)
        per_seed.append({"seed": seed, **metrics})

    summary = {}
    metric_names = [k for k in per_seed[0].keys() if k != "seed"] if per_seed else []
    for m in metric_names:
        values = [row[m] for row in per_seed if row[m] is not None]
        if not values:
            continue
        summary[m] = {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {"seeds": seeds, "per_seed_results": per_seed, "summary": summary}
