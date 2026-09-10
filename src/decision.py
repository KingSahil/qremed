"""
decision.py
============
Applies the Phase 1 decision criteria (plan Section 10-11): recommend a
PRIMARY and BACKUP feature set, biased toward the smaller (4-feature) set
unless it costs a scientifically meaningful amount of sensitivity or
specificity relative to the 6-feature set.

Thresholds are conservative and stated explicitly here (not buried in
run_phase1.py) so the recommendation logic itself is inspectable.
"""

SENSITIVITY_DROP_THRESHOLD = 0.02   # 2 percentage points
SPECIFICITY_DROP_THRESHOLD = 0.05   # 5 percentage points


def recommend_feature_set(results_df, selected_feature_sets):
    """
    results_df: DataFrame with columns [subset, model, accuracy, sensitivity,
                specificity, precision, f1, roc_auc]
    selected_feature_sets: {"4_features": [...], "6_features": [...]}
    """
    avg_by_subset = results_df.groupby("subset")[["sensitivity", "specificity", "f1", "accuracy"]].mean()

    sens_4 = avg_by_subset.loc["4_features", "sensitivity"]
    sens_6 = avg_by_subset.loc["6_features", "sensitivity"]
    spec_4 = avg_by_subset.loc["4_features", "specificity"]
    spec_6 = avg_by_subset.loc["6_features", "specificity"]

    sensitivity_drop = sens_6 - sens_4
    specificity_drop = spec_6 - spec_4

    meaningful_loss = (
        sensitivity_drop > SENSITIVITY_DROP_THRESHOLD
        or specificity_drop > SPECIFICITY_DROP_THRESHOLD
    )

    if meaningful_loss:
        primary, backup = "6_features", "4_features"
        rationale = (
            f"4-feature subset costs a meaningful amount of performance "
            f"(sensitivity drop={sensitivity_drop:.4f}, "
            f"specificity drop={specificity_drop:.4f}, "
            f"thresholds={SENSITIVITY_DROP_THRESHOLD}/{SPECIFICITY_DROP_THRESHOLD}), "
            f"so the 6-feature set is preferred despite the larger qubit count."
        )
    else:
        primary, backup = "4_features", "6_features"
        rationale = (
            f"4-feature subset maintains sensitivity/specificity within the "
            f"pre-registered thresholds of the 6-feature subset "
            f"(sensitivity drop={sensitivity_drop:.4f}, "
            f"specificity drop={specificity_drop:.4f}, "
            f"thresholds={SENSITIVITY_DROP_THRESHOLD}/{SPECIFICITY_DROP_THRESHOLD}), "
            f"so the smaller (4-qubit) set is preferred per the project's "
            f"'smallest useful circuit' bias."
        )

    decision = {
        "primary_feature_set_label": primary,
        "primary_features": selected_feature_sets[primary],
        "backup_feature_set_label": backup,
        "backup_features": selected_feature_sets[backup],
        "sensitivity_drop_6_minus_4": round(float(sensitivity_drop), 4),
        "specificity_drop_6_minus_4": round(float(specificity_drop), 4),
        "sensitivity_threshold": SENSITIVITY_DROP_THRESHOLD,
        "specificity_threshold": SPECIFICITY_DROP_THRESHOLD,
        "rationale": rationale,
        "avg_metrics_by_subset": avg_by_subset.round(4).to_dict(orient="index"),
    }
    return decision


def write_decision_markdown(decision, path):
    lines = [
        "# Feature Set Decision — Q-REMED Phase 1",
        "",
        f"**Primary feature set:** {decision['primary_feature_set_label']} "
        f"({len(decision['primary_features'])} features)",
    ]
    lines += [f"- {f}" for f in decision["primary_features"]]
    lines += [
        "",
        f"**Backup feature set:** {decision['backup_feature_set_label']} "
        f"({len(decision['backup_features'])} features)",
    ]
    lines += [f"- {f}" for f in decision["backup_features"]]
    lines += [
        "",
        "## Rationale",
        decision["rationale"],
        "",
        "## Average metrics by subset (across both classical models)",
        "",
    ]
    avg = decision["avg_metrics_by_subset"]
    lines.append("| Subset | Sensitivity | Specificity | F1 | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for subset, m in avg.items():
        lines.append(
            f"| {subset} | {m['sensitivity']:.4f} | {m['specificity']:.4f} "
            f"| {m['f1']:.4f} | {m['accuracy']:.4f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
