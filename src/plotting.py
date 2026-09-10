"""
plotting.py
============
Small set of deliberately minimal plots for Phase 1 -- understanding, not
presentation polish, per project scope. Every figure saved here is one
that directly informs a decision (class balance, feature redundancy,
selection agreement, or baseline performance).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120


def save_fig(fig, figures_dir, name):
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(y, target_names, figures_dir):
    counts = y.value_counts().sort_index()
    labels = [target_names[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts.values, color=["#c0392b", "#2980b9"])
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (Breast Cancer Wisconsin)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    return save_fig(fig, figures_dir, "class_distribution.png")


def plot_correlation_heatmap(X, figures_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = X.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap (all 30 features)")
    return save_fig(fig, figures_dir, "correlation_heatmap.png")


def plot_confusion_matrices(results_by_model, figures_dir, subset_label):
    n = len(results_by_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (model_name, result) in zip(axes, results_by_model.items()):
        cm = result["confusion_matrix"]
        matrix = np.array([[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred Malignant", "Pred Benign"],
                    yticklabels=["Actual Malignant", "Actual Benign"],
                    cbar=False)
        ax.set_title(model_name)
    fig.suptitle(f"Confusion Matrices -- {subset_label}")
    return save_fig(fig, figures_dir, f"confusion_matrices_{subset_label}.png")


def plot_roc_curves(results_by_model, figures_dir, subset_label):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, result in results_by_model.items():
        fpr = result["roc_curve"]["fpr"]
        tpr = result["roc_curve"]["tpr"]
        auc = result["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves -- {subset_label}")
    ax.legend(loc="lower right")
    return save_fig(fig, figures_dir, f"roc_curves_{subset_label}.png")
