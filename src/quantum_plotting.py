"""
quantum_plotting.py
======================
Phase-2-specific plots (training loss curve, circuit diagram). Confusion
matrix and ROC curve plots reuse Phase 1's plotting.py helpers directly
(same result-dict shape), rather than duplicating that code.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_loss(loss_history, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker=".", markersize=3)
    ax.set_xlabel("Optimizer evaluation")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("VQC Training Loss (ideal Aer simulator)")
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, "vqc_training_loss.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_circuit_diagram(circuit, figures_dir, filename, decompose=False):
    os.makedirs(figures_dir, exist_ok=True)
    draw_circuit = circuit.decompose() if decompose else circuit
    fig = draw_circuit.draw("mpl", fold=-1)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
