# Q-REMED — Quantum vs Classical ML for Breast Cancer Diagnosis

Q-REMED benchmarks classical machine learning against quantum machine
learning (variational circuits and quantum kernels) on the Breast Cancer
Wisconsin (Diagnostic) dataset, with an emphasis on leakage-safe
methodology, honest reporting of quantum underperformance, and a
documented recovery process rather than a cherry-picked final number.

The project runs in four stages: **Phase 1** (classical baseline +
feature selection), **Phase 2** (first variational quantum classifier,
VQC), **Phase 2B** (diagnosing and recovering from the VQC's poor initial
result), and **Phase 2C** (robustness checks + a quantum kernel model).

## TL;DR results

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (classical) | 0.930 | 0.952 | 0.917 | 0.909 | 0.991 |
| Random Forest (classical) | 0.930 | 0.929 | 0.931 | 0.907 | 0.986 |
| VQC, ZZFeatureMap+RealAmplitudes (Phase 2, original) | 0.596 | 0.333 | 0.750 | 0.378 | 0.639 |
| VQC, ZZFeatureMap+EfficientSU2 (Phase 2B, recovered) | 0.807 | 0.571 | 0.944 | 0.686 | 0.886 |
| VQC, EfficientSU2 + validation-tuned threshold (Phase 2C) | 0.807 | 0.881 | 0.764 | 0.771 | 0.877 |
| Quantum Kernel SVM (Phase 2C, 80/455 training subsample) | 0.877 | 0.690 | 0.986 | 0.806 | 0.924 |

**Headline finding:** the first VQC configuration badly underperformed
the classical baselines. Rather than hide or explain that away, the
project ran a pre-registered failure analysis (ruled out data,
preprocessing, and measurement bugs) and a pre-registered recovery
protocol (ruled out shot noise and optimizer budget as the cause;
identified the ansatz choice as the fix). The gap to classical ML
narrows substantially but is not closed — that residual gap is reported
as a real, current limitation, not smoothed over.

## Repository layout

```
q_remed_phase1_2_2b_2c/
├── README.md                          <- this file
├── Q_REMED_Results_Walkthrough.ipynb  <- notebook reproducing the key results/figures
├── src/                                <- pipeline source code
│   ├── run_phase1.py                   <- entry point: classical pipeline
│   ├── run_phase2.py                   <- entry point: first VQC pipeline
│   ├── finalize_phase2.py              <- Phase 2 failure-analysis + honest write-up
│   ├── finalize_phase2b.py             <- Phase 2B recovery decision + comparison table
│   ├── finalize_phase2c.py             <- Phase 2C robustness/kernel write-up
│   ├── config.py                       <- all experiment constants (single source of truth)
│   ├── data_loader.py                  <- dataset loading + verification report
│   ├── preprocessing.py                <- leakage-safe split + scaling
│   ├── feature_selection.py            <- ANOVA (primary) + Random Forest (cross-check) ranking
│   ├── decision.py                     <- Phase 1 primary/backup feature-set rule
│   ├── classical_models.py             <- Logistic Regression / Random Forest training
│   ├── evaluation.py                   <- shared metric computation (classical + quantum)
│   ├── quantum_preprocessing.py        <- [0, π] rescaling for angle encoding
│   ├── quantum_circuit.py              <- ZZFeatureMap / RealAmplitudes / EfficientSU2 circuits
│   ├── quantum_model.py                <- VQC construction + training loop
│   ├── quantum_kernel.py               <- quantum kernel construction for the kernel SVM
│   ├── quantum_evaluation.py           <- VQC test-set evaluation
│   ├── exact_training.py               <- statevector (noise-free) VQC training variant
│   ├── failure_analysis.py             <- Phase 2 failure-analysis checks
│   ├── hardware_compatibility.py       <- static transpile check against an IBM basis gate set
│   ├── plotting.py / quantum_plotting.py <- confusion matrices, ROC curves, circuit diagrams
├── results/                            <- every JSON/CSV artifact the pipeline produces
└── figures/                            <- every PNG the pipeline produces
```

## The four phases

### Phase 1 — Classical baseline (`run_phase1.py`)
- Loads the Breast Cancer Wisconsin dataset (569 samples, 30 features,
  malignant/benign) via scikit-learn and verifies the label encoding
  from the data itself rather than assuming it.
- Leakage-safe pipeline: stratified 80/20 split **first**, then a
  `StandardScaler` fit on the training fold only.
- Feature ranking on the training fold only: ANOVA F-score (primary),
  Random Forest importance (cross-check only — never drives selection).
- Trains Logistic Regression and Random Forest on a 4-feature and a
  6-feature subset, and picks a **primary** feature set using a
  pre-registered rule (prefer the smaller/4-feature set unless it costs
  more than 2 pts of sensitivity or 5 pts of specificity vs. the
  6-feature set). Result: the 4-feature set wins
  (`worst concave points`, `worst perimeter`, `mean concave points`,
  `worst radius`).

### Phase 2 — First variational quantum classifier (`run_phase2.py`, `finalize_phase2.py`)
- Reuses Phase 1's exact train/test split (verified bit-for-bit, not
  just re-derived from the same seed) and its primary 4-feature set.
- Adds a second, quantum-specific `[0, π]` scaling on top of Phase 1's
  standardization, so periodic angle encoding never wraps two different
  feature values onto the same quantum state.
- Circuit: 4-qubit `ZZFeatureMap` (reps=1) + `RealAmplitudes` ansatz
  (reps=1), trained with COBYLA on an ideal (noise-free) Aer simulator.
- Result: the VQC substantially underperforms both classical baselines
  and does not clearly beat a majority-class trivial baseline at the
  default 0.5 threshold, despite an above-chance ROC-AUC.
- A pre-registered failure analysis was triggered (gap > 0.10 accuracy)
  and rules out: preprocessing/label/scaling bugs, circuit-construction
  errors, and measurement-interpretation errors — pointing instead to a
  genuine optimization/trainability limitation of this specific
  circuit.

### Phase 2B — Diagnosing and recovering trainability (`finalize_phase2b.py`)
Three candidate fixes were tested against three pre-registered cases:
1. **Exact/statevector training** (removes shot noise) — barely moved
   accuracy → shot noise was not the cause.
2. **Extended optimizer budget** (232 effective evaluations vs. 100) —
   barely moved accuracy, and the optimizer kept re-converging early →
   optimizer budget was not the cause.
3. **Alternative ansatz** (`EfficientSU2` instead of `RealAmplitudes`) —
   accuracy rose from 0.596 to 0.807, F1 from 0.378 to 0.686, ROC-AUC
   from 0.639 to 0.886 → **this was the fix**.

Because Case 3 succeeded, the 6-qubit fallback (only triggered if all
three cases fail) was correctly skipped. `EfficientSU2` is adopted as
the new candidate circuit going forward.

### Phase 2C — Robustness and a quantum kernel model (`finalize_phase2c.py`)
- **Seed robustness:** the recovered VQC was retrained across 5 seeds;
  accuracy ranged 0.72–0.86 (std ≈ 0.05) — no seed catastrophically
  failed, but there is real run-to-run variance the single headline
  number hides.
- **Validation-selected decision threshold:** choosing the
  classification threshold on an internal validation split (not the
  test set) to maximize F1 raised sensitivity from 0.571 to 0.881 at a
  real, reported specificity cost (0.944 → 0.764).
- **Quantum kernel SVM:** a fidelity-based quantum kernel (trained on an
  80-sample stratified subsample for computational feasibility) reached
  0.877 accuracy / 0.924 ROC-AUC — the strongest quantum result in the
  project, though still trained on much less data than the classical
  models and still short of them on sensitivity.
- The `EfficientSU2` VQC (not the kernel model) is recommended to carry
  forward into hardware/noise work, because its inference is a small
  fixed circuit that fits a realistic hardware shot budget, whereas the
  kernel method needs one circuit evaluation per train/test pair.

## Setup and Installation

### 1. Create a Virtual Environment

Using **uv** (recommended):
```bash
uv venv
# On Windows PowerShell:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Or using standard **Python venv**:
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

Install all core and quantum packages from the repository root:

```bash
# Using uv
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

> **Note:** If installing purely for Phase 1 (classical baseline), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn` are sufficient. The quantum phases (Phase 2, 2B, 2C) require `qiskit`, `qiskit-aer`, and `qiskit-machine-learning`.

## Reproducing the Pipeline

Execute the pipeline stages in order from `src/`:

```bash
cd src

# Phase 1: Classical baseline + ANOVA feature selection
python run_phase1.py          # writes results/ + figures/ for Phase 1

# Phase 2: First VQC (ZZFeatureMap + RealAmplitudes) + Failure Analysis
python run_phase2.py          # writes results/ + figures/ for Phase 2
python finalize_phase2.py     # Phase 2 failure analysis + honest interpretation

# Phase 2B: Trainability Recovery (EfficientSU2 ansatz)
python finalize_phase2b.py    # Phase 2B recovery decision + comparison table

# Phase 2C: Robustness Checks + Threshold Tuning + Quantum Kernel SVM
python finalize_phase2c.py    # Phase 2C robustness + kernel writeup
```

Alternatively, you can run them directly from the project root using `uv`:
```bash
uv run src/run_phase1.py
uv run src/run_phase2.py
uv run src/finalize_phase2.py
uv run src/finalize_phase2b.py
uv run src/finalize_phase2c.py
```

## Reading the results

`Q_REMED_Results_Walkthrough.ipynb` walks through the pipeline stage by
stage using the artifacts already saved in `results/` and `figures/`
(it does not require Qiskit or a live training run to execute) and ends
with the final comparison table and scientific interpretation. For
programmatic access, every stage's numeric output is also saved directly
under `results/` (CSV tables and JSON reports) and every plot under
`figures/`.

## Scientific-integrity note

Every phase in this project reports its result as measured, including
the substantial VQC underperformance in Phase 2. No result was adjusted,
hidden, or re-run silently until it looked better; recovery experiments
(Phase 2B/2C) are documented as separate, explicit, pre-registered
attempts, and the remaining gap to classical ML is stated plainly in the
final comparison table rather than implied away.
