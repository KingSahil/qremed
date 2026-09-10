# Q-REMED — Quantum-Enhanced ML Research & Evaluation Dashboard

Q-REMED is a full-stack research platform that benchmarks **classical machine learning against quantum machine learning** (variational circuits and quantum kernels) on any uploaded binary-classification dataset, with first-class support for the Breast Cancer Wisconsin (Diagnostic) benchmark.

The project is built in two layers:

- **Research pipeline** (`src/`) — four-phase experimental study on WDBC, focused on leakage-safe methodology, honest reporting of quantum underperformance, and a documented recovery process rather than cherry-picked final numbers.
- **Interactive web platform** (`backend/` + `frontend/`) — a FastAPI + vanilla-JS dashboard that lets you upload *any* tabular dataset, run the full Q-REMED pipeline, and explore results across 12 guided steps — dynamically, without touching code.

---

## TL;DR results (WDBC benchmark)

| Model | Accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (classical) | 0.930 | 0.952 | 0.917 | 0.909 | 0.991 |
| Random Forest (classical) | 0.930 | 0.929 | 0.931 | 0.907 | 0.986 |
| VQC, ZZFeatureMap+RealAmplitudes (Phase 2, original) | 0.596 | 0.333 | 0.750 | 0.378 | 0.639 |
| VQC, ZZFeatureMap+EfficientSU2 (Phase 2B, recovered) | 0.807 | 0.571 | 0.944 | 0.686 | 0.886 |
| VQC, EfficientSU2 + validation-tuned threshold (Phase 2C) | 0.807 | 0.881 | 0.764 | 0.771 | 0.877 |
| Quantum Kernel SVM (Phase 2C, 80/455 training subsample) | 0.877 | 0.690 | 0.986 | 0.806 | 0.924 |

**Headline finding:** the first VQC badly underperformed both classical baselines. Rather than hide that, the project ran a pre-registered failure analysis (ruling out data, preprocessing, and measurement bugs) and a pre-registered recovery protocol (ruling out shot noise and optimizer budget; identifying the ansatz as the fix). The gap to classical ML narrows substantially but is not closed — that residual gap is reported as a real limitation, not smoothed over.

---

## Repository layout

```
q_remed_phase1_2_2b_2c/
├── README.md
├── Q_REMED_Results_Walkthrough.ipynb   ← notebook reproducing key results/figures
├── requirements.txt
├── .env.example                         ← copy to .env and set GROQ_API_KEY
│
├── src/                                 ← batch research scripts (WDBC pipeline)
│   ├── run_phase1.py                    ← classical baseline + feature selection
│   ├── run_phase2.py                    ← first VQC pipeline
│   ├── finalize_phase2.py               ← failure analysis + honest write-up
│   ├── finalize_phase2b.py              ← trainability recovery decision
│   ├── finalize_phase2c.py              ← robustness checks + quantum kernel SVM
│   ├── config.py                        ← all experiment constants
│   ├── data_loader.py                   ← dataset loading + verification
│   ├── preprocessing.py                 ← leakage-safe split + scaling
│   ├── feature_selection.py             ← ANOVA (primary) + RF (cross-check)
│   ├── decision.py                      ← Phase 1 feature-set selection rule
│   ├── classical_models.py              ← Logistic Regression / Random Forest
│   ├── evaluation.py                    ← shared metrics (classical + quantum)
│   ├── quantum_preprocessing.py         ← [0, π] rescaling for angle encoding
│   ├── quantum_circuit.py               ← ZZFeatureMap / RealAmplitudes / EfficientSU2
│   ├── quantum_model.py                 ← VQC construction + training loop
│   ├── quantum_kernel.py                ← fidelity quantum kernel for SVM
│   ├── quantum_evaluation.py            ← VQC test-set evaluation
│   ├── exact_training.py                ← statevector (noise-free) VQC variant
│   ├── failure_analysis.py              ← Phase 2 failure-analysis checks
│   ├── hardware_compatibility.py        ← static transpile check (IBM basis gates)
│   └── plotting.py / quantum_plotting.py
│
├── backend/
│   └── app/
│       ├── main.py                      ← FastAPI app + all REST endpoints
│       └── engine/
│           ├── dataset_analysis.py      ← multi-type column analysis + validation
│           ├── preprocessing.py         ← leakage-safe pipeline (binary/continuous/multi)
│           ├── feature_selection.py     ← ANOVA ranking
│           ├── feature_count_selection.py ← feature-count auto-selection
│           ├── classical_models.py      ← RF / LR / XGBoost training
│           ├── evaluation.py            ← shared metrics
│           ├── comparison.py            ← model comparison table
│           ├── quantum_circuit.py       ← dynamic qubit-count VQC circuits
│           ├── quantum_model.py         ← VQC training loop
│           ├── quantum_kernel.py        ← quantum kernel SVM
│           ├── quantum_preprocessing.py ← angle-encoding rescaler
│           ├── quantum_evaluation.py    ← VQC evaluation
│           ├── hardware_compatibility.py ← transpile + hardware readiness report
│           ├── robustness.py            ← multi-seed robustness evaluation
│           ├── threshold_analysis.py    ← decision-threshold sweep
│           ├── groq_analysis.py         ← Groq LLM AI analysis + chat
│           ├── device_detection.py      ← GPU/CPU device detection
│           └── preexisting_data.py      ← WDBC pre-trained benchmark loader
│
├── frontend/
│   ├── index.html                       ← single-page app (12-step wizard)
│   ├── app.js                           ← full client-side logic (~1800 lines)
│   └── style.css                        ← dark-mode design system
│
├── results/                             ← JSON/CSV artifacts from batch pipeline
└── figures/                             ← PNG plots from batch pipeline
```

---

## Web Platform — Interactive Dashboard

The web platform supports **any tabular binary-classification CSV** (binary, continuous numeric, and multiclass targets via binarization are all supported). The WDBC benchmark is available as a one-click pre-trained reference.

### Quick start

```bash
# From the project root:
uv run uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Then open **http://localhost:8000/** in your browser.

### 12-step guided workflow

| Step | Section | What it does |
|---|---|---|
| 1 | **Upload** | Upload any CSV or click "Run WDBC Benchmark" for the pre-trained reference |
| 2 | **Analyze** | Inspects columns, detects target type (binary/continuous/multiclass), auto-suggests target column |
| 3 | **Preprocess** | Leakage-safe stratified train/test split + StandardScaler fit on train fold only |
| 4 | **Feature selection** | ANOVA F-score ranking on training fold; barchart of top features |
| 5 | **Feature count** | Selects optimal feature count (auto or manual) |
| 6 | **Quantum config** | Sets qubit count, feature map, ansatz, optimizer, reps — transpiles circuit live |
| 7 | **Train models** | Trains Random Forest, Logistic Regression, XGBoost, and VQC in one click |
| 8 | **Compare** | Side-by-side comparison table: accuracy, sensitivity, specificity, F1, ROC-AUC |
| 9 | **Threshold** | Decision-threshold sweep for any trained model; pick operating point |
| 10 | **Robustness** | Multi-seed (5×) robustness check for any classical model |
| 11 | **Hardware** | Hardware-readiness report: transpiled depth, gate counts, IBM basis-gate compatibility |
| 12 | **Report** | Full session summary + optional Groq AI narrative analysis + Markdown export |

### Supported dataset types

| Target type | How it's handled |
|---|---|
| Binary (0/1 or two string classes) | Native |
| Continuous numeric | Binarized via median, mean, or custom threshold |
| Multiclass | Binarized via One-vs-Rest (pick the positive class) |
| Categorical features | Leakage-safe label encoding on train fold only |

### Optional: Groq AI analysis

Copy `.env.example` to `.env` and set your key:

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile   # optional override
```

Or paste the key directly in the Step 12 UI. The AI uses the actual session results (features, metrics, model comparisons) — not hardcoded data.

---

## REST API

The backend exposes a JSON API (Swagger UI at `/docs`):

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/dataset/upload` | Upload CSV → returns session_id, columns, suggested target |
| `POST` | `/api/dataset/analyze` | Validate target column, inspect class balance |
| `POST` | `/api/preprocess` | Leakage-safe split + scaling |
| `POST` | `/api/feature-selection` | ANOVA ranking |
| `POST` | `/api/feature-selection/select` | Confirm feature count |
| `POST` | `/api/quantum/configure` | Configure + transpile quantum circuit |
| `POST` | `/api/models/run` | Train RF / LR / XGBoost / VQC |
| `GET`  | `/api/comparison` | Model comparison table |
| `POST` | `/api/threshold` | Threshold sweep for a model |
| `GET`  | `/api/threshold` | Retrieve stored threshold results |
| `POST` | `/api/robustness` | Multi-seed robustness evaluation |
| `GET`  | `/api/robustness` | Retrieve stored robustness results |
| `GET`  | `/api/hardware-readiness` | Hardware compatibility report |
| `GET`  | `/api/report` | Full session report |
| `POST` | `/api/ai-analysis` | Groq LLM narrative analysis |
| `POST` | `/api/ai-chat` | Groq LLM chat (session-aware) |
| `POST` | `/api/benchmark/pretrained` | Load pre-trained WDBC benchmark session |
| `GET`  | `/api/hardware/device-status` | GPU/CPU status |
| `GET`  | `/api/health` | Health check |

---

## Research Pipeline — The four phases

### Phase 1 — Classical baseline (`src/run_phase1.py`)
- Loads the Breast Cancer Wisconsin dataset (569 samples, 30 features) via scikit-learn and verifies label encoding from the data itself.
- Leakage-safe pipeline: stratified 80/20 split **first**, then `StandardScaler` fit on training fold only.
- Feature ranking on training fold only: ANOVA F-score (primary), Random Forest importance (cross-check only — never drives selection).
- Trains Logistic Regression and Random Forest on 4-feature and 6-feature subsets; picks primary set using a pre-registered rule. Result: 4-feature set wins (`worst concave points`, `worst perimeter`, `mean concave points`, `worst radius`).

### Phase 2 — First variational quantum classifier (`src/run_phase2.py`, `src/finalize_phase2.py`)
- Reuses Phase 1's exact train/test split and 4-feature set.
- Circuit: 4-qubit `ZZFeatureMap` (reps=1) + `RealAmplitudes` ansatz (reps=1), COBYLA optimizer on ideal Aer simulator.
- Result: VQC substantially underperforms both classical baselines. Pre-registered failure analysis (gap > 0.10 accuracy) rules out preprocessing/label/scaling/circuit bugs — points to a genuine trainability limitation.

### Phase 2B — Diagnosing and recovering trainability (`src/finalize_phase2b.py`)
Three pre-registered candidate fixes:
1. **Exact/statevector training** — barely moved accuracy → shot noise was not the cause.
2. **Extended optimizer budget** — barely moved accuracy → budget was not the cause.
3. **Alternative ansatz** (`EfficientSU2`) — accuracy rose 0.596 → 0.807, F1 0.378 → 0.686, ROC-AUC 0.639 → 0.886 → **this was the fix**.

### Phase 2C — Robustness and quantum kernel (`src/finalize_phase2c.py`)
- **Seed robustness:** recovered VQC retrained across 5 seeds; accuracy ranged 0.72–0.86 (std ≈ 0.05).
- **Validation-selected threshold:** choosing threshold on internal validation split to maximize F1 raised sensitivity 0.571 → 0.881 at a real, reported specificity cost (0.944 → 0.764).
- **Quantum kernel SVM:** fidelity-based kernel on 80-sample subsample reached 0.877 accuracy / 0.924 ROC-AUC — strongest quantum result, though trained on less data than classical models.

---

## Setup & Installation

### 1. Create a virtual environment

```bash
# Using uv (recommended)
uv venv

# Activate — Windows PowerShell:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

> **Note:** Classical-only use (Phase 1 or web platform without VQC) needs only `scikit-learn`, `pandas`, `numpy`, `fastapi`, `uvicorn`, and `xgboost`. The quantum phases require `qiskit`, `qiskit-aer`, and `qiskit-machine-learning`.

### 3. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY for AI analysis
```

---

## Running the batch research pipeline

```bash
cd src

python run_phase1.py          # Classical baseline + ANOVA feature selection
python run_phase2.py          # First VQC + Failure Analysis
python finalize_phase2.py     # Phase 2 failure analysis write-up
python finalize_phase2b.py    # Phase 2B recovery decision
python finalize_phase2c.py    # Phase 2C robustness + kernel
```

Or from the project root with `uv`:

```bash
uv run src/run_phase1.py
uv run src/run_phase2.py
uv run src/finalize_phase2.py
uv run src/finalize_phase2b.py
uv run src/finalize_phase2c.py
```

Results (JSON/CSV) are written to `results/` and plots to `figures/`.

---

## Reading the results

`Q_REMED_Results_Walkthrough.ipynb` walks through the pipeline stage-by-stage using pre-saved artifacts in `results/` and `figures/` — it does not require a live Qiskit training run to execute — and ends with the final comparison table and scientific interpretation.

---

## Scientific integrity note

Every phase reports its result as measured, including the substantial VQC underperformance in Phase 2. No result was adjusted, hidden, or re-run silently until it looked better. Recovery experiments (Phase 2B/2C) are documented as separate, explicit, pre-registered attempts, and the remaining gap to classical ML is stated plainly in the final comparison table rather than implied away.
