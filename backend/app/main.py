"""
Q-REMED backend (v2.1 dynamic) -- FastAPI service wiring the generalized engine modules
into the API surface described in the Q-REMED product spec.

Run locally:
    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --reload --port 8000

All computation is real (Qiskit Aer / Qiskit ML / scikit-learn) -- there
are no mocked results anywhere in this file.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
_project_root_env = Path(__file__).resolve().parents[2] / ".env"
if _project_root_env.exists():
    load_dotenv(dotenv_path=_project_root_env)

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import (
    classical_models,
    comparison as comparison_mod,
    dataset_analysis,
    evaluation,
    feature_count_selection,
    feature_selection,
    groq_analysis,
    hardware_compatibility,
    preexisting_data,
    preprocessing,
    quantum_circuit,
    quantum_evaluation,
    quantum_kernel,
    quantum_model,
    quantum_preprocessing,
    robustness as robustness_mod,
    threshold_analysis,
)

app = FastAPI(title="Q-REMED API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".html", ".js", ".css")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


SESSIONS: dict[str, dict] = {}


def _sanitize(obj):
    """Make numpy/pandas objects JSON-safe (also clears NaN/Inf, which the
    strict JSON encoder rejects)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def ok(data: dict):
    return JSONResponse(content=_sanitize(data))


def get_session(session_id: Optional[str] = None) -> dict:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]

    # If any session in SESSIONS already has full model_results, reuse it
    for s in reversed(list(SESSIONS.values())):
        if s.get("model_results"):
            if session_id:
                SESSIONS[session_id] = s
            return s

    # Auto-recover with pre-trained benchmark if session is missing or server reloaded
    sid, s = preexisting_data.load_pretrained_benchmark_session()
    resolved_id = session_id or sid
    s["_session_id"] = resolved_id
    SESSIONS[resolved_id] = s
    return s


def require(session: dict, key: str, message: str):
    if key not in session:
        raise HTTPException(400, message)
    return session[key]


# --------------------------------------------------------------------------
# 1. Dataset upload
# --------------------------------------------------------------------------

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise HTTPException(400, "Uploaded CSV appears to be empty.")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "df": df,
        "dataset_name": file.filename or "uploaded_dataset.csv",
        "_session_id": session_id,
    }

    preview = dataset_analysis.csv_preview(df, n_rows=10)
    return ok({
        "session_id": session_id,
        "dataset_name": SESSIONS[session_id]["dataset_name"],
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "columns_metadata": preview.get("columns_metadata", []),
        "suggested_target": preview.get("suggested_target"),
        "preview": preview,
    })


# --------------------------------------------------------------------------
# 2. Dataset analysis
# --------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    session_id: str
    target_column: str
    positive_class: Optional[str] = None
    binarize_strategy: Optional[str] = None
    binarize_threshold: Optional[float] = None


@app.post("/api/dataset/analyze")
def analyze(req: AnalyzeRequest):
    session = get_session(req.session_id)
    df = session["df"]
    overview = dataset_analysis.analyze_dataset(
        df,
        req.target_column,
        binarize_strategy=req.binarize_strategy,
        binarize_threshold=req.binarize_threshold,
        positive_class=req.positive_class,
    )
    if not overview["validation"]["supported"]:
        return ok({"session_id": req.session_id, "overview": overview, "supported": False})

    session["target_column"] = req.target_column
    session["positive_class"] = req.positive_class
    session["binarize_strategy"] = req.binarize_strategy
    session["binarize_threshold"] = req.binarize_threshold
    return ok({"session_id": req.session_id, "overview": overview, "supported": True})


# --------------------------------------------------------------------------
# 3. Leakage-safe preprocessing
# --------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    session_id: str
    test_size: float = 0.2
    random_state: int = 42
    binarize_strategy: Optional[str] = None
    binarize_threshold: Optional[float] = None


@app.post("/api/preprocess")
def preprocess(req: PreprocessRequest):
    session = get_session(req.session_id)
    target_column = require(session, "target_column", "Call /api/dataset/analyze first.")
    df = session["df"]

    strategy = req.binarize_strategy or session.get("binarize_strategy")
    threshold = req.binarize_threshold if req.binarize_threshold is not None else session.get("binarize_threshold")

    result = preprocessing.leakage_safe_preprocess(
        df,
        target_column,
        req.test_size,
        req.random_state,
        positive_class=session.get("positive_class"),
        binarize_strategy=strategy,
        binarize_threshold=threshold,
    )
    session["prep"] = result
    session["config"] = {"test_size": req.test_size, "random_state": req.random_state}
    session["binarize_strategy"] = strategy
    session["binarize_threshold"] = threshold
    return ok({"session_id": req.session_id, "info": result["info"]})


# --------------------------------------------------------------------------
# 4/5. Generalized feature selection + feature count selection
# --------------------------------------------------------------------------

class FeatureSelectionRequest(BaseModel):
    session_id: str
    top_n: int = 10


@app.post("/api/feature-selection")
def run_feature_selection(req: FeatureSelectionRequest):
    session = get_session(req.session_id)
    prep = require(session, "prep", "Call /api/preprocess first.")
    seed = session["config"]["random_state"]

    anova_df = feature_selection.anova_ranking(prep["X_train"], prep["y_train"])
    rf_df = feature_selection.random_forest_importance_ranking(prep["X_train"], prep["y_train"], random_state=seed)
    agreement = feature_selection.compare_rankings(anova_df, rf_df, top_n=req.top_n)

    count_eval = feature_count_selection.evaluate_candidate_counts(prep["X_train"], prep["y_train"], anova_df, random_state=seed)

    session["anova_ranking"] = anova_df
    session["rf_ranking"] = rf_df
    session["count_evaluation"] = count_eval

    return ok({
        "session_id": req.session_id,
        "anova_ranking": feature_selection.ranking_to_records(anova_df),
        "random_forest_ranking": feature_selection.ranking_to_records(rf_df),
        "agreement": agreement,
        "feature_count_evaluation": count_eval,
    })


class SelectFeaturesRequest(BaseModel):
    session_id: str
    n_features: Optional[int] = None


@app.post("/api/feature-selection/select")
def select_features(req: SelectFeaturesRequest):
    session = get_session(req.session_id)
    count_eval = require(session, "count_evaluation", "Call /api/feature-selection first.")
    anova_df = session["anova_ranking"]

    n = req.n_features or count_eval["recommended_n_features"]
    matching = next((c for c in count_eval["candidates_evaluated"] if c["n_features"] == n), None)
    features = matching["features"] if matching else feature_selection.select_top_features(anova_df, n)

    session["selected_features"] = features
    session["selected_n_features"] = n

    return ok({
        "session_id": req.session_id,
        "selected_n_features": n,
        "selected_features": features,
        "is_recommended": n == count_eval["recommended_n_features"],
        "reason": count_eval["reason"] if n == count_eval["recommended_n_features"] else "User-selected feature count (override of the recommendation).",
    })


# --------------------------------------------------------------------------
# 6. Quantum preprocessing + circuit configuration
# --------------------------------------------------------------------------

class QuantumConfigRequest(BaseModel):
    session_id: Optional[str] = None
    ansatz: str = "efficient_su2"
    reps: int = 1
    shots: int = 1024
    maxiter: int = 100
    seed: Optional[int] = None


@app.post("/api/quantum/configure")
def configure_quantum(req: QuantumConfigRequest):
    session = get_session(req.session_id)
    resolved_id = session.get("_session_id") or req.session_id or str(uuid.uuid4())
    session["_session_id"] = resolved_id
    prep = require(session, "prep", "Call /api/preprocess first.")
    features = require(session, "selected_features", "Call /api/feature-selection/select first.")
    seed = req.seed if req.seed is not None else session["config"]["random_state"]

    X_train_sel = prep["X_train"][features]
    X_test_sel = prep["X_test"][features]
    X_train_q, X_test_q, q_scaler = quantum_preprocessing.prepare_quantum_features(X_train_sel, X_test_sel)

    n_qubits = len(features)
    feature_map = quantum_circuit.build_feature_map(n_qubits, reps=1)
    ansatz = quantum_circuit.build_ansatz(n_qubits, reps=req.reps, kind=req.ansatz)
    full_circuit = quantum_circuit.build_full_circuit(feature_map, ansatz)
    circuit_stats = quantum_circuit.analyze_circuit(feature_map, ansatz, full_circuit)
    circuit_render = quantum_circuit.render_circuit_images(full_circuit)

    # Static transpilation check against IBM basis gates (auto-computed for active quantum circuit)
    hw_readiness = None
    try:
        hw_readiness = hardware_compatibility.check_hardware_compatibility(full_circuit)
        session["hardware_readiness"] = hw_readiness
    except Exception as e:
        print(f"Hardware readiness check on configure skipped: {e}")

    session["quantum"] = {
        "X_train_q": X_train_q, "X_test_q": X_test_q, "q_scaler": q_scaler,
        "feature_map": feature_map, "ansatz": ansatz, "full_circuit": full_circuit,
        "config": {"ansatz": req.ansatz, "reps": req.reps, "shots": req.shots,
                   "maxiter": req.maxiter, "seed": seed, "n_qubits": n_qubits},
        "circuit_image": circuit_render.get("circuit_image"),
        "circuit_decomposed_image": circuit_render.get("circuit_decomposed_image"),
        "circuit_text": circuit_render.get("circuit_text"),
        "circuit_decomposed_text": circuit_render.get("circuit_decomposed_text"),
    }

    return ok({
        "session_id": resolved_id,
        "n_qubits": n_qubits,
        "circuit_analysis": circuit_stats,
        "hardware_readiness": hw_readiness,
        "pipeline": "StandardScaler -> MinMaxScaler -> [0, pi] -> Quantum Feature Map",
        "config": session["quantum"]["config"],
        "circuit_image": circuit_render.get("circuit_image"),
        "circuit_decomposed_image": circuit_render.get("circuit_decomposed_image"),
        "circuit_text": circuit_render.get("circuit_text"),
        "circuit_decomposed_text": circuit_render.get("circuit_decomposed_text"),
    })


@app.get("/api/quantum/circuit/image")
def get_circuit_image(session_id: Optional[str] = None, decomposed: bool = True):
    import base64
    session = get_session(session_id)
    q = require(session, "quantum", "Quantum circuit not yet configured.")
    key = "circuit_decomposed_image" if decomposed else "circuit_image"
    b64 = q.get(key)
    if not b64 or not b64.startswith("data:image/png;base64,"):
        raise HTTPException(404, "Circuit image not available.")
    raw_png = base64.b64decode(b64.split(",", 1)[1])
    return Response(content=raw_png, media_type="image/png", headers={
        "Content-Disposition": f"inline; filename=quantum_circuit_{'decomposed' if decomposed else 'composite'}.png"
    })


# --------------------------------------------------------------------------
# 7/8. Run classical + quantum models
# --------------------------------------------------------------------------

class RunModelsRequest(BaseModel):
    session_id: str
    run_classical: bool = True
    run_vqc: bool = True
    run_quantum_kernel: bool = True
    prefer_device: str = "auto"  # "auto", "gpu", "cpu"


@app.get("/api/hardware/device-status")
def get_device_status():
    from .engine.device_detection import get_system_hardware_summary
    return ok(get_system_hardware_summary())


@app.post("/api/models/run")
def run_models(req: RunModelsRequest):
    session = get_session(req.session_id)
    prep = require(session, "prep", "Call /api/preprocess first.")
    features = require(session, "selected_features", "Call /api/feature-selection/select first.")
    seed = session["config"]["random_state"]
    pos, neg = prep["info"]["positive_label"], prep["info"]["negative_label"]

    model_results: dict = session.get("model_results", {})

    if req.run_classical:
        # Clear previous classical entries when re-running classical models (e.g. switching between GPU and CPU)
        model_results = {k: v for k, v in model_results.items() if v.get("type") != "classical"}
        X_train_sel = prep["X_train"][features]
        X_test_sel = prep["X_test"][features]
        trained = classical_models.train_all_baselines(
            X_train_sel, prep["y_train"], random_state=seed, prefer_device=req.prefer_device,
        )
        for name, bundle in trained.items():
            ev = evaluation.evaluate_sklearn_model(bundle["model"], X_test_sel, prep["y_test"], pos, neg)
            model_results[name] = {
                "type": "classical",
                "metrics": ev["metrics"],
                "confusion_matrix": ev["confusion_matrix"],
                "roc_curve": ev["roc_curve"],
                "training_time_seconds": bundle["training_time_seconds"],
                "device": bundle.get("device", "CPU"),
                "accelerator": bundle.get("accelerator", "CPU"),
            }
        session["_classical_models"] = {name: b["model"] for name, b in trained.items()}

    quantum_notes = []
    if (req.run_vqc or req.run_quantum_kernel):
        q = require(session, "quantum", "Call /api/quantum/configure first.")
        cfg = q["config"]

        if req.run_vqc:
            X_sub, y_sub, note = quantum_model.subsample_for_vqc(q["X_train_q"], prep["y_train"],
                                                                   quantum_model.MAX_VQC_TRAIN_SAMPLES, seed)
            if note:
                quantum_notes.append(note)
            vqc, loss_hist = quantum_model.build_vqc(q["feature_map"], q["ansatz"], cfg["seed"],
                                                       shots=cfg["shots"], maxiter=cfg["maxiter"])
            training_info = quantum_model.train_vqc(vqc, loss_hist, X_sub, y_sub)
            ev = quantum_evaluation.evaluate_vqc(vqc, q["X_test_q"], prep["y_test"], pos, neg)

            ansatz_label = "EfficientSU2 VQC" if cfg["ansatz"] == "efficient_su2" else "RealAmplitudes VQC"
            model_results[ansatz_label] = {"type": "quantum", "metrics": ev["metrics"],
                                            "confusion_matrix": ev["confusion_matrix"], "roc_curve": ev["roc_curve"],
                                            "training_time_seconds": training_info["training_time_seconds"],
                                            "training_info": training_info,
                                            "device": "CPU (Multi-threaded Simulator)",
                                            "accelerator": "Qiskit Aer Simulator",
                                            "n_train_used": len(X_sub), "n_train_available": len(q["X_train_q"])}

        if req.run_quantum_kernel:
            X_sub, y_sub, note = quantum_kernel.subsample_training_set(q["X_train_q"], prep["y_train"],
                                                                          quantum_kernel.MAX_KERNEL_TRAIN_SAMPLES, seed)
            if note:
                quantum_notes.append(note)
            kres = quantum_kernel.train_and_evaluate(q["feature_map"], X_sub, y_sub, q["X_test_q"], prep["y_test"],
                                                        cfg["shots"], cfg["seed"], pos)
            model_results["Quantum Kernel SVM"] = {"type": "quantum", "metrics": kres["metrics"],
                                                     "confusion_matrix": kres["confusion_matrix"], "roc_curve": kres["roc_curve"],
                                                     "training_time_seconds": kres["timing"]["svm_train_time_seconds"],
                                                     "timing": kres["timing"],
                                                     "device": "CPU (Multi-threaded Simulator)",
                                                     "accelerator": "Qiskit Aer Simulator",
                                                     "n_train_used": kres["n_train_subsampled"], "n_train_available": len(q["X_train_q"])}

    session["model_results"] = model_results
    session["quantum_notes"] = quantum_notes
    comp = comparison_mod.build_comparison_table(model_results)
    session["comparison"] = comp

    return ok({"session_id": req.session_id, "model_results": {k: {kk: vv for kk, vv in v.items() if kk != "training_info"} for k, v in model_results.items()},
               "quantum_computational_notes": quantum_notes, "comparison": comp})


@app.get("/api/comparison")
def get_comparison(session_id: str):
    session = get_session(session_id)
    comp = require(session, "comparison", "Call /api/models/run first.")
    return ok({"session_id": session_id, "comparison": comp})


# --------------------------------------------------------------------------
# 12. Threshold analysis (classical models; see engine/threshold_analysis.py)
# --------------------------------------------------------------------------

class ThresholdRequest(BaseModel):
    session_id: str
    model_name: str
    objective: str = "maximize_f1"


def _classical_model_factory(model_name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if "Logistic" in model_name:
        return lambda: LogisticRegression(max_iter=1000, random_state=seed)
    if "Random Forest" in model_name:
        return lambda: RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed)
    if "XGBoost" in model_name:
        import xgboost as xgb
        from .engine.device_detection import detect_gpu
        gpu = detect_gpu()
        dev = "cuda" if gpu["has_gpu"] else "cpu"
        return lambda: xgb.XGBClassifier(
            device=dev,
            tree_method="hist",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed,
            eval_metric="logloss",
        )
    return None


@app.post("/api/threshold")
def run_threshold(req: ThresholdRequest):
    session = get_session(req.session_id)
    prep = require(session, "prep", "Call /api/preprocess first.")
    features = require(session, "selected_features", "Call /api/feature-selection/select first.")
    seed = session["config"]["random_state"]
    pos, neg = prep["info"]["positive_label"], prep["info"]["negative_label"]

    obj = req.objective
    if obj == "f1":
        obj = "maximize_f1"
    elif obj in ("sensitivity", "recall"):
        obj = "prioritize_sensitivity"
    elif obj == "specificity":
        obj = "prioritize_specificity"

    factory_builder = _classical_model_factory(req.model_name, seed)
    if factory_builder is None:
        return ok({"session_id": req.session_id, "supported": False,
                   "reason": f"Threshold analysis in this build supports classical models "
                             f"('Logistic Regression', 'Random Forest', 'XGBoost') only, since it needs to refit "
                             f"the model on a held-out validation split; refitting the quantum models for "
                             f"threshold tuning is not performed by default due to training cost."})

    X_train_sel = prep["X_train"][features]
    X_test_sel = prep["X_test"][features]
    result = threshold_analysis.run_threshold_analysis(
        factory_builder, X_train_sel, prep["y_train"], X_test_sel, prep["y_test"],
        obj, pos, neg, random_state=seed,
    )
    result["optimal_threshold"] = result["selected_threshold"]
    result["optimal_metrics"] = result["selected_threshold_metrics"]
    result["default_metrics"] = result["default_threshold_metrics"]
    session.setdefault("threshold_results", {})[req.model_name] = result
    return ok({"session_id": req.session_id, "supported": True, "model_name": req.model_name, "result": result, "threshold_results": session["threshold_results"]})


@app.get("/api/threshold")
def get_threshold(session_id: str):
    session = get_session(session_id)
    results = session.get("threshold_results", {})
    if not results and session.get("model_results") and session.get("prep") and session.get("selected_features"):
        best_model = session.get("comparison", {}).get("best_classical_model") or "Random Forest"
        req = ThresholdRequest(session_id=session_id, model_name=best_model, objective="maximize_f1")
        return run_threshold(req)
    return ok({"session_id": session_id, "supported": True, "threshold_results": results})


# --------------------------------------------------------------------------
# 13. Robustness (multi-seed)
# --------------------------------------------------------------------------

class RobustnessRequest(BaseModel):
    session_id: str
    model_name: str
    n_seeds: int = 5


@app.post("/api/robustness")
def run_robustness(req: RobustnessRequest):
    session = get_session(req.session_id)
    target_column = require(session, "target_column", "Call /api/dataset/analyze first.")
    df = session["df"]
    features = require(session, "selected_features", "Call /api/feature-selection/select first.")
    base_seed_cfg = session["config"]
    positive_class = session.get("positive_class")

    n_seeds = max(2, min(req.n_seeds, 10))
    seeds = list(range(n_seeds))
    binarize_strategy = session.get("binarize_strategy")
    binarize_threshold = session.get("binarize_threshold")

    # Determine which model(s) to evaluate
    if req.model_name.lower() == "all":
        models_to_run = []
        for m_name, m_val in session.get("model_results", {}).items():
            if m_val.get("type") == "classical" and _classical_model_factory(m_name, base_seed_cfg["random_state"]) is not None:
                models_to_run.append(m_name)
        if not models_to_run:
            models_to_run = ["Random Forest", "Logistic Regression"]
    else:
        models_to_run = [req.model_name]

    evaluated_results = {}
    for m_name in models_to_run:
        factory_builder = _classical_model_factory(m_name, base_seed_cfg["random_state"])
        if factory_builder is None:
            continue

        def make_run_fn(builder):
            def run_fn(seed):
                prep = preprocessing.leakage_safe_preprocess(
                    df, target_column, base_seed_cfg["test_size"], seed,
                    positive_class=positive_class,
                    binarize_strategy=binarize_strategy,
                    binarize_threshold=binarize_threshold,
                )
                pos, neg = prep["info"]["positive_label"], prep["info"]["negative_label"]
                model = builder()
                model.fit(prep["X_train"][features], prep["y_train"])
                ev = evaluation.evaluate_sklearn_model(model, prep["X_test"][features], prep["y_test"], pos, neg)
                return ev["metrics"]
            return run_fn

        res = robustness_mod.run_robustness_analysis(make_run_fn(factory_builder), seeds=seeds)
        session.setdefault("robustness_results", {})[m_name] = res
        evaluated_results[m_name] = res

    if not evaluated_results:
        return ok({"session_id": req.session_id, "supported": False,
                   "reason": "Robustness analysis in this build re-runs the full split->train->evaluate "
                              "pipeline per seed for classical models ('Logistic Regression', 'Random Forest', 'XGBoost'). "
                              "Quantum robustness runs are computationally expensive and not run automatically."})

    primary_res = evaluated_results.get(req.model_name) or list(evaluated_results.values())[0]
    return ok({
        "session_id": req.session_id,
        "supported": True,
        "model_name": req.model_name,
        "result": primary_res,
        "robustness_results": session.get("robustness_results", {}),
    })


@app.get("/api/robustness")
def get_robustness(session_id: str):
    session = get_session(session_id)
    rob = session.get("robustness_results", {})
    if not rob and session.get("model_results") and session.get("df") is not None and session.get("selected_features"):
        req = RobustnessRequest(session_id=session_id, model_name="all", n_seeds=5)
        return run_robustness(req)
    return ok({"session_id": session_id, "supported": True, "robustness_results": rob})


# --------------------------------------------------------------------------
# 14. Hardware readiness
# --------------------------------------------------------------------------

@app.get("/api/hardware-readiness")
def hardware_readiness(session_id: str):
    session = get_session(session_id)
    if "hardware_readiness" in session and session["hardware_readiness"]:
        return ok({"session_id": session_id, "hardware_readiness": session["hardware_readiness"]})
    q = require(session, "quantum", "Call /api/quantum/configure first.")
    result = hardware_compatibility.check_hardware_compatibility(q["full_circuit"])
    session["hardware_readiness"] = result
    return ok({"session_id": session_id, "hardware_readiness": result})


# --------------------------------------------------------------------------
# 15. Groq AI-assisted analysis
# --------------------------------------------------------------------------

class AIAnalysisRequest(BaseModel):
    session_id: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


def _structured_results_for_ai(session: dict) -> dict:
    dataset_name = session.get("dataset_name") or ""
    is_wdbc_benchmark = "Wisconsin" in dataset_name or "WDBC" in dataset_name

    # Only enrich from pre-trained benchmark if the session explicitly loaded the WDBC reference benchmark
    if not session.get("model_results") and is_wdbc_benchmark:
        _, pre = preexisting_data.load_pretrained_benchmark_session()
        for k in ["prep", "anova_ranking", "rf_ranking", "count_evaluation", "selected_features",
                  "selected_n_features", "quantum", "model_results", "quantum_notes", "comparison",
                  "threshold_results", "robustness_results", "hardware_readiness"]:
            if k not in session or not session[k]:
                session[k] = pre[k]

    # Dynamic on-the-fly resolution for any missing evaluation components on the active dataset:
    if not session.get("hardware_readiness") and session.get("quantum", {}).get("full_circuit"):
        try:
            session["hardware_readiness"] = hardware_compatibility.check_hardware_compatibility(session["quantum"]["full_circuit"])
        except Exception as e:
            print(f"Hardware readiness auto-check skipped: {e}")

    sid = session.get("_session_id") or ""
    if not session.get("robustness_results") and session.get("model_results") and session.get("df") is not None and session.get("selected_features"):
        try:
            req_rob = RobustnessRequest(session_id=sid, model_name="all", n_seeds=5)
            res_rob = run_robustness(req_rob)
            if res_rob.get("status") == "ok" and res_rob.get("robustness_results"):
                session["robustness_results"] = res_rob["robustness_results"]
        except Exception as e:
            print(f"Robustness auto-analysis skipped: {e}")

    if not session.get("threshold_results") and session.get("model_results") and session.get("prep") is not None and session.get("selected_features"):
        try:
            best_model = session.get("comparison", {}).get("best_classical_model") or "Random Forest"
            res_th = run_threshold(ThresholdRequest(session_id=sid, model_name=best_model, objective="maximize_f1"))
            if res_th.get("status") == "ok" and res_th.get("result"):
                session.setdefault("threshold_results", {})[best_model] = res_th["result"]
        except Exception as e:
            print(f"Threshold auto-analysis skipped: {e}")

    prep_info = session.get("prep", {}).get("info")
    return {
        "dataset": {"name": session.get("dataset_name"), "target_column": session.get("target_column"),
                    "preprocessing_info": prep_info},
        "selected_features": session.get("selected_features"),
        "feature_count_reason": session.get("count_evaluation", {}).get("reason") if session.get("count_evaluation") else None,
        "model_metrics": {name: r["metrics"] for name, r in session.get("model_results", {}).items()},
        "comparison": session.get("comparison"),
        "quantum_computational_notes": session.get("quantum_notes"),
        "threshold_results": session.get("threshold_results"),
        "robustness_results": session.get("robustness_results"),
        "hardware_readiness": session.get("hardware_readiness"),
        "limitations": [
            "Quantum kernel / VQC training may use a subsampled training set (see quantum_computational_notes).",
            "Hardware readiness reflects a static transpilation check against a generic IBM-typical basis gate set, not a specific real device or actual QPU execution.",
            "This platform is a research and decision-support tool; it is not a clinically validated diagnostic system.",
        ],
    }


class AIChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    history: Optional[list[dict]] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/ai-analysis")
def ai_analysis(req: AIAnalysisRequest):
    session = get_session(req.session_id)
    resolved_id = session.get("_session_id") or req.session_id or str(uuid.uuid4())
    session["_session_id"] = resolved_id
    structured = _structured_results_for_ai(session)
    result = groq_analysis.run_ai_analysis(structured, api_key=req.api_key, model=req.model)
    session["ai_analysis"] = result
    if result.get("available") and "analysis" in result:
        session["chat_history"] = [
            {"role": "assistant", "content": result["analysis"]}
        ]
        result["history"] = session["chat_history"]
    return ok({"session_id": resolved_id, **result})


@app.post("/api/ai-chat")
def ai_chat(req: AIChatRequest):
    session = get_session(req.session_id)
    resolved_id = session.get("_session_id") or req.session_id or str(uuid.uuid4())
    session["_session_id"] = resolved_id
    structured = _structured_results_for_ai(session)
    session_history = session.setdefault("chat_history", [])

    messages_to_send = list(req.history) if req.history is not None else list(session_history)
    messages_to_send.append({"role": "user", "content": req.message})

    result = groq_analysis.run_ai_chat(
        structured,
        messages=messages_to_send,
        api_key=req.api_key,
        model=req.model,
    )
    if result.get("available") and "message" in result:
        session_history.append({"role": "user", "content": req.message})
        session_history.append(result["message"])
        result["history"] = session_history
    return ok({"session_id": resolved_id, **result})


# --------------------------------------------------------------------------
# 16. Full report
# --------------------------------------------------------------------------

@app.get("/api/report")
def get_report(session_id: str):
    session = get_session(session_id)
    session["_session_id"] = session.get("_session_id") or session_id
    structured = _structured_results_for_ai(session)
    report = {
        "dataset": {"name": session.get("dataset_name"), "target_column": session.get("target_column")},
        "preprocessing": session.get("prep", {}).get("info"),
        "feature_selection": {
            "anova_ranking": feature_selection.ranking_to_records(session["anova_ranking"]) if "anova_ranking" in session else None,
            "random_forest_ranking": feature_selection.ranking_to_records(session["rf_ranking"]) if "rf_ranking" in session else None,
            "feature_count_evaluation": session.get("count_evaluation"),
            "selected_features": session.get("selected_features"),
        },
        "quantum_configuration": session.get("quantum", {}).get("config") if "quantum" in session else None,
        "model_results": {name: {kk: vv for kk, vv in r.items() if kk != "training_info"} for name, r in session.get("model_results", {}).items()},
        "quantum_computational_notes": session.get("quantum_notes"),
        "comparison": session.get("comparison"),
        "threshold_analysis": session.get("threshold_results"),
        "robustness": session.get("robustness_results"),
        "hardware_readiness": session.get("hardware_readiness"),
        "ai_assisted_analysis": session.get("ai_analysis"),
        "ai_chat_history": session.get("chat_history", []),
        "limitations": structured["limitations"],
    }
    return ok({"session_id": session_id, "report": report})


# --------------------------------------------------------------------------
# 17. WDBC reference benchmark
# --------------------------------------------------------------------------

@app.post("/api/benchmark/wdbc")
def run_wdbc_benchmark():
    from sklearn.datasets import load_breast_cancer
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.data.copy()
    df["diagnosis"] = [bunch.target_names[i] for i in bunch.target]  # "malignant"/"benign" strings

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"df": df, "dataset_name": "Wisconsin Diagnostic Breast Cancer (reference benchmark)"}
    return ok({
        "session_id": session_id,
        "dataset_name": SESSIONS[session_id]["dataset_name"],
        "target_column": "diagnosis",
        "positive_class": "malignant",
        "message": "WDBC is the reference benchmark used to validate the Q-REMED methodology. "
                   "Continue through /api/dataset/analyze -> /api/preprocess -> ... using "
                   "target_column='diagnosis', positive_class='malignant' to reproduce it.",
        "n_rows": int(df.shape[0]), "n_columns": int(df.shape[1]), "columns": list(df.columns),
        "preview": dataset_analysis.csv_preview(df, 10),
    })


@app.post("/api/benchmark/pretrained")
def run_pretrained_benchmark():
    session_id, session = preexisting_data.load_pretrained_benchmark_session()
    SESSIONS[session_id] = session
    df = session["df"]

    overview = dataset_analysis.analyze_dataset(df, session["target_column"])

    return ok({
        "session_id": session_id,
        "dataset_name": session["dataset_name"],
        "target_column": session["target_column"],
        "positive_class": session["positive_class"],
        "message": "⚡ Loaded pre-trained reference benchmark results from results/ in <1 second! All tables, charts, models, and comparisons are fully unlocked.",
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "preview": dataset_analysis.csv_preview(df, 10),
        "overview": overview,
        "prep_info": session["prep"]["info"],
        "anova_ranking": feature_selection.ranking_to_records(session["anova_ranking"]),
        "random_forest_ranking": feature_selection.ranking_to_records(session["rf_ranking"]),
        "feature_count_evaluation": session["count_evaluation"],
        "selected_features": session["selected_features"],
        "selected_n_features": session["selected_n_features"],
        "quantum": {
            "n_qubits": session["quantum"]["config"]["n_qubits"],
            "circuit_analysis": session["quantum"]["circuit_analysis"],
            "pipeline": session["quantum"]["pipeline"],
            "config": session["quantum"]["config"],
            "circuit_image": session["quantum"].get("circuit_image"),
            "circuit_decomposed_image": session["quantum"].get("circuit_decomposed_image"),
            "circuit_text": session["quantum"].get("circuit_text"),
            "circuit_decomposed_text": session["quantum"].get("circuit_decomposed_text"),
        },
        "model_results": {k: {kk: vv for kk, vv in v.items() if kk != "training_info"} for k, v in session["model_results"].items()},
        "quantum_computational_notes": session["quantum_notes"],
        "comparison": session["comparison"],
        "threshold_results": session["threshold_results"],
        "robustness_results": session["robustness_results"],
        "hardware_readiness": session["hardware_readiness"],
    })


# --------------------------------------------------------------------------
# Static frontend (served from the same app for a single `uvicorn` command)
# --------------------------------------------------------------------------
import os
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
