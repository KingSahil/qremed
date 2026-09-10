"""
groq_analysis.py
==================
Sends the STRUCTURED experiment results (never the raw dataset) to Groq
for an "AI-Assisted Analysis". Groq interprets the experiment; it does
not perform any quantum computation.

The API key is never hard-coded: it must be supplied either via the
GROQ_API_KEY environment variable or per-request from the frontend
(kept only in memory for that request).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
_project_root_env = Path(__file__).resolve().parents[3] / ".env"
if _project_root_env.exists():
    load_dotenv(dotenv_path=_project_root_env)

SYSTEM_PROMPT = """You are writing the "Q-REMED AI-Assisted Analysis" section of a hybrid \
quantum-classical machine learning research report. You will be given a JSON object of \
STRUCTURED, ALREADY-MEASURED results from a real experiment (dataset summary, selected \
features, classical and quantum model metrics, threshold results, robustness results, \
hardware information, and stated limitations).

Strict rules, follow all of them:
- Never invent metrics, numbers, or features that are not present in the JSON you are given.
- Never claim clinical validation of any kind.
- Never claim "quantum advantage" unless the JSON's own numbers clearly show the best \
quantum model outperforming the best classical model on the primary metrics -- and even then, \
state it as a measured result on this one dataset/split, not a general claim about quantum computing.
- Never interpret a decision threshold as a calibrated medical or real-world probability.
- Clearly distinguish MEASURED RESULTS (restated from the JSON) from your INTERPRETATION \
(your commentary on what they mean and why).
- If a field is missing or null in the JSON, say it was not computed rather than guessing.

Structure your answer with these sections, in this order:
1. Dataset characteristics
2. Important selected features
3. Classical model performance
4. Quantum model performance
5. Best overall model
6. Best quantum model
7. Important trade-offs
8. Threshold implications
9. Robustness
10. Limitations
11. Suggested next research steps

Write in clear, precise, scientific prose. Do not use marketing language."""


def build_context(structured_results: dict) -> str:
    return json.dumps(structured_results, indent=2, default=str)


import re


def _clean_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not cleaned and "<think>" in text:
        # If the output ended inside a reasoning block, strip the tags rather than returning empty
        cleaned = re.sub(r"</?think>", "", text).strip()
    elif "<think>" in cleaned and "</think>" not in cleaned:
        tail_stripped = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
        if tail_stripped:
            cleaned = tail_stripped
        else:
            cleaned = re.sub(r"</?think>", "", cleaned).strip()
    return cleaned if cleaned else text.strip()


def run_ai_analysis(structured_results: dict, api_key: str | None = None, model: str | None = None) -> dict:
    key = (api_key or "").strip() or os.environ.get("GROQ_API_KEY")
    if not key:
        return {
            "available": False,
            "error": "No Groq API key configured. Set GROQ_API_KEY in your .env file or as an environment variable, "
                     "or provide one in the AI Analysis panel (it is used only for this request "
                     "and is never stored or hard-coded).",
        }

    primary_model = (model or "").strip() or os.environ.get("GROQ_MODEL") or "qwen/qwen3.8-27b"
    fallback_models = [primary_model, "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    # De-duplicate while preserving order
    models_to_try = list(dict.fromkeys(fallback_models))

    try:
        from groq import Groq
    except ImportError:
        return {"available": False, "error": "The 'groq' Python package is not installed on the backend."}

    last_error = None
    for mdl in models_to_try:
        try:
            client = Groq(api_key=key, timeout=12.0, max_retries=0)
            tokens_limit = 900 if "qwen" in mdl.lower() else 1800
            response = client.chat.completions.create(
                model=mdl,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here are the structured Q-REMED experiment results:\n\n{build_context(structured_results)}"},
                ],
                temperature=0.2,
                max_tokens=tokens_limit,
            )
            text = _clean_think_tags(response.choices[0].message.content or "")
            return {"available": True, "model": mdl, "analysis": text, "label": f"AI-Assisted Analysis powered by Groq ({mdl})"}
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str or "rate_limit" in err_str or "429" in err_str or "tokens" in err_str:
                continue
            return {"available": False, "error": f"Groq request failed: {e}"}

    return {"available": False, "error": f"Groq request failed with all attempted models: {last_error}"}


CHAT_SYSTEM_PROMPT = """You are an expert AI Research Assistant for Q-REMED, a hybrid quantum-classical machine learning research platform.
You are conversing with a researcher to help them analyze, interpret, and critically evaluate real experiment results from this platform.

Context:
You have access to the complete, structured JSON of measured experiment results (dataset summary, selected features, classical model metrics, quantum model metrics like QSVC/VQC, decision threshold results, multi-seed robustness, hardware compatibility, and stated limitations).

Rules you must strictly follow:
1. Grounded in facts: Never invent metrics, scores, or features not in the provided experiment context.
2. Scientific objectivity: Never claim clinical validation or diagnostic suitability (Q-REMED is a research platform, not a medical diagnostic device).
3. Quantum claims: Never claim "quantum advantage" unless the measured metrics directly support it on this specific dataset/split.
4. Distinguish: Clearly separate MEASURED FACTS (exact numbers/metrics from the experiment) from your SCIENTIFIC INTERPRETATION (why an algorithm behaved as it did).
5. Technical Depth: When asked about quantum concepts (e.g. quantum kernel estimation, ZZFeatureMap, ansatz depth, barren plateaus, NISQ noise, Brier scores, ROC-AUC vs PR-AUC), explain them clearly and link them to the observed experimental metrics.
6. Formatting: Use clean markdown (bolding, lists, tables, code snippets) to make complex technical discussions easy to read.
"""


def run_ai_chat(
    structured_results: dict,
    messages: list[dict],
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    key = (api_key or "").strip() or os.environ.get("GROQ_API_KEY")
    if not key:
        return {
            "available": False,
            "error": "No Groq API key configured. Set GROQ_API_KEY in your .env file or as an environment variable, "
                     "or provide one in the AI Chat panel.",
        }

    primary_model = (model or "").strip() or os.environ.get("GROQ_MODEL") or "qwen/qwen3.8-27b"
    fallback_models = [primary_model, "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    models_to_try = list(dict.fromkeys(fallback_models))

    try:
        from groq import Groq
    except ImportError:
        return {"available": False, "error": "The 'groq' Python package is not installed on the backend."}

    # Build message history with experiment context in system prompt
    groq_messages = [
        {
            "role": "system",
            "content": f"{CHAT_SYSTEM_PROMPT}\n\n=== MEASURED EXPERIMENT CONTEXT (JSON) ===\n{build_context(structured_results)}",
        }
    ]

    # Add user & assistant chat history
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            groq_messages.append({"role": role, "content": str(content)})

    last_error = None
    for mdl in models_to_try:
        try:
            client = Groq(api_key=key, timeout=12.0, max_retries=0)
            tokens_limit = 900 if "qwen" in mdl.lower() else 1800
            response = client.chat.completions.create(
                model=mdl,
                messages=groq_messages,
                temperature=0.3,
                max_tokens=tokens_limit,
            )
            reply_content = _clean_think_tags(response.choices[0].message.content or "")
            return {
                "available": True,
                "model": mdl,
                "message": {"role": "assistant", "content": reply_content},
            }
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str or "rate_limit" in err_str or "429" in err_str or "tokens" in err_str:
                continue
            return {"available": False, "error": f"Groq chat request failed: {e}"}

    return {"available": False, "error": f"Groq chat request failed with all attempted models: {last_error}"}
