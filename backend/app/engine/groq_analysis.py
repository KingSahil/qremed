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


def run_ai_analysis(structured_results: dict, api_key: str | None = None, model: str = "llama-3.3-70b-versatile") -> dict:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return {
            "available": False,
            "error": "No Groq API key configured. Set GROQ_API_KEY as an environment variable, "
                     "or provide one in the AI Analysis panel (it is used only for this request "
                     "and is never stored or hard-coded).",
        }

    try:
        from groq import Groq
    except ImportError:
        return {"available": False, "error": "The 'groq' Python package is not installed on the backend."}

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here are the structured Q-REMED experiment results:\n\n{build_context(structured_results)}"},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        return {"available": True, "model": model, "analysis": text, "label": "AI-Assisted Analysis powered by Groq"}
    except Exception as e:
        return {"available": False, "error": f"Groq request failed: {e}"}
