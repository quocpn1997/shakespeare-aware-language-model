"""
Automated rubric scorer for results/evaluation_results.csv.

Reads the CSV produced by evaluate.py and calls the local gemma3:4b model
(via Ollama) to assign 1-5 scores on each of the five assignment criteria.
Scores are written back into the same CSV file.

Scoring criteria (from the assignment spec):
  correctness          — factually accurate w.r.t. the plays? (1–5)
  grounding            — answer supported by retrieved passages? (1–5)
  retrieval_relevance  — retrieved passages relevant to the question? (1–5)
  usefulness           — helps a beginner understand? (1–5)
  style_quality        — Shakespearean tone without losing clarity? (1–5)

N/A rules:
  grounding + retrieval_relevance → N/A for baseline rows (no retrieval).
  style_quality                   → N/A for non-stylised questions.
  all scores except rejected      → N/A for off_topic rows.

For off_topic rows the scorer only checks whether the system correctly
rejected the query instead of answering it (rejected = True/False).

Usage:
  python src/score.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, OLLAMA_MODEL

CSV_PATH    = RESULTS_DIR / "evaluation_results.csv"
BACKUP_PATH = RESULTS_DIR / "evaluation_results_unscored.csv"

# The exact prefix the RAG pipeline returns when it rejects a query.
REJECTION_PREFIX = "I could not find relevant information"

SCORE_FIELDS = [
    "correctness_score",
    "grounding_score",
    "retrieval_relevance_score",
    "usefulness_score",
    "style_quality_score",
]

# ── Rubric prompt sent as the system message ──────────────────────────────────

RUBRIC_SYSTEM = """You are an academic evaluator assessing a Shakespeare question-answering system.

Score the generated answer on the criteria listed below. Use integers 1–5:
  5 = Excellent  — fully meets the criterion, no notable weakness
  4 = Good       — mostly meets the criterion, minor gaps only
  3 = Adequate   — partially meets the criterion, noticeable gaps
  2 = Weak       — criterion attempted but largely unmet
  1 = Poor       — criterion not met or answer is wrong / empty

Criteria:
  correctness         — Is the answer factually accurate about the play?
  grounding           — Is every claim explicitly supported by the retrieved passages shown?
  retrieval_relevance — Are the retrieved passages topically relevant to the question?
  usefulness          — Would this answer help a beginner with no Shakespeare background?
  style_quality       — (stylised questions only) Does the response reflect Shakespearean
                        tone without becoming incomprehensible?

Return ONLY a valid JSON object with exactly these keys — no extra text:
{
  "correctness": <1-5>,
  "grounding": <1-5 or "N/A">,
  "retrieval_relevance": <1-5 or "N/A">,
  "usefulness": <1-5>,
  "style_quality": <1-5 or "N/A">,
  "justification": "<one sentence: main strength or weakness>"
}"""


def _is_rejected(response: str) -> bool:
    return response.strip().startswith(REJECTION_PREFIX)


def _na_scores(justification: str = "") -> Dict[str, str]:
    """Return all-N/A score dict for off_topic rows."""
    return {
        "correctness_score":         "N/A",
        "grounding_score":           "N/A",
        "retrieval_relevance_score": "N/A",
        "usefulness_score":          "N/A",
        "style_quality_score":       "N/A",
        "justification":             justification,
    }


def _build_user_prompt(row: Dict[str, str]) -> str:
    is_baseline = row["system"] == "baseline"
    passages = (
        row.get("retrieved_passages", "").strip()
        or "(none — baseline system, no retrieval)"
    )
    style_note = (
        "  style_quality: score 1–5 (stylised question)\n"
        if row.get("question_type") == "stylised_generation"
        else '  style_quality: "N/A" (not a stylised question)\n'
    )
    grounding_note = (
        '  grounding: "N/A"\n  retrieval_relevance: "N/A"\n'
        if is_baseline
        else "  grounding: score 1–5\n  retrieval_relevance: score 1–5\n"
    )

    return (
        f"System: {row['system']}\n"
        f"Question type: {row.get('question_type', '')}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Expected focus:\n{row.get('expected_focus', '')}\n\n"
        f"Generated answer:\n{row.get('generated_response', '')}\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        f"Scoring instructions for this row:\n"
        f"{grounding_note}"
        f"{style_note}"
    )


def score_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Call gemma3:4b to score one CSV row.
    Returns a dict with the five score fields plus a justification.
    """
    user_prompt = _build_user_prompt(row)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        options={
            "temperature": 0.0,   # deterministic scoring
            "num_predict": 150,   # JSON scores are short
            "num_gpu":     99,
        },
    )
    raw = response.message.content.strip()

    # Strip markdown code fences if the model adds them.
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    return {
        "correctness_score":         str(data.get("correctness", "")),
        "grounding_score":           str(data.get("grounding", "")),
        "retrieval_relevance_score": str(data.get("retrieval_relevance", "")),
        "usefulness_score":          str(data.get("usefulness", "")),
        "style_quality_score":       str(data.get("style_quality", "")),
        "justification":             data.get("justification", ""),
    }


def run_scoring() -> None:
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run evaluate.py first.")
        sys.exit(1)

    # Back up the unscored CSV before overwriting.
    shutil.copy(CSV_PATH, BACKUP_PATH)
    print(f"Backup saved → {BACKUP_PATH}")

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys())

    # Add justification column if not already present.
    if "justification" not in fieldnames:
        fieldnames.append("justification")

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        qid    = row.get("question_id", "")
        system = row.get("system", "")
        qt     = row.get("question_type", "")
        print(f"[{i:02d}/{total}] {system:8s} | {qid:5s} | {row['question'][:50]}...")

        # Off-topic rows: skip LLM call, just mark rejected status.
        if qt == "off_topic":
            rejected = _is_rejected(row.get("generated_response", ""))
            label = "correctly rejected" if rejected else "incorrectly answered"
            row.update(_na_scores(justification=f"Off-topic query — {label} by {system}."))
            row["rejected"] = str(rejected)
            if "rejected" not in fieldnames:
                fieldnames.append("rejected")
            continue

        # On-topic rows: call the LLM judge.
        try:
            scores = score_row(row)
            row.update(scores)
        except Exception as exc:
            print(f"  WARNING: scoring failed ({exc}) — leaving row blank.")

        # Small delay to avoid hammering the local Ollama server.
        if i < total:
            time.sleep(0.5)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nScores written → {CSV_PATH}")


if __name__ == "__main__":
    run_scoring()
