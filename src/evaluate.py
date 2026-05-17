"""
End-to-end evaluation pipeline.

Loads all evaluation questions, runs both the baseline and the RAG system
on each question, and writes the results to results/evaluation_results.csv.

Question sources:
  results/instructor_questions.json  — 9 merged instructor questions (IQ1-IQ9)
  results/designed_questions.json    — 15 designed questions (DQ1-DQ15)

For each question two rows are written (baseline + rag), giving 48 rows total.

Output columns:
  question_id, question, question_type, expected_focus,
  system, generated_response, retrieved_passages,
  top_retrieval_score, rejected,
  correctness_score, grounding_score, retrieval_relevance_score,
  usefulness_score, style_quality_score, comments

Score columns are left empty — run src/score.py afterwards to fill them
automatically using the local Ollama model as a rubric judge.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR, EMBEDDING_MODEL_NAME
from baseline import baseline_answer
from retrieval import EmbeddingRetriever
from build_index import build_or_load_index
from rag_chatbot import rag_answer, detect_play
from chunking import format_chunk_for_display

OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"

# The prefix the RAG pipeline returns when it rejects a low-relevance query.
REJECTION_PREFIX = "I could not find relevant information"

FIELDNAMES = [
    "question_id",
    "question",
    "question_type",
    "expected_focus",
    "system",
    "generated_response",
    "retrieved_passages",
    "top_retrieval_score",
    "rejected",
    "correctness_score",
    "grounding_score",
    "retrieval_relevance_score",
    "usefulness_score",
    "style_quality_score",
    "comments",
]


def load_questions() -> List[Dict[str, str]]:
    """Load questions from both source files, preserving order IQ then DQ."""
    sources = [
        RESULTS_DIR / "instructor_questions.json",
        RESULTS_DIR / "designed_questions.json",
    ]

    questions: List[Dict] = []
    for path in sources:
        if not path.exists():
            print(f"Warning: {path} not found — skipping.")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data:
            questions.append({
                "question_id":    q.get("question_id", ""),
                "question":       q.get("question", "").strip(),
                "question_type":  q.get("question_type", "contextual_qa"),
                "expected_focus": q.get("expected_focus", ""),
            })

    return questions


def format_retrieved(retrieved) -> str:
    """Serialise (chunk, score) pairs to a readable string for the CSV."""
    parts = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        header = f"[{rank}] score={score:.4f} | {chunk.get('chunk_id', '')}"
        parts.append(f"{header}\n{format_chunk_for_display(chunk)}")
    return "\n\n---\n\n".join(parts)


def run_evaluation() -> None:
    print("Loading questions...")
    questions = load_questions()
    print(f"  {len(questions)} questions loaded "
          f"({sum(1 for q in questions if q['question_id'].startswith('IQ'))} instructor, "
          f"{sum(1 for q in questions if q['question_id'].startswith('DQ'))} designed)")

    print("\nLoading embedding index...")
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    build_or_load_index(retriever)

    rows = []
    total = len(questions) * 2
    done  = 0

    for q in questions:
        qid  = q["question_id"]
        text = q["question"]
        qt   = q["question_type"]
        play = detect_play(text)   # auto-detect play from character names in query

        # ── Baseline ──────────────────────────────────────────────────────────
        print(f"[{done+1:02d}/{total}] baseline | {qid}: {text[:55]}...")
        t0       = time.time()
        b_answer = baseline_answer(text)
        b_time   = time.time() - t0

        rows.append({
            **q,
            "system":                    "baseline",
            "generated_response":        b_answer,
            "retrieved_passages":        "",
            "top_retrieval_score":       "",
            "rejected":                  "",
            "correctness_score":         "",
            "grounding_score":           "",
            "retrieval_relevance_score": "",
            "usefulness_score":          "",
            "style_quality_score":       "",
            "comments":                  f"generation_time={b_time:.1f}s",
        })
        done += 1

        # ── RAG ───────────────────────────────────────────────────────────────
        print(f"[{done+1:02d}/{total}] rag      | {qid}: {text[:55]}...")
        t0 = time.time()
        r_answer, retrieved = rag_answer(
            text, retriever, question_type=qt, play=play
        )
        r_time = time.time() - t0

        top_score = retrieved[0][1] if retrieved else 0.0
        rejected  = r_answer.strip().startswith(REJECTION_PREFIX)

        rows.append({
            **q,
            "system":                    "rag",
            "generated_response":        r_answer,
            "retrieved_passages":        format_retrieved(retrieved),
            "top_retrieval_score":       f"{top_score:.4f}",
            "rejected":                  str(rejected),
            "correctness_score":         "",
            "grounding_score":           "",
            "retrieval_relevance_score": "",
            "usefulness_score":          "",
            "style_quality_score":       "",
            "comments":                  f"generation_time={r_time:.1f}s",
        })
        done += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    on_topic  = sum(1 for q in questions if q["question_type"] != "off_topic")
    off_topic = sum(1 for q in questions if q["question_type"] == "off_topic")
    print(f"\nWrote {len(rows)} rows  "
          f"({on_topic} on-topic + {off_topic} off-topic) × 2 systems")
    print(f"Output → {OUTPUT_PATH}")
    print("Next:   python src/score.py")


if __name__ == "__main__":
    run_evaluation()
