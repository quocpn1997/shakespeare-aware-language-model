"""
Prompt-only baseline system.

The baseline calls phi4-mini via Ollama with NO retrieved context — the model
answers from its parametric (pre-trained) knowledge alone.

Why this design?
  Using the same LLM (phi4-mini) for both baseline and RAG means any difference
  in evaluation scores directly reflects the contribution of retrieval, not a
  difference in model capability. A weaker LLM or a keyword-search baseline
  would confound the comparison.

The baseline system prompt is intentionally minimal — it does not instruct the
model to cite sources or restrict itself to specific passages. This maximises
the contrast with the grounded RAG system prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama
from config import OLLAMA_MODEL

# Minimal system prompt: no retrieval, no citation requirement, no passage grounding.
# Deliberately weaker than the RAG system prompt so the evaluation gap is meaningful.
BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant with general knowledge of Shakespeare's plays. "
    "Answer the user's question briefly and clearly in 3-5 sentences."
)


def baseline_answer(query: str) -> str:
    """
    Generate an answer using phi4-mini with no retrieved context.

    The model relies entirely on its pre-trained knowledge of Shakespeare.
    This is the comparison point for the RAG system in evaluation.
    """
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        options={"temperature": 0.2},  # Low temperature for consistent, factual answers.
    )
    # Extract the text content from the Ollama response object.
    return response.message.content.strip()


if __name__ == "__main__":
    questions = [
        "Who is Hamlet?",
        "Why does Macbeth kill Duncan?",
    ]
    for q in questions:
        print(f"Question: {q}")
        print(f"Answer:   {baseline_answer(q)}")
        print()
