"""
RAG chatbot: retrieval-augmented generation using phi4-mini via Ollama.

Pipeline:
  1. Load (or build) the embedding index from data/processed/.
  2. For a query, retrieve the top-k most similar utterance-window chunks.
  3. Build a prompt that injects the retrieved passages as context.
  4. Call phi4-mini via Ollama to generate a grounded answer.

Three generation modes:
  - qa (default): grounded factual answers with act/scene citations.
    Uses prompts/system_prompt.txt. Temperature 0.2 for consistency.
  - concept: structured definition of a character, concept, or relationship.
    Uses prompts/concept_prompt.txt. Temperature 0.2 for consistency.
  - stylised: short Shakespearean-voice creative responses.
    Uses prompts/stylised_prompt.txt. Temperature 0.7 for creativity.

Mode is selected by question_type from the evaluation JSON, or by keyword
detection in the query text when question_type is not provided:
  - "stylised_generation"  → stylised  (keywords: "soliloquy", "in the voice of", …)
  - "concept_explanation"  → concept   (keywords: "who is", "what is", …)
  - "contextual_qa"        → qa        (default)

Public API (used by evaluate.py):
  rag_answer(query, retriever, top_k, question_type) -> (answer, retrieved_passages)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama
from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    INDEX_PATH,
    CHUNKS_PATH,
    OLLAMA_MODEL,
    PROMPT_DIR,
)
from chunking import format_chunk_for_display
from retrieval import EmbeddingRetriever
from build_index import build_or_load_index


Chunk = Dict[str, Any]

# Keywords for mode detection when question_type is not supplied explicitly.
STYLISED_KEYWORDS = ("shakespearean-style", "in the voice of", "soliloquy", "stylised")
CONCEPT_KEYWORDS  = ("who is", "what is", "what are", "what role", "what does")

# Maps mode name → (prompt file, temperature).
_MODE_CONFIG: Dict[str, Tuple[str, float]] = {
    "qa":       ("system_prompt.txt",  0.2),
    "concept":  ("concept_prompt.txt", 0.2),
    "stylised": ("stylised_prompt.txt", 0.7),
}


def _load_prompt(filename: str) -> str:
    """Read a prompt template file from the prompts/ directory."""
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            "Run from the project root so PROMPT_DIR resolves correctly."
        )
    return path.read_text(encoding="utf-8").strip()


def get_mode(query: str, question_type: str = "") -> str:
    """
    Return the generation mode: 'stylised', 'concept', or 'qa'.

    Checks the explicit question_type field first (from evaluation JSON),
    then falls back to keyword detection in the query text.
    """
    if question_type == "stylised_generation":
        return "stylised"
    if question_type == "concept_explanation":
        return "concept"
    if question_type == "contextual_qa":
        return "qa"

    # Keyword fallback for interactive use (no question_type provided).
    query_lower = query.lower()
    if any(kw in query_lower for kw in STYLISED_KEYWORDS):
        return "stylised"
    if any(kw in query_lower for kw in CONCEPT_KEYWORDS):
        return "concept"
    return "qa"


def build_rag_user_block(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Build the user-turn block for the RAG prompt.

    The system prompt (mode-specific instructions) is passed separately as
    the 'system' role in ollama.chat(). This function produces only the
    context + question portion that goes in the 'user' role.
    """
    # Note: we intentionally omit "Passage N" labels — phi4-mini was inventing
    # citations like "(Passage 1)" that pointed to the wrong scene. With only the
    # Play/Act/Scene header visible inside each chunk, the model is forced to
    # cite using (Play, Act X, Scene Y), which is the format we want.
    context = "\n\n---\n\n".join(
        format_chunk_for_display(chunk) for chunk, _ in retrieved
    )

    return (
        f"Question: {query}\n\n"
        f"Use the following retrieved passages to answer. If the topic appears in any "
        f"passage, describe what the passages show — a partial answer is always better "
        f"than refusing.\n\n"
        f"Retrieved passages:\n\n"
        f"{context}"
    )


def generate_answer(user_block: str, mode: str = "qa") -> str:
    """
    Call phi4-mini via Ollama to generate an answer.

    Selects the system prompt and temperature based on mode:
      qa       → system_prompt.txt,  temp 0.2 (factual, consistent)
      concept  → concept_prompt.txt, temp 0.2 (structured definition)
      stylised → stylised_prompt.txt, temp 0.7 (creative variation)
    """
    prompt_file, temperature = _MODE_CONFIG.get(mode, _MODE_CONFIG["qa"])
    system_prompt = _load_prompt(prompt_file)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_block},
        ],
        options={"temperature": temperature},
    )
    return response.message.content.strip()


def rag_answer(
    query: str,
    retriever: EmbeddingRetriever,
    top_k: int = DEFAULT_TOP_K,
    question_type: str = "",
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Full RAG pipeline: retrieve → prompt → generate.

    Returns:
        answer:    Generated text from phi4-mini.
        retrieved: List of (chunk, score) pairs used as context.

    This is the entry point used by evaluate.py.
    """
    mode = get_mode(query, question_type)
    retrieved = retriever.retrieve(query, top_k=top_k)
    user_block = build_rag_user_block(query, retrieved)
    answer = generate_answer(user_block, mode=mode)
    return answer, retrieved


def main() -> None:
    """Interactive RAG chatbot loop."""
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    build_or_load_index(retriever)

    print("\nShakespeare RAG Chatbot (phi4-mini)")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        mode = get_mode(query)
        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        # Show retrieved evidence so the user can see what grounded the answer.
        print("\nRetrieved passages:")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"  [{rank}] score={score:.4f} | {chunk['chunk_id']}")
            print(f"       {chunk['text'][:120].strip()!r}")

        user_block = build_rag_user_block(query, retrieved)
        answer = generate_answer(user_block, mode=mode)

        print(f"\nAnswer ({mode} mode):")
        print(answer)
        print()


if __name__ == "__main__":
    main()
