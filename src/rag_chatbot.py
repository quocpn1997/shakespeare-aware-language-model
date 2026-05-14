"""
RAG chatbot: retrieval-augmented generation using phi4-mini via Ollama.

Pipeline:
  1. Load (or build) the embedding index from data/processed/.
  2. For a query, retrieve the top-k most similar utterance-window chunks.
  3. Build a prompt that injects the retrieved passages as context.
  4. Call phi4-mini via Ollama to generate a grounded answer.

Two generation modes:
  - QA mode (default): grounded factual answers with act/scene citations.
    Uses prompts/system_prompt.txt. Temperature 0.2 for consistency.
  - Stylised mode: short Shakespearean-voice creative responses.
    Uses prompts/stylised_prompt.txt. Temperature 0.7 for creativity.

Stylised mode is triggered when the question type is "stylised_generation"
or when the query contains keywords like "Shakespearean-style", "in the voice
of", or "soliloquy".

Public API (used by evaluate.py):
  rag_answer(query, retriever, top_k, stylised) -> (answer, retrieved_passages)
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

# Keywords that signal the user wants a stylised Shakespearean response.
STYLISED_KEYWORDS = ("shakespearean-style", "in the voice of", "soliloquy", "stylised")


def _load_prompt(filename: str) -> str:
    """Read a prompt template file from the prompts/ directory."""
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            "Run from the project root so PROMPT_DIR resolves correctly."
        )
    return path.read_text(encoding="utf-8").strip()


def is_stylised(query: str, question_type: str = "") -> bool:
    """
    Return True if the query should use the stylised generation prompt.

    Checks the explicit question_type field first (from evaluation JSON),
    then falls back to keyword detection in the query text.
    """
    if question_type == "stylised_generation":
        return True
    query_lower = query.lower()
    return any(kw in query_lower for kw in STYLISED_KEYWORDS)


def build_rag_user_block(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Build the user-turn block for the RAG prompt.

    The system prompt (grounding rules, citation instructions) is passed
    separately as the 'system' role in ollama.chat(). This function produces
    only the context + question portion that goes in the 'user' role.
    """
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Passage {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )

    context = "\n\n".join(context_blocks)

    return (
        f"Question: {query}\n\n"
        f"Use the following retrieved passages to answer. If the topic appears in any "
        f"passage, describe what the passages show — a partial answer is always better "
        f"than refusing.\n\n"
        f"Retrieved passages:\n\n"
        f"{context}"
    )


def generate_answer(user_block: str, stylised: bool = False) -> str:
    """
    Call phi4-mini via Ollama to generate an answer.

    Loads the appropriate system prompt from prompts/ and passes it as the
    'system' role so the model knows whether to produce a grounded factual
    answer or a Shakespearean-style creative response.
    """
    prompt_file = "stylised_prompt.txt" if stylised else "system_prompt.txt"
    system_prompt = _load_prompt(prompt_file)

    # Lower temperature for factual QA (reproducible answers);
    # higher for stylised (creative variation is desirable).
    temperature = 0.7 if stylised else 0.2

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

    This helper is the entry point used by evaluate.py so it doesn't need
    to duplicate the retrieve → prompt → generate logic.
    """
    stylised = is_stylised(query, question_type)
    retrieved = retriever.retrieve(query, top_k=top_k)
    user_block = build_rag_user_block(query, retrieved)
    answer = generate_answer(user_block, stylised=stylised)
    return answer, retrieved


def main() -> None:
    """Interactive RAG chatbot loop."""
    # Load the cached index (or build it if missing).
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

        stylised = is_stylised(query)
        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        # Show retrieved evidence so the user can see what grounded the answer.
        print("\nRetrieved passages:")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"  [{rank}] score={score:.4f} | {chunk['chunk_id']}")
            print(f"       {chunk['text'][:120].strip()!r}")

        user_block = build_rag_user_block(query, retrieved)
        answer = generate_answer(user_block, stylised=stylised)

        mode = "stylised" if stylised else "QA"
        print(f"\nAnswer ({mode} mode):")
        print(answer)
        print()


if __name__ == "__main__":
    main()
