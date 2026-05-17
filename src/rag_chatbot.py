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

import re
import sys
import time
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
from baseline import baseline_answer
from chunking import format_chunk_for_display
from retrieval import EmbeddingRetriever
from build_index import build_or_load_index


Chunk = Dict[str, Any]

# Keywords for mode detection when question_type is not supplied explicitly.
STYLISED_KEYWORDS = ("shakespearean-style", "in the voice of", "soliloquy", "stylised")
CONCEPT_KEYWORDS  = ("who is", "what is", "what are", "what role", "what does")

# Character / place names that uniquely identify each play. Used to filter
# retrieval to one play and prevent cross-play contamination (e.g. an R&J
# question that retrieves a Macbeth passage at low similarity). Keys are the
# canonical play names as stored in chunk['play'].
PLAY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Macbeth": (
        "macbeth", "duncan", "banquo", "macduff", "lady macbeth", "malcolm",
        "donalbain", "fleance", "hecate", "lennox", "ross", "siward",
        "dunsinane", "birnam", "cawdor", "glamis", "scone",
    ),
    "Hamlet": (
        "hamlet", "claudius", "gertrude", "ophelia", "polonius", "laertes",
        "horatio", "rosencrantz", "guildenstern", "fortinbras", "yorick",
        "denmark", "elsinore",
    ),
    "Romeo and Juliet": (
        "romeo", "juliet", "tybalt", "mercutio", "benvolio", "capulet",
        "montague", "paris", "rosaline", "friar lawrence", "friar laurence",
        "verona",
    ),
}


# Patterns for phi4-mini's recurring citation mistakes. Each captures the act
# and scene numbers separately so we can reassemble in the canonical form
# "(Play, Act X, Scene Y)" — comma-and-all — regardless of what the model wrote.
#   1. "(Act 3, Scene 1)" or "(Act 3 Scene 1)"  — paren but missing play, may
#      also be missing the comma between Act and Scene.
#   2. "in Act 3, Scene 1," or "In Act 3 Scene 1," — narrative opener instead
#      of a citation, with or without the comma.
# The system prompt shows CORRECT/WRONG examples but phi4-mini still drifts as
# the answer gets longer. Post-processing is more reliable.
_BARE_PAREN_CITATION_RE = re.compile(
    r"\(Act\s+(\d+)[,\s]+Scene\s+(\d+)\)", re.IGNORECASE
)
_INLINE_CITATION_RE = re.compile(
    r"(?<![A-Za-z])[Ii]n\s+Act\s+(\d+)[,\s]+Scene\s+(\d+)(?=[,\s.])",
    re.IGNORECASE,
)


def fix_citations(answer: str, play: str | None) -> str:
    """
    Repair citations phi4-mini produces in non-standard formats.

    Transformations (all converge on "(Play, Act X, Scene Y)"):
      "(Act X, Scene Y)"     → "(Play, Act X, Scene Y)"
      "(Act X Scene Y)"      → "(Play, Act X, Scene Y)"  ← missing comma fixed
      "In Act X, Scene Y, …" → "(Play, Act X, Scene Y) …"
      "in Act X Scene Y …"   → "(Play, Act X, Scene Y) …"

    Only runs when we know which play the answer is about (passed in or
    auto-detected). If the play is unknown, the answer is returned unchanged.
    Citations that already include the play name are left alone — the regex
    only matches strings that start with "Act" or "In Act".
    """
    if not play:
        return answer
    answer = _BARE_PAREN_CITATION_RE.sub(
        lambda m: f"({play}, Act {m.group(1)}, Scene {m.group(2)})", answer
    )
    answer = _INLINE_CITATION_RE.sub(
        lambda m: f"({play}, Act {m.group(1)}, Scene {m.group(2)})", answer
    )
    return answer


def detect_play(query: str) -> str | None:
    """
    Return the canonical play name if the query clearly refers to one play.

    Counts how many play-specific keywords appear in the query for each play,
    returns the winner if there is a unique top scorer. Returns None for
    ambiguous queries (no keywords matched, or a tie) — in that case the
    caller should retrieve without a play filter.
    """
    query_lower = query.lower()
    scores = {
        play: sum(1 for kw in keywords if kw in query_lower)
        for play, keywords in PLAY_KEYWORDS.items()
    }
    scores = {p: s for p, s in scores.items() if s > 0}
    if not scores:
        return None
    top_score = max(scores.values())
    top_plays = [p for p, s in scores.items() if s == top_score]
    return top_plays[0] if len(top_plays) == 1 else None

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
    )
    return response.message.content.strip()


def rag_answer(
    query: str,
    retriever: EmbeddingRetriever,
    top_k: int = DEFAULT_TOP_K,
    question_type: str = "",
    play: str | None = None,
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Full RAG pipeline: retrieve → prompt → generate.

    Args:
        query:         User question text.
        retriever:     Embedding retriever with a built/loaded index.
        top_k:         Number of passages to retrieve.
        question_type: From the evaluation JSON ("contextual_qa" | "concept_explanation"
                       | "stylised_generation"); selects the generation mode.
        play:          Canonical play name ("Hamlet" / "Macbeth" / "Romeo and Juliet").
                       When supplied, retrieval is restricted to that play's chunks.
                       When omitted, we try to auto-detect the play from character
                       names in the query, falling back to no filter if unclear.

    Returns:
        answer:    Generated text from phi4-mini.
        retrieved: List of (chunk, score) pairs used as context.

    This is the entry point used by evaluate.py.
    """
    # Minimum cosine similarity for the top retrieved chunk to be considered
    # relevant. Below this threshold the corpus contains nothing about the
    # topic and we return a "not found" response without calling the LLM.
    # Calibrated on the corpus: on-topic queries score ≥0.64, off-topic ≤0.51.
    MIN_RELEVANCE_SCORE = 0.58

    mode = get_mode(query, question_type)
    play_filter = play if play else detect_play(query)
    retrieved = retriever.retrieve(query, top_k=top_k, play_filter=play_filter)

    top_score = retrieved[0][1] if retrieved else 0.0
    if top_score < MIN_RELEVANCE_SCORE:
        not_found = (
            "I could not find relevant information about that topic in the "
            "Shakespeare passages I have access to (Hamlet, Macbeth, and "
            "Romeo and Juliet)."
        )
        return not_found, retrieved

    user_block = build_rag_user_block(query, retrieved)
    answer = generate_answer(user_block, mode=mode)
    # Repair any "(Act X, Scene Y)" citations missing the play name.
    if mode != "stylised":
        answer = fix_citations(answer, play_filter)
    return answer, retrieved


def main() -> None:
    """Interactive RAG chatbot loop with /rag and /baseline toggle."""
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    build_or_load_index(retriever)

    system_mode = "rag"  # current system: "rag" or "baseline"

    print(f"\nShakespeare RAG Chatbot ({OLLAMA_MODEL})")
    print("Commands: /rag  /baseline  /quit")
    print(f"Current mode: {system_mode}\n")

    while True:
        query = input(f"[{system_mode}] Question: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "/quit"}:
            break

        # Toggle commands.
        if query.lower() == "/rag":
            system_mode = "rag"
            print(f"Switched to RAG mode.\n")
            continue
        if query.lower() == "/baseline":
            system_mode = "baseline"
            print(f"Switched to baseline mode.\n")
            continue

        if system_mode == "baseline":
            t0 = time.time()
            answer = baseline_answer(query)
            elapsed = time.time() - t0
            print(f"\nAnswer (baseline) [{elapsed:.1f}s]:")
            print(answer)
            print()
        else:
            play_filter = detect_play(query)
            mode = get_mode(query)

            # Route through rag_answer() so the relevance threshold is enforced.
            t0 = time.time()
            answer, retrieved = rag_answer(query, retriever, top_k=DEFAULT_TOP_K)
            elapsed = time.time() - t0

            filter_note = f" [filtered to {play_filter}]" if play_filter else ""
            print(f"\nRetrieved passages{filter_note}:")
            for rank, (chunk, score) in enumerate(retrieved, start=1):
                print(f"  [{rank}] score={score:.4f} | {chunk['chunk_id']}")
                print(f"       {chunk['text'][:120].strip()!r}")

            print(f"\nAnswer ({mode} mode) [{elapsed:.1f}s]:")
            print(answer)
            print()


if __name__ == "__main__":
    main()
