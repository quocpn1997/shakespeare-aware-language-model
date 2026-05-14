"""
Build (or reload) the retrieval index and run a sanity-check query.

Workflow:
  1. Load all scenes from data/raw/ and create utterance-window chunks.
  2. Try to load a previously saved index from data/processed/.
     If found, skip the ~5.6s embedding step.
  3. If no cached index exists, encode all chunks and save to disk.
  4. Run a sanity-check query and print the top-3 results so you can
     verify that retrieval is returning plausible Shakespeare passages.

Run this script once before using rag_chatbot.py or evaluate.py.
"""

import sys
from pathlib import Path

# Allow running as: python src/build_index.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_PATH, CHUNKS_PATH
from data_loader import load_all_scenes
from chunking import create_utterance_window_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def build_or_load_index(retriever: EmbeddingRetriever) -> None:
    """
    Load the index from disk if it exists, otherwise build and save it.

    This avoids re-encoding 633 chunks every time a script starts up.
    """
    if retriever.load_index(INDEX_PATH, CHUNKS_PATH):
        # Cache hit — skip encoding.
        print(f"Loaded cached index ({len(retriever.chunks)} chunks).")
        return

    # Cache miss — load raw data, chunk, encode, persist.
    print("No cached index found. Building from scratch...")
    scenes = load_all_scenes()
    chunks = create_utterance_window_chunks(scenes)
    print(f"Loaded {len(scenes)} scenes → {len(chunks)} utterance-window chunks.")

    retriever.build_index(chunks)
    retriever.save_index(INDEX_PATH, CHUNKS_PATH)


def main() -> None:
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    build_or_load_index(retriever)

    # Sanity-check query: top-3 results should all come from Macbeth Act 1-2,
    # covering the witches' prophecy, Lady Macbeth's persuasion, or the murder.
    query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    print(f"\nSanity-check query: {query!r}")
    print()

    for rank, (chunk, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk))
        print()


if __name__ == "__main__":
    main()
