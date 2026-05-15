"""
Configuration for the Assignment 2 starter code.

Students should adjust these values to match their own implementation.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/processed holds generated artefacts (index, chunk cache).
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# data/raw holds the instructor-provided Shakespeare JSON files.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Play JSON files sourced from the raw dataset directory.
PLAY_FILES = {
    "hamlet": RAW_DATA_DIR / "hamlet.json",
    "macbeth": RAW_DATA_DIR / "macbeth.json",
    "romeo_and_juliet": RAW_DATA_DIR / "romeo_and_juliet.json",
}

# Number of chunks to retrieve per query.
# Tuned to 5 after empirical testing:
#   - top_k=3 (scaffold default) misses corroborating evidence for "why/how"
#     questions where the answer is split across multiple scenes.
#   - top_k=8 was tried earlier but caused phi4-mini to over-quote and stitch
#     unrelated passages together, increasing hallucination rate especially in
#     concept mode ("Who is Hamlet?" pulled in seven scenes and the model
#     embellished each one with parametric knowledge).
#   - top_k=5 gives the model enough context for synthesis without
#     overwhelming its small instruction-following capacity.
# Performance cost at 633 chunks is negligible (~0.02ms per query either way).
DEFAULT_TOP_K = 5

# Embedding model for encoding chunks and queries into dense vectors.
# BAAI/bge-small-en-v1.5 was chosen over larger alternatives because:
#   - 512-token context fits our utterance-window chunks without truncation
#   - Zero NaN embeddings (decoder-based models like EmbeddingGemma-300M
#     produce NaN via mean pooling on ~3% of chunks)
#   - Benchmarking showed identical top-3 results vs bge-base at half the
#     build time (5.6s vs 11.8s) on the 633-chunk corpus
# BGE models require a prefix on queries (not documents) at retrieval time;
# this is handled in EmbeddingRetriever.retrieve() in retrieval.py.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Local Ollama model used for both the prompt-only baseline and RAG generation.
OLLAMA_MODEL = "phi4-mini:latest"

# Where the built embedding matrix and serialised chunks are saved so that
# subsequent runs can skip the expensive re-embedding step.
INDEX_PATH = DATA_DIR / "index.npz"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
