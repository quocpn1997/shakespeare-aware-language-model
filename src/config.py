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

DEFAULT_TOP_K = 3

# Embedding model for encoding chunks and queries into dense vectors.
# google/embeddinggemma-300m is a Gemma-based decoder model fine-tuned for
# retrieval. Its larger context window (2048+ tokens) and 300M parameters
# give better semantic coverage than smaller BERT-based alternatives.
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"

# Local Ollama model used for both the prompt-only baseline and RAG generation.
OLLAMA_MODEL = "phi4-mini:latest"

# Where the built embedding matrix and serialised chunks are saved so that
# subsequent runs can skip the expensive re-embedding step.
INDEX_PATH = DATA_DIR / "index.npz"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
