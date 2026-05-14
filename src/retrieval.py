"""
Embedding and retrieval utilities.

Design choices (justified in the report):
- BAAI/bge-small-en-v1.5 for embeddings: 512-token context, zero NaN issues,
  identical top-3 results to bge-base at half the build time on this corpus.
- numpy dot product for retrieval: after L2-normalisation cosine similarity
  reduces to a dot product (embeddings @ query), ~25x faster than sklearn's
  cosine_similarity which recomputes norms on every call.
- BGE query prefix: BGE models are trained with an asymmetric setup — documents
  are encoded as-is, queries get the prefix
  "Represent this sentence for searching relevant passages: {text}".
  Omitting it measurably degrades retrieval quality.
- Index persistence: embeddings are saved to .npz and chunks to JSONL so
  repeat runs skip the ~5.6s encoding step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Chunk = Dict[str, Any]

# BGE models use this prefix on queries only (not on indexed documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingRetriever:
    """
    Embedding-based retriever using numpy dot product over L2-normalised vectors.
    """

    def __init__(self, embedding_model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.chunks: List[Chunk] = []
        # Stored as L2-normalised float32 matrix (n_chunks × dims).
        self.embeddings: Optional[np.ndarray] = None

    def _is_bge(self) -> bool:
        """Return True if the model is a BGE model that needs a query prefix."""
        return "bge" in self.model_name.lower()

    def _normalise(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise a batch of vectors row-wise."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Avoid division by zero for any zero vectors.
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Encode all chunks and build the normalised embedding matrix.

        Embeddings are L2-normalised so that retrieval is a plain dot product.
        """
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Encode documents without any prefix (BGE asymmetric convention).
        raw = np.asarray(self.model.encode(texts, show_progress_bar=True))
        self.embeddings = self._normalise(raw)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Return the top-k most similar chunks for a query.

        After L2 normalisation, cosine similarity = dot product, so retrieval
        is a single matrix-vector multiply: embeddings @ query_vec.
        """
        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build_index() first.")

        # BGE models need the query prefix at search time only.
        query_text = f"{BGE_QUERY_PREFIX}{query}" if self._is_bge() else query
        query_vec = self._normalise(np.asarray(self.model.encode([query_text])))

        # Dot product over normalised vectors = cosine similarity.
        scores = (self.embeddings @ query_vec.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save_index(self, npz_path: Path, chunks_path: Path) -> None:
        """
        Persist the embedding matrix and chunk metadata to disk.

        - npz_path:    numpy .npz file storing the normalised embedding matrix.
        - chunks_path: JSONL file storing the chunk dicts (one per line).

        Saving means subsequent runs can call load_index() and skip re-encoding.
        """
        if self.embeddings is None:
            raise RuntimeError("Nothing to save — build_index() has not been called.")

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embeddings=self.embeddings)

        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                # Exclude the raw metadata field to keep the file small.
                record = {k: v for k, v in chunk.items() if k != "metadata"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(self.chunks)} chunks to {chunks_path}")
        print(f"Saved embeddings to {npz_path}")

    def load_index(self, npz_path: Path, chunks_path: Path) -> bool:
        """
        Load a previously saved index from disk.

        Returns True if loading succeeded, False if either file is missing
        (caller should then call build_index() and save_index()).
        """
        if not npz_path.exists() or not chunks_path.exists():
            return False

        self.embeddings = np.load(npz_path)["embeddings"]

        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

        print(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
        return True
