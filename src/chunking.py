"""
Chunking utilities for the Shakespeare RAG pipeline.

Strategy: utterance-window chunking.
  - Slide a window of `window` utterances over each scene with step `stride`.
  - Overlap = window - stride utterances, preserving dialogue continuity across
    speaker boundaries (important for exchanges like the Romeo/Juliet balcony scene).
  - Each chunk's text is prefixed with scene metadata so the embedding captures
    play/act/scene context even when the dialogue alone is ambiguous.
  - Stage directions are included and labelled [Direction] so the model retains
    narrative context (entrances, exits, off-stage events) without confusing them
    with spoken lines.

Why utterance-window over scene-level chunks?
  google/embeddinggemma-300m supports up to 2048 tokens, so truncation is no longer
  a hard limit — but utterance windows still produce more focused, topically coherent
  embeddings than dumping an entire scene into one vector. Shorter, overlapping chunks
  also improve citation precision: retrieved evidence points to a specific exchange
  rather than a whole scene.
"""

from __future__ import annotations

from typing import Any, Dict, List


Record = Dict[str, Any]
Chunk = Dict[str, Any]
Scene = Dict[str, Any]


_COPYRIGHT_MARKERS = ("COPYRIGHT", "WORLD LIBRARY", "COMMERCIALLY", "MACHINE READABLE")


def _is_noise(utt: Dict[str, Any]) -> bool:
    """Return True if an utterance is dataset boilerplate rather than play content."""
    text = utt.get("text", "").upper()
    return any(marker in text for marker in _COPYRIGHT_MARKERS)


def _render_utterance(utt: Dict[str, Any]) -> str:
    """
    Render one utterance as a single display line.

    Stage directions get a [Direction] label to distinguish them from spoken lines.
    """
    speaker = utt.get("speaker", "").strip()
    text = utt.get("text", "").strip()

    if speaker == "STAGE_DIRECTION":
        return f"[Direction]: {text}"
    return f"{speaker}: {text}"


def create_utterance_window_chunks(
    scenes: List[Scene],
    window: int = 8,
    stride: int = 6,
) -> List[Chunk]:
    """
    Slide a fixed-size window over each scene's utterances to produce chunks.

    Each chunk contains:
      - A header line with play, act, scene, and the instructor-provided scene summary.
      - Up to `window` utterances rendered as "SPEAKER: text" (or "[Direction]: text").

    Args:
        scenes:  List of scene dicts from load_all_scenes(), each with an
                 "utterances" list and scene-level metadata.
        window:  Number of utterances per chunk (default 8).
        stride:  How many utterances to advance the window each step (default 6).
                 window - stride = overlap between consecutive chunks.

    Returns:
        Flat list of chunk dicts ready for embedding.
    """
    chunks: List[Chunk] = []

    for scene in scenes:
        scene_id = scene.get("scene_id", "unknown")
        play = scene.get("play", scene.get("play_key", "Unknown"))
        act = scene.get("act", "?")
        scene_num = scene.get("scene", "?")
        summary = scene.get("scene_summary", "").strip()
        utterances = scene.get("utterances", [])

        if not utterances:
            continue

        # Header prepended to every chunk from this scene so the embedding
        # captures play/act/scene context even for ambiguous short exchanges.
        scene_header = f"[{play} | Act {act}, Scene {scene_num} | {summary}]"

        # Slide the window; range() stops before we'd start a chunk with nothing.
        for start in range(0, len(utterances), stride):
            window_utts = utterances[start : start + window]

            # Drop boilerplate (copyright notices embedded in the source data).
            window_utts = [u for u in window_utts if not _is_noise(u)]
            if not window_utts:
                continue

            # Skip direction-only windows — pure stage directions embed poorly
            # and surface as false positives for unrelated dialogue queries.
            has_dialogue = any(
                u.get("speaker") and u["speaker"] != "STAGE_DIRECTION"
                for u in window_utts
            )
            if not has_dialogue:
                continue

            # Render each utterance as a display line.
            lines = [_render_utterance(u) for u in window_utts]
            chunk_text = scene_header + "\n" + "\n".join(lines)

            # Collect unique non-direction speakers for metadata filtering.
            speakers = list({
                u["speaker"]
                for u in window_utts
                if u.get("speaker") and u["speaker"] != "STAGE_DIRECTION"
            })

            # source_ids link back to the original utterance records.
            source_ids = [u.get("source_id", "") for u in window_utts if u.get("source_id")]

            chunks.append({
                "chunk_id": f"{scene_id}_w{start:04d}",
                "play": play,
                "act": act,
                "scene": scene_num,
                "scene_summary": summary,
                "speakers": speakers,
                "source_ids": source_ids,
                "text": chunk_text,
            })

    return chunks


def _get_text(record: Record) -> str:
    """
    Extract text from a flat record using common field names.
    Used by the legacy create_chunks() function.
    """
    for key in ["text", "utterance", "excerpt", "content", "passage"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallback: combine speaker + summary fields if no direct text field exists.
    parts = []
    for key in ["speaker", "summary", "modern_summary"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return " ".join(parts).strip()


def create_chunks(records: List[Record]) -> List[Chunk]:
    """
    Legacy chunker: one flat record becomes one retrieval chunk.

    Kept for backwards compatibility with the original scaffold. The preferred
    approach is create_utterance_window_chunks() which handles overlap and
    scene-level context injection.
    """
    chunks: List[Chunk] = []

    for i, record in enumerate(records):
        text = _get_text(record)
        if not text:
            continue

        chunk = {
            "chunk_id": record.get("source_id") or record.get("id") or f"chunk_{i:06d}",
            "play": record.get("play", record.get("play_key", "unknown")),
            "act": record.get("act", None),
            "scene": record.get("scene", None),
            "speaker": record.get("speaker", None),
            "text": text,
            "metadata": record,
        }
        chunks.append(chunk)

    return chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    """
    Format a retrieved chunk as a readable string for printing or prompt injection.

    Handles both utterance-window chunks (which have a "speakers" list) and
    legacy flat chunks (which have a single "speaker" string field).
    """
    play = chunk.get("play", "Unknown play")
    act = chunk.get("act", "?")
    scene = chunk.get("scene", "?")

    # Utterance-window chunks store a list of speakers; legacy chunks a single string.
    speakers = chunk.get("speakers")
    if speakers:
        speaker_str = ", ".join(speakers) if speakers else ""
        header = f"{play}, Act {act}, Scene {scene} | Speakers: {speaker_str}"
    else:
        speaker = chunk.get("speaker", "")
        header = f"{play}, Act {act}, Scene {scene}"
        if speaker:
            header += f", Speaker: {speaker}"

    return f"[{header}]\n{chunk.get('text', '')}"
