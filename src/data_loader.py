"""
Data loading utilities for the Shakespeare RAG pipeline.

Two public APIs are provided:
- load_all_plays()  : flat list of scene/utterance records (legacy, kept for
                      backwards compatibility with the original scaffold).
- load_all_scenes() : flat list of scene dicts, each carrying its full
                      utterances list. Used by the utterance-window chunker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES


Record = Dict[str, Any]
Scene = Dict[str, Any]


def _extract_records(obj: Any) -> List[Record]:
    """
    Pull a list of records out of a JSON object regardless of its top-level shape.

    The instructor dataset uses {"scenes": [...]} at the top level.
    This helper also handles plain lists and other common wrapper keys so the
    loader stays flexible if the schema changes.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Try common wrapper keys in priority order.
        for key in ["records", "utterances", "scenes", "chunks", "data"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

    raise ValueError(
        "Could not extract records. Expected a list or a dictionary containing "
        "one of: records, utterances, scenes, chunks, data."
    )


def load_json_records(path: Path) -> List[Record]:
    """
    Load one Shakespeare JSON file and return its records as a flat list.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {path}\n"
            "Ensure the raw dataset files are present in data/raw/."
        )

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    return _extract_records(obj)


def load_all_plays() -> List[Record]:
    """
    Load scene-level records from all three plays as a flat list.

    Each record gets a 'play_key' field (e.g. 'hamlet') so downstream code
    can identify which play a record belongs to without parsing the filename.
    """
    all_records: List[Record] = []

    for play_key, path in PLAY_FILES.items():
        records = load_json_records(path)
        for r in records:
            r.setdefault("play_key", play_key)
        all_records.extend(records)

    return all_records


def load_all_scenes() -> List[Scene]:
    """
    Load every scene from all three plays, preserving the nested utterances list.

    Each returned scene dict looks like:
        {
            "scene_id":      "macbeth_1_3",
            "play":          "Macbeth",
            "act":           1,
            "scene":         3,
            "location":      "A heath",
            "scene_summary": "...",
            "keywords":      [...],
            "utterances":    [{"speaker": ..., "text": ..., "source_id": ...}, ...],
            "text":          "...full scene text..."
        }

    This is the entry point used by create_utterance_window_chunks() in chunking.py.
    """
    all_scenes: List[Scene] = []

    for play_key, path in PLAY_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find dataset file: {path}\n"
                "Ensure the raw dataset files are present in data/raw/."
            )

        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        # The top-level JSON has a "scenes" key containing the scene list.
        scenes = obj.get("scenes", [])
        for scene in scenes:
            # Stamp the normalised play key so callers don't have to parse
            # the play name string (e.g. "Romeo and Juliet" vs "romeo_and_juliet").
            scene.setdefault("play_key", play_key)
        all_scenes.extend(scenes)

    return all_scenes


if __name__ == "__main__":
    # Quick smoke-test: print counts and the first scene summary.
    scenes = load_all_scenes()
    total_utterances = sum(len(s.get("utterances", [])) for s in scenes)
    print(f"Loaded {len(scenes)} scenes, {total_utterances} total utterances.")
    print("\nFirst scene summary:")
    print(f"  {scenes[0]['play']} — {scenes[0]['scene_id']}: {scenes[0].get('scene_summary', '')}")
