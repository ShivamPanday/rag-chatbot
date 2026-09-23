"""
chunker.py
-----------
Splits document text into overlapping chunks for embedding, trying to
break on paragraph/sentence boundaries first (rather than blindly cutting
mid-sentence), which keeps retrieved chunks more coherent and readable.
"""

import re

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Try splitting on these separators in order, falling back to the next
# one only if a piece is still too large
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]

    sep = separators[0]
    pieces = text.split(sep)

    result = []
    for piece in pieces:
        if len(piece) <= CHUNK_SIZE:
            if piece.strip():
                result.append(piece)
        else:
            result.extend(_split_recursive(piece, separators[1:]))
    return result


def _merge_with_overlap(pieces: list[str]) -> list[str]:
    """Greedily pack small pieces into chunks up to CHUNK_SIZE, adding
    character-level overlap between consecutive chunks."""
    chunks = []
    current = ""

    for piece in pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the end of the last one
            overlap_text = current[-CHUNK_OVERLAP:] if current else ""
            current = (overlap_text + " " + piece).strip()

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Returns a list of {"text": ..., "source": ..., "chunk_id": ...} dicts.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    pieces = _split_recursive(text, _SEPARATORS)
    chunks = _merge_with_overlap(pieces)

    return [
        {"text": chunk, "source": source, "chunk_id": f"{source}::{i}"}
        for i, chunk in enumerate(chunks)
    ]
