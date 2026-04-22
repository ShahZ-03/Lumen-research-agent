# backend/chunker.py — sentence-aware chunking with overlap
from __future__ import annotations

import re
from collections import deque

# Sentence boundary: end punctuation followed by whitespace or EOS
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")

MIN_CHUNK_CHARS = 50
MAX_SENTENCE_CHARS = 800
OVERLAP_CHARS = 100
TARGET_CHUNK_CHARS = 800


def _truncate_sentence(s: str) -> list[str]:
    """Hard-truncate any single sentence longer than MAX_SENTENCE_CHARS."""
    s = s.strip()
    if not s:
        return []
    if len(s) <= MAX_SENTENCE_CHARS:
        return [s]
    return [s[i : i + MAX_SENTENCE_CHARS] for i in range(0, len(s), MAX_SENTENCE_CHARS)]


def _sentence_units(text: str) -> list[str]:
    text = text.replace("\r", "\n").strip()
    if not text:
        return []
    out: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(text):
        p = part.strip()
        if p:
            out.extend(_truncate_sentence(p))
    return out


def chunk_text(text: str, target_chunk_chars: int = TARGET_CHUNK_CHARS) -> list[str]:
    """
    Split on sentence boundaries (regex), hard-cap any sentence at 800 chars,
    pack into chunks up to ``target_chunk_chars``, each new chunk prefixed with the
    last ``OVERLAP_CHARS`` of the previous chunk, drop chunks under ``MIN_CHUNK_CHARS``.
    """
    units = deque(_sentence_units(text))
    if not units:
        return []

    chunks: list[str] = []
    overlap_prefix = ""

    while units:
        parts: list[str] = []
        length = 0

        if overlap_prefix:
            parts.append(overlap_prefix)
            length = len(overlap_prefix)

        while units:
            s = units[0]
            sep = 1 if parts else 0
            add = len(s) + sep
            if parts and length + add > target_chunk_chars:
                break
            units.popleft()
            parts.append(s)
            length += add

        only_overlap = bool(overlap_prefix) and parts == [overlap_prefix]
        if only_overlap and units:
            s = units.popleft()
            parts.append(s)

        body = " ".join(parts).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append(body)
            overlap_prefix = body[-OVERLAP_CHARS:] if len(body) >= OVERLAP_CHARS else body
        else:
            overlap_prefix = overlap_prefix if chunks else ""
            if units:
                units.popleft()

    return chunks
