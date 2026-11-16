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

    return chunks


def clean_text_for_chunking(text: str) -> str:
    """
    Improve chunk quality by removing common web/markdown noise before chunking.
    """
    if not text:
        return ""
    # Strip markdown hyperlinks, keep only the label text
    text = re.sub(r"\[([^\]]*)\]\(https?://[^\)]+\)", r"\1", text)

    # Strip bare URLs that appear mid-text
    text = re.sub(r"https?://\S+", "", text)

    # Strip lines that are at least 40% punctuation/special chars (nav/breadcrumb lines)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append(line)
            continue
        special_ratio = sum(1 for c in stripped if not c.isalnum() and c != " ") / len(stripped)
        if special_ratio < 0.4:
            clean_lines.append(line)
    text = "\n".join(clean_lines)

    cleaned = text

    # Strip markdown image syntax: ![alt](url)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)

    # Strip obvious navigation/UI boilerplate lines.
    nav_line = re.compile(
        r"^\s*(home|menu|navigation|nav|login|log in|sign in|sign up|subscribe|privacy|terms|"
        r"cookie|cookies|back|next|previous|share|follow|search)\s*$",
        re.IGNORECASE,
    )
    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop very short UI fragments/noise lines.
        if len(stripped) < 25:
            if nav_line.match(stripped):
                continue
            if re.fullmatch(r"[\W\d_]+", stripped):
                continue
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prepare_source_chunks(extracted_pairs: list[tuple[dict, str]]) -> tuple[list[dict], list[dict]]:
    """Clean source text, chunk it, and assign source-local and job-global IDs.

    Full nonempty Tavily text takes precedence over the truncated extraction
    fallback, matching the evaluated encoder policy.
    """
    items: list[dict] = []
    groups: list[dict] = []
    for result, extracted in extracted_pairs:
        raw = result.get("raw_content") or ""
        text = clean_text_for_chunking(raw if raw.strip() else extracted)
        pieces = chunk_text(text)
        if not pieces:
            continue
        group = {"url": result.get("url") or "", "chunk_count": len(pieces), "chunks": []}
        for source_index, piece in enumerate(pieces):
            index = len(items)
            group["chunks"].append({
                "index": source_index, "chunk_index": index,
                "length": len(piece), "text": piece,
            })
            items.append({
                "text": piece, "url": result.get("url"), "title": result.get("title"),
                "chunk_index": index, "source_chunk_index": source_index,
            })
        groups.append(group)
    if not items:
        raise RuntimeError("Search sources contained no usable research text")
    return items, groups
