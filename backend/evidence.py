"""Evidence duplicate detection, source ordering, and final source selection."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse, urlunparse

from .models import ScoredChunk
# Compatibility for experiment scripts importing the old helper location.
from .chunker import clean_text_for_chunking


def source_key(value: str | None) -> str:
    """
    Normalize URL/title source identifiers so retrieval hits and corpus counts align.
    """
    raw = (value or "").strip()
    if not raw:
        return "unknown"
    if raw.startswith("http://") or raw.startswith("https://"):
        try:
            parsed = urlparse(raw)
            normalized_path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
            normalized = urlunparse(
                (
                    parsed.scheme.lower() or "https",
                    parsed.netloc.lower(),
                    normalized_path,
                    "",
                    "",
                    "",
                )
            )
            return normalized
        except Exception:
            return raw.lower()
    return raw.lower()




def cap_hits_per_source(
    hits: list[dict],
    *,
    top_k: int,
    source_chunk_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    Reorder candidates by source allowance, then backfill to preserve the pool.
    This is a soft ordering policy, not a hard cap after backfill.
    """
    if not hits:
        return []
    out: list[dict] = []
    source_counts: dict[str, int] = {}
    overflow: list[dict] = []

    for hit in hits:
        md = hit.get("metadata") or {}
        src = source_key(md.get("url") or hit.get("url") or md.get("title"))
        total_for_source = (source_chunk_counts or {}).get(src, 0)
        max_per_source = max(3, total_for_source // 10)
        n = source_counts.get(src, 0)
        if n < max_per_source and len(out) < top_k:
            out.append(hit)
            source_counts[src] = n + 1
        else:
            overflow.append(hit)

    # Backfill up to top_k if strict capping underfilled.
    for hit in overflow:
        if len(out) >= top_k:
            break
        out.append(hit)

    return out[:top_k]


def is_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    prefix_sim = SequenceMatcher(None, text_a[:300], text_b[:300]).ratio()
    if prefix_sim > threshold:
        return True
    mid_a = text_a[len(text_a) // 3 : len(text_a) // 3 + 200]
    mid_b = text_b[len(text_b) // 3 : len(text_b) // 3 + 200]
    mid_sim = SequenceMatcher(None, mid_a, mid_b).ratio()
    return mid_sim > threshold


def _duplicate_similarity(text_a: str, text_b: str) -> float:
    prefix_sim = SequenceMatcher(None, text_a[:300], text_b[:300]).ratio()
    mid_a = text_a[len(text_a) // 3 : len(text_a) // 3 + 200]
    mid_b = text_b[len(text_b) // 3 : len(text_b) // 3 + 200]
    mid_sim = SequenceMatcher(None, mid_a, mid_b).ratio()
    return max(prefix_sim, mid_sim)


def deduplicate_kept(
    chunks: list[dict],
) -> tuple[list[dict], list[tuple[dict, dict, float]]]:
    deduplicated: list[dict] = []
    flagged_pairs: list[tuple[dict, dict, float]] = []
    for candidate in chunks:
        matched_kept: dict | None = None
        matched_score = 0.0
        for kept in deduplicated:
            sim = _duplicate_similarity(candidate["text"], kept["text"])
            if is_duplicate(candidate["text"], kept["text"]):
                matched_kept = kept
                matched_score = sim
                break
        if matched_kept is not None:
            flagged_pairs.append((matched_kept, candidate, matched_score))
        else:
            deduplicated.append(candidate)
    return deduplicated, flagged_pairs


def deduplicate_rerank_candidates(
    ranked_pairs: list[tuple[dict, ScoredChunk]],
) -> tuple[list[tuple[dict, ScoredChunk]], list[tuple[dict, dict, float]]]:
    """
    Remove near-duplicate chunks before source-diversity selection.
    Preserves score order: first occurrence wins.
    """
    chunks: list[dict] = []
    pairs: list[tuple[dict, ScoredChunk]] = []
    for item, scored in ranked_pairs:
        text = str(item.get("text") or scored.text or "")
        if not text.strip():
            continue
        chunks.append({**item, "text": text})
        pairs.append((item, scored))

    kept_chunks, flagged_pairs = deduplicate_kept(chunks)
    kept_ids = {id(c) for c in kept_chunks}
    deduped_pairs = [pair for chunk, pair in zip(chunks, pairs) if id(chunk) in kept_ids]
    return deduped_pairs, flagged_pairs


def build_dedup_stats(
    candidates_before: list,
    deduped_pairs: list,
    flagged_pairs: list[tuple[dict, dict, float]],
) -> dict:
    return {
        "candidates_before_dedup": len(candidates_before),
        "dropped_as_duplicates": len(candidates_before) - len(deduped_pairs),
        "duplicate_pairs": [
            {
                "chunk_a": {"url": a.get("url"), "chunk_index": a.get("chunk_index")},
                "chunk_b": {"url": b.get("url"), "chunk_index": b.get("chunk_index")},
                "similarity": round(score, 3),
                "same_domain": urlparse(a.get("url") or "").netloc
                == urlparse(b.get("url") or "").netloc,
            }
            for a, b, score in flagged_pairs
        ],
    }


def select_diverse_top_chunks(
    ranked_pairs: list[tuple[dict, ScoredChunk]], target_n: int
) -> list[tuple[dict, ScoredChunk]]:
    """
    Select top chunks with proportional source diversity so large sources
    do not dominate all synthesis slots.
    """
    if not ranked_pairs or target_n <= 0:
        return []

    by_source: dict[str, list[tuple[dict, ScoredChunk]]] = {}
    for pair in ranked_pairs:
        item, _ = pair
        source_key = (item.get("url") or item.get("title") or "Unknown").strip() or "Unknown"
        by_source.setdefault(source_key, []).append(pair)

    total = len(ranked_pairs)
    source_items = list(by_source.items())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0

    # Initial proportional allocation using floor.
    for source, pairs in source_items:
        raw_quota = target_n * (len(pairs) / total)
        q = int(raw_quota)
        quotas[source] = q
        assigned += q
        remainders.append((raw_quota - q, source))

    # Give leftover slots to highest fractional remainders.
    leftovers = target_n - assigned
    for _, source in sorted(remainders, key=lambda x: x[0], reverse=True):
        if leftovers <= 0:
            break
        quotas[source] += 1
        leftovers -= 1

    selected: list[tuple[dict, ScoredChunk]] = []
    selected_ids: set[int] = set()
    for source, pairs in source_items:
        take_n = min(quotas.get(source, 0), len(pairs))
        for pair in pairs[:take_n]:
            selected.append(pair)
            selected_ids.add(id(pair))

    # If quotas underfill (due to small groups), fill by global best remaining.
    if len(selected) < target_n:
        for pair in ranked_pairs:
            if id(pair) in selected_ids:
                continue
            selected.append(pair)
            selected_ids.add(id(pair))
            if len(selected) >= target_n:
                break

    return selected[:target_n]
