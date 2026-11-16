"""The shared retrieve, rerank, deduplicate, and select evidence boundary."""
from __future__ import annotations

import asyncio
import os

from . import evidence, reranker
from .hybrid_retrieval import hybrid_query

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "40"))
MAX_SYNTHESIS_CHUNKS = int(os.getenv("MAX_SYNTHESIS_CHUNKS", "8"))


async def select_evidence(topic: str, job_id: str, vec, corpus: list[str], metas: list[dict], *, snapshot: dict, progress=None):
    if RETRIEVAL_TOP_K < 1 or MAX_SYNTHESIS_CHUNKS < 1:
        raise ValueError("Retrieval and synthesis chunk counts must be positive")
    hits = await asyncio.to_thread(
        hybrid_query, job_id, topic, vec, corpus, metas,
        top_k=RETRIEVAL_TOP_K, vec_pool=RETRIEVAL_TOP_K, bm25_pool=RETRIEVAL_TOP_K,
    )
    snapshot["retrieved_chunks"] = [
        dict(hit, rank=i + 1, chunk_index=(hit.get("metadata") or {}).get("chunk_index"),
             url=(hit.get("metadata") or {}).get("url") or hit.get("url"),
             title=(hit.get("metadata") or {}).get("title") or hit.get("title"),
             snippet=str(hit.get("text") or "")[:280])
        for i, hit in enumerate(hits)
    ]
    snapshot["retrieved_count"] = len(hits)
    counts: dict[str, int] = {}
    for meta in metas:
        key = evidence.source_key(meta.get("url") or meta.get("title"))
        counts[key] = counts.get(key, 0) + 1
    hits = evidence.cap_hits_per_source(hits, top_k=RETRIEVAL_TOP_K, source_chunk_counts=counts)
    items = []
    for hit in hits:
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        meta = hit.get("metadata") or {}
        items.append({
            "text": text, "url": meta.get("url") or hit.get("url"),
            "title": meta.get("title") or hit.get("title"),
            "chunk_index": meta.get("chunk_index"),
            "source_chunk_index": meta.get("source_chunk_index"),
        })
    if not items:
        raise RuntimeError("No usable evidence was retrieved for this topic")
    if progress:
        await progress("reranking")
    scored, model, latency = await reranker.score_chunks_with_cross_encoder(items, topic)
    if len(scored) != len(items):
        raise RuntimeError("Cross-encoder scores do not align with retrieval candidates")
    ranked = sorted(zip(items, scored), key=lambda pair: pair[1].score, reverse=True)
    deduped, duplicates = evidence.deduplicate_rerank_candidates(ranked)
    kept = evidence.select_diverse_top_chunks(deduped, target_n=MAX_SYNTHESIS_CHUNKS)
    kept_ids = {id(item) for item, _ in kept}
    snapshot.update({
        "rerank_method": "cross_encoder", "cross_encoder_model": model,
        "scoring_provider": model, "rerank_latency_ms": latency,
        "dedup_stats": evidence.build_dedup_stats(ranked, deduped, duplicates),
        "kept_count": len(kept), "dropped_count": len(scored) - len(kept),
        "scored_chunks": [dict(item, score=float(score.score), kept=id(item) in kept_ids)
                          for item, score in ranked],
        "chunks_kept_after_rerank": [
            dict(item, rank=i + 1, score=float(score.score), bge_score=float(score.score))
            for i, (item, score) in enumerate(kept)
        ],
    })
    return [dict(item, source=item.get("url") or item.get("title") or "Unknown") for item, _ in kept]
