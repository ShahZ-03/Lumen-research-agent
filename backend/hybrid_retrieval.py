# backend/hybrid_retrieval.py — BM25 + dense vectors merged with RRF
from __future__ import annotations

import logging
import re
from rank_bm25 import BM25Okapi

from .vecstore import VectorStore

logger = logging.getLogger("research_agent")

_RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ids in rankings:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


def hybrid_query(
    job_id: str,
    query_text: str,
    vec: VectorStore,
    job_corpus: list[str],
    job_metas: list[dict],
    *,
    top_k: int = 20,
    vec_pool: int = 40,
    bm25_pool: int = 40,
) -> list[dict]:
    """
    Combine BM25 and FAISS rankings with reciprocal rank fusion.
    ``job_corpus[i]`` aligns with ``job_metas[i]`` and metadata ``chunk_index`` == i.
    """
    if not job_corpus or len(job_corpus) != len(job_metas):
        return vec.query(query_text, top_k=top_k, job_id=job_id)

    tokenized = [tokenize(t) for t in job_corpus]
    if not any(tokenized):
        return vec.query(query_text, top_k=top_k, job_id=job_id)

    try:
        bm25 = BM25Okapi(tokenized)
        q_tokens = tokenize(query_text)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_order = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:bm25_pool]
    except Exception as e:
        logger.warning("BM25 failed, falling back to vector only: %s", e)
        return vec.query(query_text, top_k=top_k, job_id=job_id)

    v_hits = vec.query(query_text, top_k=vec_pool, job_id=job_id)
    vec_order: list[int] = []
    seen_v: set[int] = set()
    for h in v_hits:
        md = h.get("metadata") or {}
        ci = md.get("chunk_index")
        if ci is None:
            continue
        try:
            idx = int(ci)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(job_corpus) and idx not in seen_v:
            seen_v.add(idx)
            vec_order.append(idx)

    fused = reciprocal_rank_fusion([vec_order, bm25_order], k=_RRF_K)

    out: list[dict] = []
    for idx in fused:
        if len(out) >= top_k:
            break
        if 0 <= idx < len(job_corpus):
            meta = dict(job_metas[idx])
            meta.setdefault("job_id", job_id)
            meta.setdefault("chunk_index", idx)
            out.append({"text": job_corpus[idx], "metadata": meta})
    return out
