# backend/grounding.py — claim extraction + embedding similarity vs job chunks
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vecstore import VectorStore
import numpy as np

from .groq_client import groq_generate
from .models import GroundedClaim, GroundingResult

logger = logging.getLogger("research_agent")


def _default_threshold() -> float:
    return float(os.getenv("GROUNDING_THRESHOLD", "0.45"))


def _parse_claims_array(raw: str) -> list[str]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    if not isinstance(data, list):
        return []
    out = [str(x).strip() for x in data if x is not None and str(x).strip()]
    return out[:20]


def _empty_grounding_ok() -> GroundingResult:
    return GroundingResult(
        claims=[],
        verified_count=0,
        unverified_count=0,
        overall_score=1.0,
    )


async def extract_claims(report: str) -> list[str]:
    prompt = f"""You are a fact extraction assistant.
Extract every distinct factual claim from the following research report.
A claim is a specific assertion that could be verified against a source.
Ignore headings, meta-commentary, and the Sources section.

Respond ONLY with a JSON array of strings, no preamble, no markdown fences.
Example: ["Claim one here.", "Claim two here."]

Report:
{report}
"""
    try:
        raw = await groq_generate(prompt, max_tokens=1024, temperature=0.1)
    except Exception as e:
        logger.exception("extract_claims: %s", e)
        return []
    return _parse_claims_array(raw)


def _encode_normalized(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    return embs / norms


async def ground_report(
    report: str,
    job_id: str,
    threshold: float | None = None,
    *,
    vec: "VectorStore | None" = None,
) -> GroundingResult:
    """
    Compare extracted claims to job source chunks via cosine similarity.
    Uses ``vec.model`` (SentenceTransformer) — pass the same ``VectorStore`` as the
    pipeline (e.g. ``services.vec``) so no second model is loaded.
    """
    from . import services

    th = _default_threshold() if threshold is None else threshold
    store = vec if vec is not None else services.vec
    model = store.model

    claims = await extract_claims(report)
    if not claims:
        return _empty_grounding_ok()

    corpus, metas = services.get_job_corpus(job_id)
    chunks: list[dict] = []
    for i, text in enumerate(corpus):
        if not (text or "").strip():
            continue
        md = metas[i] if i < len(metas) else {}
        chunks.append(
            {
                "text": text.strip(),
                "url": md.get("url"),
                "title": md.get("title"),
            }
        )

    chunk_texts = [c["text"] for c in chunks]
    if not chunk_texts:
        grounded = [
            GroundedClaim(
                claim=c,
                score=0.0,
                verified=False,
                best_source_url=None,
                best_source_title=None,
            )
            for c in claims
        ]
        n = len(grounded)
        return GroundingResult(
            claims=grounded,
            verified_count=0,
            unverified_count=n,
            overall_score=0.0,
        )

    def _compute() -> tuple[np.ndarray, np.ndarray]:
        ce = _encode_normalized(model, chunk_texts)
        qe = _encode_normalized(model, claims)
        return ce, qe

    chunk_embs, claim_embs = await asyncio.to_thread(_compute)

    grounded_claims: list[GroundedClaim] = []
    for i, claim in enumerate(claims):
        q = claim_embs[i : i + 1]
        scores = (q @ chunk_embs.T).flatten()
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best = chunks[best_idx]
        verified = best_score >= th
        grounded_claims.append(
            GroundedClaim(
                claim=claim,
                score=round(best_score, 3),
                verified=verified,
                best_source_url=best.get("url"),
                best_source_title=best.get("title"),
            )
        )

    vc = sum(1 for c in grounded_claims if c.verified)
    uc = len(grounded_claims) - vc
    n = len(grounded_claims)
    overall = round(vc / max(n, 1), 2)
    return GroundingResult(
        claims=grounded_claims,
        verified_count=vc,
        unverified_count=uc,
        overall_score=overall,
    )


if __name__ == "__main__":
    from unittest.mock import AsyncMock, patch

    async def _demo():
        logging.basicConfig(level=logging.INFO)
        from .vecstore import VectorStore

        v = VectorStore(path=os.getenv("VEC_PATH", "data/faiss_index"))
        r = "## Summary\nStub report.\n"
        fake_corpus = [
            "Python is a high-level programming language popular in data science.",
            "At standard atmospheric pressure, water boils at 100 degrees Celsius.",
        ]
        fake_meta = [{"url": "https://a.example", "title": "A"}, {"url": "https://b.example", "title": "B"}]

        with patch(
            "backend.grounding.extract_claims",
            new=AsyncMock(
                return_value=[
                    "Python is a language.",
                    "Mars is made of cheese.",
                ]
            ),
        ):
            with patch(
                "backend.services.get_job_corpus",
                lambda jid: (fake_corpus, fake_meta),
            ):
                out = await ground_report(r, "test-job", threshold=0.45, vec=v)
                print(out.model_dump())

    asyncio.run(_demo())
