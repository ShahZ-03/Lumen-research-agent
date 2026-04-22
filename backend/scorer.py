# backend/scorer.py — Groq relevance scoring (batches of up to 10)
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any

from .groq_client import groq_generate
from .models import ScoredChunk

logger = logging.getLogger("research_agent")

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))
SCORE_BATCH_SIZE = 5
# Random sleep between Groq scoring batches (rate limits; seconds)
SCORE_BATCH_DELAY_MIN = float(os.getenv("SCORE_BATCH_DELAY_MIN", "3"))
SCORE_BATCH_DELAY_MAX = float(os.getenv("SCORE_BATCH_DELAY_MAX", "5"))


def _parse_score_response(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from model output; tolerate markdown fences."""
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
    return [x for x in data if isinstance(x, dict)]


async def score_chunks(
    items: list[dict], topic: str, threshold: float | None = None
) -> list[ScoredChunk]:
    """
    ``items`` are dicts with at least ``text``, and optionally ``url``, ``title``.
    Groq must return ONLY: [{"text": "...", "score": 0.85}, ...]
    """
    cutoff = threshold if threshold is not None else RELEVANCE_THRESHOLD
    out: list[ScoredChunk] = []
    if not items:
        return out

    for batch_start in range(0, len(items), SCORE_BATCH_SIZE):
        if batch_start > 0:
            delay = random.uniform(SCORE_BATCH_DELAY_MIN, SCORE_BATCH_DELAY_MAX)
            await asyncio.sleep(delay)
        batch = items[batch_start : batch_start + SCORE_BATCH_SIZE]
        texts_only = [str(b.get("text") or "")[:1500] for b in batch]
        prompt = f"""You are a relevance judge. Given the research topic and text chunks, score each chunk's relevance to the topic from 0.0 to 1.0.

Topic: {topic}

Respond ONLY with a JSON array (no other text), same order as the chunks:
[{{"text": <exact chunk string as given>, "score": <float>}}, ...]

Chunks (JSON array of strings):
{json.dumps(texts_only)}
"""
        try:
            raw = await groq_generate(prompt, max_tokens=2048, temperature=0.1)
        except Exception as e:
            logger.exception("Groq scoring batch failed: %s", e)
            for b in batch:
                out.append(
                    ScoredChunk(
                        text=str(b.get("text") or ""),
                        score=0.0,
                        url=b.get("url"),
                        title=b.get("title"),
                    )
                )
            continue

        parsed = _parse_score_response(raw)
        score_by_text: dict[str, float] = {}
        for row in parsed:
            t = str(row.get("text") or "")
            try:
                s = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            score_by_text[t.strip()] = max(0.0, min(1.0, s))

        for b in batch:
            t = str(b.get("text") or "")
            s = score_by_text.get(t.strip())
            if s is None:
                for key, val in score_by_text.items():
                    if key and (key in t or t in key):
                        s = val
                        break
            if s is None:
                s = 0.0
            if s >= cutoff:
                out.append(
                    ScoredChunk(
                        text=t,
                        score=s,
                        url=b.get("url"),
                        title=b.get("title"),
                    )
                )

    return out


if __name__ == "__main__":
    import asyncio

    async def _demo():
        logging.basicConfig(level=logging.INFO)
        chunks = [
            {"text": "Python asyncio runs coroutines on an event loop.", "url": None, "title": None},
            {"text": "Banana bread recipes use ripe bananas.", "url": None, "title": None},
        ]
        scored = await score_chunks(chunks, topic="asyncio in Python", threshold=0.3)
        for s in scored:
            print(s.model_dump())

    asyncio.run(_demo())
