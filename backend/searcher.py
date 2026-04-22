# backend/searcher.py — Tavily API (sequential queries, rate-limit friendly)
import asyncio
import logging
import os

import httpx

from .retry import async_retry

logger = logging.getLogger("research_agent")

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"
SEARCH_DELAY_SEC = float(os.getenv("TAVILY_SEARCH_DELAY", "0.35"))


def _normalize_result(item: dict) -> dict:
    return {
        "url": item.get("url") or "",
        "title": item.get("title") or "",
        "raw_content": item.get("raw_content") or item.get("content") or "",
        "snippet": item.get("content") or item.get("snippet") or "",
    }


async def search(query: str, max_results: int = 5) -> list[dict]:
    """POST to Tavily; return ``[{url, title, raw_content, snippet}, ...]``."""
    if not TAVILY_KEY:
        logger.warning("TAVILY_API_KEY not set; skipping search")
        return []
    payload = {
        "api_key": TAVILY_KEY,
        "query": query,
        "max_results": max_results,
        "include_raw_content": True,
    }

    async def _do():
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(TAVILY_URL, json=payload)
            r.raise_for_status()
            return r.json()

    try:
        data = await async_retry(_do, label="Tavily search")
    except Exception as e:
        logger.exception("Tavily search failed for query %r: %s", query, e)
        return []

    raw_list = data.get("results") or []
    return [_normalize_result(x) for x in raw_list if isinstance(x, dict)]


async def search_queries_sequential(
    queries: list[str], max_results: int = 5
) -> list[dict]:
    """Run each query one after another with a short delay (no asyncio.gather)."""
    merged: list[dict] = []
    seen_urls: set[str] = set()
    for i, q in enumerate(queries):
        if i > 0:
            await asyncio.sleep(SEARCH_DELAY_SEC)
        try:
            batch = await search(q.strip(), max_results=max_results)
        except Exception as e:
            logger.exception("search_queries_sequential: %s", e)
            continue
        for row in batch:
            u = row.get("url") or ""
            if u and u in seen_urls:
                continue
            if u:
                seen_urls.add(u)
            merged.append(row)
    return merged


if __name__ == "__main__":
    async def _demo():
        logging.basicConfig(level=logging.INFO)
        q = "Python asyncio best practices"
        out = await search(q, max_results=3)
        print("results:", len(out))
        if out:
            print("first keys:", out[0].keys())

    asyncio.run(_demo())
