# backend/scraper.py — web text extraction (fallback when Tavily raw_content is empty)
import asyncio
import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("research_agent")


async def extract(url: str) -> str:
    """GET URL (10s timeout, follow redirects), strip noisy tags, truncate to 4000 chars."""
    if not url or not str(url).strip():
        return ""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            html = r.text
    except Exception as e:
        logger.debug("scraper extract failed %s: %s", url, e)
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000] if text else ""
    except Exception as e:
        logger.debug("scraper parse failed %s: %s", url, e)
        return ""


async def text_for_result(result: dict) -> str:
    """Prefer Tavily ``raw_content``; if empty, scrape ``url``."""
    raw = (result.get("raw_content") or "").strip()
    if raw:
        return raw[:4000]
    return await extract(result.get("url") or "")


async def extract_parallel(
    results: list[dict], *, concurrency: int = 5
) -> list[tuple[dict, str]]:
    """
    For each search row, resolve body text. Uses ``asyncio.gather`` with a
    ``Semaphore`` so at most ``concurrency`` HTTP fetches run at once.
    Order matches ``results``.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(r: dict) -> tuple[dict, str]:
        async with sem:
            try:
                t = await text_for_result(r)
            except Exception as e:
                logger.debug("extract_parallel row failed: %s", e)
                t = ""
        return (r, t)

    return list(await asyncio.gather(*(_one(r) for r in results)))
