# backend/groq_client.py
from __future__ import annotations

import asyncio
import json
import os

import httpx

from .retry import async_retry

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Limit concurrent Groq requests
GROQ_SEMAPHORE = asyncio.Semaphore(2)

# 🔥 Reusable HTTP client (major performance improvement)
CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(120.0, read=None)
)


async def groq_generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """Standard non-streaming generation via Groq."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    url = f"{GROQ_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async def _post():
        async with GROQ_SEMAPHORE:
            try:
                response = await CLIENT.post(url, headers=headers, json=payload)

                if response.status_code == 429:
                    raise RuntimeError("RATE_LIMIT")

                response.raise_for_status()
                return response.json()

            except httpx.RequestError as e:
                raise RuntimeError(f"Network error calling Groq: {e}") from e

    data = await async_retry(_post)

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Groq response format: {data}")


async def groq_stream_content(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    model: str | None = None,
):
    """Yield decoded assistant text deltas from Groq streaming API."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    url = f"{GROQ_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    backoffs = (1.0, 2.0, 4.0)
    last_exc: Exception | None = None

    for attempt in range(1 + len(backoffs)):
        if attempt > 0:
            await asyncio.sleep(backoffs[attempt - 1])

        try:
            async with GROQ_SEMAPHORE:
                async with CLIENT.stream(
                    "POST", url, headers=headers, json=payload
                ) as resp:

                    if resp.status_code == 429:
                        raise RuntimeError("RATE_LIMIT")

                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        payload_txt = line[5:].strip()

                        if payload_txt == "[DONE]":
                            return

                        try:
                            obj = json.loads(payload_txt)
                        except json.JSONDecodeError:
                            continue

                        try:
                            choice0 = (obj.get("choices") or [None])[0] or {}
                            delta = choice0.get("delta") or {}
                            content = delta.get("content") or ""
                        except (TypeError, IndexError):
                            content = ""

                        if content:
                            yield content

            return

        except Exception as e:
            last_exc = e

    raise last_exc or RuntimeError("Streaming failed with unknown error")


async def groq_stream(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    model: str | None = None,
):
    """
    Legacy raw stream (SSE lines).
    Prefer using `groq_stream_content`.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    url = f"{GROQ_API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with CLIENT.stream("POST", url, headers=headers, json=payload) as resp:
        resp.raise_for_status()

        async for line in resp.aiter_lines():
            if line:
                yield line