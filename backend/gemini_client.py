from __future__ import annotations

import os
import httpx

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def gemini_generate(
    prompt: str,
    max_tokens: int = 512,
    model: str | None = None,
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    selected_model = model or DEFAULT_GEMINI_MODEL

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2,
        },
    }

    # Use header-based auth so exceptions do not include the API key in request URLs.
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(_gemini_url(selected_model), headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return str(data)