from __future__ import annotations

from .groq_client import groq_generate
from .gemini_client import gemini_generate


async def generate_with_fallback(
    prompt: str,
    *,
    primary: str = "groq",
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Try primary model first, fallback to secondary on failure.
    """

    try:
        if primary == "groq":
            return await groq_generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            return await gemini_generate(prompt, max_tokens=max_tokens)

    except Exception as e:
        error_msg = str(e)

        # 🔥 Only fallback on rate limit (not all errors)
        if "RATE_LIMIT" in error_msg:
            print("[Fallback] Groq rate limited → switching to Gemini")

            return await gemini_generate(
                prompt,
                max_tokens=max_tokens,
            )

        # Re-raise other errors (important for debugging)
        raise