from __future__ import annotations

from .groq_client import groq_generate, DEFAULT_MODEL as DEFAULT_GROQ_MODEL
from .gemini_client import gemini_generate, DEFAULT_GEMINI_MODEL


def is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return (getattr(response, "status_code", None) == 429
            or "RATE_LIMIT" in str(exc) or "429" in str(exc))


async def generate_with_details(prompt: str, *, primary: str = "groq", max_tokens: int = 1200,
                                temperature: float = 0.2):
    """Provider clients own retries; switch once on an exhausted rate limit."""
    if primary == "gemini":
        text = await gemini_generate(prompt, max_tokens=max_tokens)
        return text, "gemini", DEFAULT_GEMINI_MODEL, False
    try:
        text = await groq_generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return text, "groq", DEFAULT_GROQ_MODEL, False
    except Exception as exc:
        if not is_rate_limit_error(exc):
            raise
        text = await gemini_generate(prompt, max_tokens=max_tokens)
        return text, "gemini", DEFAULT_GEMINI_MODEL, True


async def generate_with_fallback(prompt: str, *, primary: str = "groq", max_tokens: int = 1024,
                                 temperature: float = 0.2) -> str:
    text, _, _, _ = await generate_with_details(
        prompt, primary=primary, max_tokens=max_tokens, temperature=temperature,
    )
    return text
