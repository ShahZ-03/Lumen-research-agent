# backend/retry.py — exponential backoff for async I/O
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger("research_agent")

# Wait 1s → 2s → 4s between attempts (four tries total)
DEFAULT_BACKOFF_SECS = (1.0, 2.0, 4.0)


async def async_retry(
    factory: Callable[[], Awaitable],
    *,
    label: str = "request",
):
    """
    Smarter retry:
    - Handles 429 with longer exponential backoff
    - Handles normal errors with shorter retry
    """

    backoffs_429 = [2, 5, 10, 20, 30]   # for rate limits
    backoffs_other = [1, 2, 4]          # for normal errors

    last_exc: BaseException | None = None

    for attempt in range(max(len(backoffs_429), len(backoffs_other))):
        try:
            return await factory()

        except Exception as e:
            last_exc = e
            err_str = str(e)

            is_429 = "429" in err_str or "RATE_LIMIT" in err_str

            if is_429:
                delay = backoffs_429[min(attempt, len(backoffs_429) - 1)]
                logger.warning(
                    "%s RATE LIMITED (attempt %s): retrying in %ss",
                    label,
                    attempt + 1,
                    delay,
                )
            else:
                delay = backoffs_other[min(attempt, len(backoffs_other) - 1)]
                logger.warning(
                    "%s failed (attempt %s): %s → retrying in %ss",
                    label,
                    attempt + 1,
                    e,
                    delay,
                )

            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
