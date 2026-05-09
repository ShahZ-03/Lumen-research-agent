# backend/synthesizer.py
from __future__ import annotations

import logging
import os
from typing import List, Dict, Any
from urllib.parse import urlparse

from .grounding import ground_report, _empty_grounding_ok
from .hybrid_retrieval import hybrid_query
from . import services
from .llm_fallback import generate_with_fallback
from .groq_client import groq_stream_content, DEFAULT_MODEL as DEFAULT_GROQ_MODEL
from .gemini_client import DEFAULT_GEMINI_MODEL, gemini_generate

logger = logging.getLogger("research_agent")
MAX_SYNTHESIS_CHUNKS = int(os.getenv("MAX_SYNTHESIS_CHUNKS", "8"))
SYNTHESIS_PROVIDER = os.getenv("SYNTHESIS_PROVIDER", "groq").strip().lower()


# =========================
# PROMPT BUILDER
# =========================
def select_synthesis_chunks(chunks: List[Dict]) -> List[Dict]:
    clean_chunks = []
    seen = set()

    for c in chunks:
        content = (
            (c.get("text") or c.get("content") or c.get("raw_content") or "")
            .strip()
        )

        source = (
            c.get("url")
            or c.get("source")
            or c.get("title")
            or "Unknown"
        )

        if not content:
            continue

        # Deduplicate by source when possible
        if source != "Unknown":
            if source in seen:
                continue
            seen.add(source)

        clean_chunks.append(
            {
                "text": content,
                "source": source,
                "url": c.get("url"),
                "title": c.get("title"),
            }
        )

    # Limit chunks to avoid large prompts (prevents 429)
    clean_chunks = clean_chunks[:MAX_SYNTHESIS_CHUNKS]
    logger.info("[Synthesizer] Using %s chunks", len(clean_chunks))
    return clean_chunks


def build_synthesis_prompt(selected_chunks: List[Dict]) -> str | None:
    clean_chunks = selected_chunks
    if not clean_chunks:
        return None

    context_blocks = [
        f"[Source {i+1}] ({c['source']})\n{c['text']}"
        for i, c in enumerate(clean_chunks)
    ]

    context_text = "\n\n".join(context_blocks)

    return f"""You are Lumen, a rigorous research synthesis engine.

====================
CORE DIRECTIVES (STRICT)
====================
- Ground every substantive claim in the provided context.
- Do not invent facts, sources, dates, statistics, or entities.
- Keep language precise, neutral, and globally understandable.
- Prefer concrete mechanisms, causal chains, assumptions, and limits.
- When evidence conflicts, explicitly note disagreement and confidence.
- If context is weak, say what is unknown instead of guessing.

====================
DOMAIN-AGNOSTIC INSTRUCTION
====================
The user query may be technical, business, policy, science, health, law, market, or mixed.
Adapt section naming and depth to fit the topic while preserving rigor.
Do not output private reasoning.

====================
OUTPUT FORMAT
====================
Use valid Markdown and produce this structure:

# [Concise Title]

## Executive Summary
- 3-6 bullets with the main answer and key takeaways.

## Key Findings
- Organize by the most important themes.
- Include evidence-backed details and trade-offs.

## Analysis
- Explain mechanisms, drivers, constraints, and implications.
- Include uncertainty and edge cases where relevant.

## Practical Guidance
- Provide concrete recommendations only when supported by evidence.
- If recommendations are not possible, state why.

## Limitations and Unknowns
- List missing evidence, weak coverage, and open questions.

## References
- Bullet list of unique sources used from the provided context.

====================
CONTEXT
====================
{context_text}
""".strip()


# =========================
# MAIN SYNTHESIS
# =========================
async def synthesize_report(
    prompt: str,
    provider: str,
    max_tokens: int = 1200,  # safer for rate limits
    *args: Any,
    **kwargs: Any,
) -> str:
    try:
        if provider == "gemini":
            return (await gemini_generate(prompt, max_tokens=max_tokens)).strip()
        result = await generate_with_fallback(
            prompt,
            *args,
            primary="groq",
            max_tokens=max_tokens,
            temperature=0.2,
            **kwargs,
        )
        return result.strip()

    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {str(e)}") from e


# =========================
# BACKWARD COMPAT WRAPPER
# =========================
async def generate(topic: str, job_id: str, vec, stream_queue=None):
    """
    Entry point used by agent pipeline.
    Retrieves chunks → synthesizes report → runs grounding.
    """

    try:
        corpus, metas = services.get_job_corpus(job_id)
        hits = hybrid_query(job_id, topic, vec, corpus, metas, top_k=10)

        chunks: list[dict] = []
        domains: set[str] = set()

        for hit in hits:
            text = (hit.get("text") or "").strip()[:600]  # reduce token size
            if not text:
                continue

            metadata = hit.get("metadata") or {}
            url = metadata.get("url") or hit.get("url")
            title = metadata.get("title") or hit.get("title")

            if isinstance(url, str) and url.strip():
                parsed = urlparse(url.strip())
                domain = parsed.netloc.split(":")[0].lower()
                if domain:
                    domains.add(domain)

            chunks.append({
                "text": text,
                "url": url,
                "title": title
            })

        selected_chunks = select_synthesis_chunks(chunks)
        prompt = build_synthesis_prompt(selected_chunks)
        provider = SYNTHESIS_PROVIDER if SYNTHESIS_PROVIDER in {"groq", "gemini"} else "groq"
        model_name = DEFAULT_GROQ_MODEL if provider == "groq" else DEFAULT_GEMINI_MODEL
        services.set_synthesis_debug(
            job_id,
            {
                "topic": topic,
                "provider": provider,
                "model": model_name,
                "retrieved_count": len(hits),
                "used_count": len(selected_chunks),
                "retrieved_chunks": [
                    {
                        "rank": i + 1,
                        "chunk_index": (h.get("metadata") or {}).get("chunk_index"),
                        "url": (h.get("metadata") or {}).get("url"),
                        "title": (h.get("metadata") or {}).get("title"),
                        "snippet": (h.get("text") or "")[:280],
                    }
                    for i, h in enumerate(hits)
                ],
                "used_chunks": [
                    {
                        "rank": i + 1,
                        "source": c.get("source"),
                        "url": c.get("url"),
                        "title": c.get("title"),
                        "snippet": (c.get("text") or "")[:400],
                    }
                    for i, c in enumerate(selected_chunks)
                ],
            },
        )

        if not prompt:
            report = "⚠️ No usable content found from retrieved sources. Try a more specific query."
            if stream_queue is not None:
                await stream_queue.put(report)
        else:
            report = ""
            streamed_ok = False
            # True live streaming is available only on Groq path.
            if stream_queue is not None and provider == "groq":
                try:
                    async for delta in groq_stream_content(
                        prompt,
                        max_tokens=1200,
                        temperature=0.2,
                    ):
                        report += delta
                        await stream_queue.put(delta)
                    streamed_ok = True
                except Exception as e:
                    logger.warning("groq token stream failed, using fallback: %s", e)

            if not streamed_ok:
                report = await synthesize_report(prompt, provider)
                if stream_queue is not None and report:
                    chunk_size = 96
                    for i in range(0, len(report), chunk_size):
                        await stream_queue.put(report[i : i + chunk_size])

        grounding = await ground_report(report, job_id, vec=vec)

        return report, len(domains), grounding

    except Exception as e:
        logger.exception("generate wrapper failed: %s", e)
        return f"Report generation failed: {e}", 0, _empty_grounding_ok()