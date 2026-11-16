"""Topic decomposition, question expansion, and legacy planning helpers."""
from __future__ import annotations

import json
import logging
import os
import re

from .llm_router import llm_generate
from .hybrid_retrieval import hybrid_query

logger = logging.getLogger("research_agent")

# Legacy experiment constants; the shared default pipeline is single-pass.
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "5"))
VECTORIZATION_BATCH_SIZE = int(os.getenv("VECTORIZATION_BATCH_SIZE", "32"))
SCRAPE_CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "5"))


def _extract_json_object(raw: str) -> dict | None:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


async def decompose_topic(topic: str) -> list[str]:
    prompt = f"""You are a research planner. Break the research topic into 3-5 specific sub-questions suitable for web search.

Topic: {topic}

Respond ONLY with JSON: {{"sub_questions": ["question1", "question2", ...]}}"""
    try:
        raw = await llm_generate(prompt, task_type="decompose")
    except Exception as e:
        logger.exception("decompose_topic: %s", e)
        return [topic]
    data = _extract_json_object(raw) or {}
    subs = data.get("sub_questions")
    if isinstance(subs, list) and subs:
        return [str(s).strip() for s in subs if str(s).strip()][:5]
    return [topic]


async def evaluate_sufficiency(topic: str, job_id: str, vec) -> tuple[bool, list[str]]:
    from . import services

    corpus, metas = services.get_job_corpus(job_id)
    hits = hybrid_query(
        job_id, topic, vec, corpus, metas, top_k=12
    )
    snippets = []
    for h in hits:
        t = (h.get("text") or "")[:1200]
        if t.strip():
            snippets.append(t)
    ctx = "\n\n".join(snippets) if snippets else "(no embedded context yet)"
    prompt = f"""You judge whether retrieved context is enough to write a comprehensive research report on the topic.

Topic: {topic}

Retrieved context (excerpts):
{ctx[:8000]}

Respond ONLY with JSON: {{"sufficient": true or false, "missing": ["gap or missing angle 1", ...]}}"""
    try:
        raw = await llm_generate(prompt, task_type="evaluate_sufficiency")
    except Exception as e:
        logger.exception("evaluate_sufficiency: %s", e)
        return True, []
    data = _extract_json_object(raw) or {}
    sufficient = bool(data.get("sufficient"))
    missing = data.get("missing")
    if not isinstance(missing, list):
        missing = []
    missing = [str(m).strip() for m in missing if str(m).strip()][:10]
    return sufficient, missing


async def followup_sub_questions(topic: str, missing: list[str]) -> list[str]:
    miss_txt = json.dumps(missing)
    prompt = f"""The research topic is: {topic}

Known gaps or missing angles: {miss_txt}

Propose 3-5 new web search sub-questions to fill these gaps.

Respond ONLY with JSON: {{"sub_questions": ["...", ...]}}"""
    try:
        raw = await llm_generate(prompt, task_type="followup_sub_questions")
    except Exception as e:
        logger.exception("followup_sub_questions: %s", e)
        return [topic]
    data = _extract_json_object(raw) or {}
    subs = data.get("sub_questions")
    if isinstance(subs, list) and subs:
        return [str(s).strip() for s in subs if str(s).strip()][:5]
    return [topic]


async def ensure_multiple_sub_questions(
    topic: str, sub_questions: list[str]
) -> tuple[list[str], str]:
    """
    Lower the threshold for multi-query search by expanding single-question plans.
    """
    unique = [q.strip() for q in sub_questions if q and q.strip()]
    unique = list(dict.fromkeys(unique))
    if len(unique) >= 3:
        return unique[:5], "decompose"

    # If decomposition collapsed to one question, expand angles via follow-up planner.
    try:
        expanded = await followup_sub_questions(
            topic,
            [
                "core concepts and definitions",
                "recent developments",
                "limitations, risks, and trade-offs",
            ],
        )
    except Exception:
        expanded = []

    merged = list(dict.fromkeys(unique + [q.strip() for q in expanded if q and q.strip()]))
    if len(merged) >= 2:
        return merged[:5], "followup"

    # Final heuristic fallback to force multi-query coverage.
    return (
        [
            topic,
            f"{topic} recent developments",
            f"{topic} risks and limitations",
        ][:5],
        "heuristic",
    )


async def plan_research(topic: str) -> tuple[list[str], str]:
    """Create and expand one search plan using the shared planning policy."""
    questions = await decompose_topic(topic)
    return await ensure_multiple_sub_questions(topic, questions)


async def resolve_followup_topic(question: str, messages: list[dict]) -> str:
    """Resolve references for discovery and retrieval using server-side history."""
    if not messages:
        return question
    prompt = f"""Rewrite the latest research question as one self-contained search topic.
Resolve pronouns and references using the conversation. Preserve the user's
intent, including when they change topics. Previous answers are unverified
conversation context, not factual evidence. Do not answer the question.

Conversation (JSON): {json.dumps(messages, ensure_ascii=False)}
Latest question: {question}

Respond ONLY with JSON: {{"research_topic": "self-contained question"}}"""
    try:
        response = await llm_generate(prompt, task_type="planning")
        resolved = (_extract_json_object(response) or {}).get("research_topic")
        if isinstance(resolved, str) and resolved.strip():
            return resolved.strip()[:4000]
    except Exception:
        logger.warning("Follow-up topic resolution failed; retaining explicit question context")
    questions = [m["content"] for m in messages if m.get("role") == "user"]
    return (f"Original topic: {questions[0][:800]}\nPrevious question: {questions[-1][:800]}\n"
            f"Follow-up: {question}") if questions else question


async def run_research_loop(topic: str, job_id: str) -> None:
    """Compatibility entry point; services owns the application job lifecycle."""
    from .services import run_research_task
    await run_research_task(topic, job_id)
