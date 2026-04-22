# backend/agent.py — research loop (sequential search; parallel capped scraping)
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from time import perf_counter

from . import chunker, searcher, scraper, scorer, synthesizer
from .llm_router import llm_generate
from .hybrid_retrieval import hybrid_query

logger = logging.getLogger("research_agent")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))
MAX_CHUNKS_FOR_SCORING = 30
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


async def run_research_loop(topic: str, job_id: str) -> None:
    from . import services

    timings: dict[str, float] = {}
    t_job = perf_counter()
    job_urls_seen: set[str] = set()

    stream_q = services.ensure_report_queue(job_id)

    await services.update_research_status(
        job_id,
        status="processing",
        step="decomposing",
        iteration=0,
        report=None,
        error=None,
        timings={},
        source_domains_count=None,
    )

    try:
        t0 = perf_counter()
        sub_questions = await decompose_topic(topic)
        timings["decompose_seconds"] = round(perf_counter() - t0, 3)

        for round_idx in range(MAX_ITERATIONS):
            await services.update_research_status(
                job_id, step="searching", iteration=round_idx, timings=dict(timings)
            )
            t0 = perf_counter()
            results = await searcher.search_queries_sequential(
                sub_questions, max_results=MAX_RESULTS_PER_QUERY
            )
            timings[f"search_round_{round_idx}_seconds"] = round(perf_counter() - t0, 3)

            await services.update_research_status(
                job_id, step="extracting", iteration=round_idx, timings=dict(timings)
            )

            to_fetch: list[dict] = []
            for r in results:
                u = (r.get("url") or "").strip()
                if u:
                    key = u.lower()
                    if key in job_urls_seen:
                        continue
                    job_urls_seen.add(key)
                to_fetch.append(r)

            paired = await scraper.extract_parallel(
                to_fetch, concurrency=SCRAPE_CONCURRENCY
            )

            chunk_items: list[dict] = []
            for r, full in paired:
                if not full.strip():
                    continue
                pieces = chunker.chunk_text(full)
                url = r.get("url")
                title = r.get("title")
                for ch in pieces:
                    chunk_items.append({"text": ch, "url": url, "title": title})

            if len(chunk_items) > MAX_CHUNKS_FOR_SCORING:
                chunk_items = chunk_items[:MAX_CHUNKS_FOR_SCORING]

            await services.update_research_status(
                job_id, step="scoring", iteration=round_idx, timings=dict(timings)
            )
            t0 = perf_counter()
            try:
                scored = await scorer.score_chunks(chunk_items, topic)
            except Exception as e:
                logger.exception("scoring: %s", e)
                scored = []
            timings[f"score_round_{round_idx}_seconds"] = round(perf_counter() - t0, 3)

            await services.update_research_status(
                job_id, step="embedding", iteration=round_idx, timings=dict(timings)
            )
            texts = [s.text for s in scored]
            base = services.job_chunk_count(job_id)
            metadatas = [
                {
                    "job_id": job_id,
                    "url": s.url,
                    "title": s.title,
                    "chunk_index": base + i,
                }
                for i, s in enumerate(scored)
            ]
            t0 = perf_counter()
            for bs in range(0, len(texts), VECTORIZATION_BATCH_SIZE):
                batch_t = texts[bs : bs + VECTORIZATION_BATCH_SIZE]
                batch_m = metadatas[bs : bs + VECTORIZATION_BATCH_SIZE]
                if not batch_t:
                    continue
                try:
                    await asyncio.to_thread(
                        services.vec.add_texts, batch_t, batch_m
                    )
                    services.append_job_chunks(job_id, batch_t, batch_m)
                except Exception as e:
                    logger.exception("embedding batch: %s", e)
            timings[f"embed_round_{round_idx}_seconds"] = round(perf_counter() - t0, 3)

            await services.update_research_status(
                job_id, step="evaluating", iteration=round_idx, timings=dict(timings)
            )
            try:
                sufficient, missing = await evaluate_sufficiency(
                    topic, job_id, services.vec
                )
            except Exception as e:
                logger.exception("evaluate_sufficiency: %s", e)
                sufficient, missing = True, []

            if sufficient:
                break
            if round_idx >= MAX_ITERATIONS - 1:
                break
            sub_questions = await followup_sub_questions(topic, missing)
            if not sub_questions:
                break

        await services.update_research_status(
            job_id, step="synthesizing", iteration=None, timings=dict(timings)
        )
        t0 = perf_counter()
        try:
            report, n_domains, grounding = await synthesizer.generate(
                topic,
                job_id,
                services.vec,
                stream_queue=stream_q,
            )
        except Exception as e:
            logger.exception("synthesize: %s", e)
            report = f"## Summary\nSynthesis failed: {e}\n"
            n_domains = 0
            from .models import GroundingResult

            grounding = GroundingResult(
                claims=[],
                verified_count=0,
                unverified_count=0,
                overall_score=1.0,
            )
        timings["synthesize_seconds"] = round(perf_counter() - t0, 3)
        timings["total_seconds"] = round(perf_counter() - t_job, 3)

        await services.finish_report_stream(job_id)

        await services.update_research_status(
            job_id,
            status="complete",
            step="complete",
            report=report,
            error=None,
            timings=dict(timings),
            source_domains_count=n_domains,
            grounding=grounding.model_dump(),
        )
    except Exception as e:
        logger.exception("run_research_loop failed: %s", e)
        timings["total_seconds"] = round(perf_counter() - t_job, 3)
        await services.finish_report_stream(job_id)
        await services.update_research_status(
            job_id,
            status="error",
            step="error",
            error=str(e),
            timings=dict(timings),
        )
