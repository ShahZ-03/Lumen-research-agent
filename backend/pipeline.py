"""One research pipeline shared by FastAPI jobs and the encoder evaluation CLI."""
from __future__ import annotations

import asyncio
from time import perf_counter

from . import agent, chunker, retrieval, scraper, searcher, services, synthesizer

PIPELINE_VERSION = "encoder_shared_v1"


async def run_pipeline(topic: str, job_id: str, *, stream_queue=None, progress=None,
                       conversation_context: list[dict] | None = None) -> dict:
    """Run once; save partial diagnostics and propagate critical stage failures.

    ``progress(step, timings)`` belongs to the adapter, so CLI jobs need not be
    registered in application history to execute the same algorithm.
    """
    if not topic.strip():
        raise ValueError("A nonempty research topic is required")
    started = perf_counter()
    snapshot = services.init_pipeline_diagnostics(topic, job_id, version=PIPELINE_VERSION)
    timings = snapshot["timings"]
    stage = "starting"
    stage_started = perf_counter()

    async def advance(next_stage: str):
        nonlocal stage, stage_started
        now = perf_counter()
        if stage != "starting":
            timings[f"{stage}_seconds"] = round(now - stage_started, 3)
        stage = next_stage
        stage_started = now
        snapshot["step"] = stage
        if progress:
            await progress(stage, dict(timings))

    try:
        await advance("decomposing")
        history = conversation_context or []
        research_topic = await agent.resolve_followup_topic(topic, history) if history else topic
        snapshot.update(conversation_context=history, research_topic=research_topic)
        questions, strategy = await agent.plan_research(research_topic)
        snapshot.update(sub_questions=questions, sub_question_strategy=strategy)

        await advance("searching")
        results = await searcher.search_sources(
            questions, max_results=agent.MAX_RESULTS_PER_QUERY,
            query_results=snapshot["search_results"],
        )
        await advance("extracting")
        pairs = await scraper.extract_parallel(results, concurrency=agent.SCRAPE_CONCURRENCY)

        items, groups = await asyncio.to_thread(chunker.prepare_source_chunks, pairs)
        snapshot["chunks_by_source"] = groups
        snapshot["chunk_count"] = len(items)

        await advance("embedding")
        await services.index_job_chunks(job_id, items)

        await advance("retrieving")
        corpus, job_metas = services.get_job_corpus(job_id)
        selected = await retrieval.select_evidence(
            research_topic, job_id, services.vec, corpus, job_metas, snapshot=snapshot, progress=advance,
        )
        if not selected:
            raise RuntimeError("No evidence remained after reranking")

        await advance("synthesizing")
        report, domains, grounding = await synthesizer.generate(
            topic, job_id, services.vec, stream_queue=stream_queue,
            selected_chunks=selected, sub_questions=questions, snapshot=snapshot,
            progress=advance, conversation_context=history,
        )
        snapshot.update(
            synthesis_output=report, synthesis_success=True,
            source_domains_count=domains, grounding=grounding.model_dump(), step="complete",
        )
        return snapshot
    except Exception as exc:
        safe_error = services.sanitize_error_message(exc)
        snapshot[f"error_{stage}"] = safe_error
        snapshot["error"] = safe_error
        snapshot["step"] = "error"
        raise
    finally:
        timings[f"{stage}_seconds"] = round(perf_counter() - stage_started, 3)
        timings["total_seconds"] = round(perf_counter() - started, 3)
        services.set_synthesis_debug(job_id, snapshot)
