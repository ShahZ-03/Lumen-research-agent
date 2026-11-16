"""Encoder diagnostics using the same implementation as the live backend.

Change the responsible backend stage module to affect both entry points;
backend/pipeline.py only coordinates their execution.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

# Re-export historical helpers for evaluation scripts importing this module.
from backend.evidence import (
    source_key, cap_hits_per_source, is_duplicate,
    _duplicate_similarity, deduplicate_kept, deduplicate_rerank_candidates,
    build_dedup_stats, select_diverse_top_chunks,
)
from backend.chunker import clean_text_for_chunking
from backend.agent import ensure_multiple_sub_questions
from backend.reranker import (
    CROSS_ENCODER_MODEL, CROSS_ENCODER_BATCH_SIZE, CROSS_ENCODER_MAX_CHARS,
    _sigmoid, _cross_encoder_score_batch, score_chunks_with_cross_encoder,
)
from backend.llm_fallback import is_rate_limit_error


def slugify_topic(topic: str, max_len: int = 40) -> str:
    base = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", topic.strip().lower()))
    return (re.sub(r"_+", "_", base).strip("_") or "topic")[:max_len]


def estimate_tokens(text: str) -> int:
    return int(round(len(text) / 4)) if text else 0


async def run_debug(topic: str) -> dict:
    from backend import services
    from backend.pipeline import run_pipeline

    job_id = f"debug-{uuid4().hex[:10]}"

    async def progress(step, timings):
        print(f"[{step}]", flush=True)

    try:
        return await run_pipeline(topic, job_id, progress=progress)
    except Exception:
        return services.get_synthesis_debug(job_id) or {
            "job_id": job_id, "topic": topic, "synthesis_success": False,
            "error": "Pipeline initialization failed",
        }
    finally:
        services.JOB_CORPUS.pop(job_id, None)
        services.JOB_CHUNK_META.pop(job_id, None)
        services.SYNTHESIS_DEBUG.pop(job_id, None)


async def searcher_scrape(results: list[dict]):
    from backend import agent, scraper
    return await scraper.extract_parallel(results, concurrency=agent.SCRAPE_CONCURRENCY)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python debug_runner_clean_encoder.py "your test topic here"')
        raise SystemExit(1)
    topic = " ".join(sys.argv[1:]).strip()
    snapshot = asyncio.run(run_debug(topic))
    out_dir = Path("debug_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    out_path = out_dir / f"{slugify_topic(topic)}_encoder_{stamp}.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {out_path.as_posix()}")
    if not snapshot.get("synthesis_success"):
        print("Pipeline failed; see the snapshot error and stage.")
        raise SystemExit(1)
    print(f"Synthesis succeeded via {snapshot['synthesis_provider']}")


if __name__ == "__main__":
    main()
