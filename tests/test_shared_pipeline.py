"""Offline integration coverage: real stages with external/model boundaries faked."""
import asyncio
import random
import string
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend import agent, pipeline, reranker, searcher, services, synthesizer
from backend.models import GroundingResult, ScoredChunk


class MemoryVectors:
    def __init__(self):
        self.texts = []
        self.metas = []
        self.queries = []

    def add_texts(self, texts, metas):
        self.texts.extend(texts)
        self.metas.extend(metas)

    def query(self, topic, top_k=5, job_id=None):
        self.queries.append((topic, top_k, job_id))
        return [{"text": text, "metadata": meta}
                for text, meta in zip(self.texts, self.metas)
                if meta["job_id"] == job_id][:top_k]


def evidence_text(seed):
    rng = random.Random(seed)
    return "\n".join("evidence retrieval " + "".join(rng.choices(string.ascii_letters + " ", k=630)) + "."
                     for _ in range(18))


@pytest.fixture
def environment(monkeypatch, tmp_path):
    vectors = MemoryVectors()
    monkeypatch.setattr(services, "vec", vectors)
    monkeypatch.setattr(services, "redis_client", None)
    monkeypatch.setattr(services, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(services, "LEGACY_JOBS_INDEX_PATH", tmp_path / "no-index.json")
    monkeypatch.setattr(services, "LEGACY_DOCS_DIR", tmp_path / "no-docs")
    monkeypatch.setattr(services, "_db_initialized", False)
    monkeypatch.setattr(services, "_db_lock", asyncio.Lock())
    for name in ("JOB_CORPUS", "JOB_CHUNK_META", "SYNTHESIS_DEBUG", "report_stream_queues"):
        monkeypatch.setattr(services, name, {})
    monkeypatch.setattr(searcher, "SEARCH_DELAY_SEC", 0)
    monkeypatch.setattr(agent, "MAX_RESULTS_PER_QUERY", 5)
    monkeypatch.setattr(synthesizer, "SYNTHESIS_PROVIDER", "groq")
    questions = ["evidence retrieval concepts", "evidence retrieval developments", "evidence retrieval limitations"]
    monkeypatch.setattr(agent, "decompose_topic", AsyncMock(return_value=questions))
    bodies = [evidence_text(i) for i in range(3)]
    calls = []

    async def search(query, max_results=5):
        calls.append((query, max_results))
        index = questions.index(query)
        return [{"url": f"https://source{index}.example/paper", "title": f"Paper {index}",
                 "raw_content": bodies[index], "snippet": "evidence"}]

    monkeypatch.setattr(searcher, "search", search)
    batches = []

    async def score(items, topic):
        batches.append((items, topic))
        return [ScoredChunk(text=item["text"], score=1 - i / 100, url=item.get("url"), title=item.get("title"))
                for i, item in enumerate(items)], "fake-cross-encoder", {"total_ms": 1}

    monkeypatch.setattr(reranker, "score_chunks_with_cross_encoder", score)
    prompts = []

    async def generate(prompt, **kwargs):
        prompts.append(prompt)
        return "# Report\nEvidence is explained [Source 1].", "groq", "fake-model", False

    async def stream(prompt, **kwargs):
        prompts.append(prompt)
        yield "# Report\n"
        yield "Evidence is explained [Source 1]."

    monkeypatch.setattr(synthesizer, "generate_with_details", generate)
    monkeypatch.setattr(synthesizer, "groq_stream_content", stream)
    grounding = GroundingResult(claims=[], verified_count=0, unverified_count=0, overall_score=0)
    ground = AsyncMock(return_value=grounding)
    monkeypatch.setattr(synthesizer, "ground_report", ground)
    return SimpleNamespace(vectors=vectors, calls=calls, batches=batches, prompts=prompts,
                           ground=ground, questions=questions)


def test_pipeline_indexes_all_chunks_and_reranks_once(environment):
    result = asyncio.run(pipeline.run_pipeline("evidence retrieval", "test-job"))
    assert result["chunk_count"] > 30  # The old app silently dropped everything after 30.
    assert len(environment.vectors.texts) == result["chunk_count"]
    assert len(environment.calls) == 3  # No extra search for diagnostics.
    assert len(environment.batches) == 1
    assert len(environment.batches[0][0]) == 40
    assert result["retrieved_count"] == 40
    assert result["kept_count"] == 8
    assert len(environment.vectors.queries) == 1  # Synthesis never retrieves again.
    assert len(result["used_chunks"]) == 8
    assert result["source_domains_count"] >= 2
    assert all(c["text"] in result["synthesis_prompt"] for c in result["used_chunks"])
    assert "Topic: evidence retrieval" in result["synthesis_prompt"]
    assert all(q in result["synthesis_prompt"] for q in environment.questions)
    assert result["synthesis_success"]
    assert result["grounding"] is not None
    assert environment.ground.await_count == 1
    corpus, metas = services.get_job_corpus("test-job")
    assert [m["chunk_index"] for m in metas] == list(range(len(corpus)))
    assert all("source_chunk_index" in m for m in metas)
    assert all("retrieval" in hit for hit in result["retrieved_chunks"])


def test_api_job_uses_pipeline_and_closes_stream(environment):
    async def run():
        await services.init_research_job("web-job", "evidence retrieval")
        await services.run_research_task("evidence retrieval", "web-job")
        status = await services.get_research_status("web-job")
        parts = []
        queue = services.ensure_report_queue("web-job")
        while not queue.empty():
            parts.append(queue.get_nowait())
        return status, parts

    status, parts = asyncio.run(run())
    debug = services.get_synthesis_debug("web-job")
    assert status["status"] == "complete"
    assert status["iteration"] is None
    assert parts[-1] is None
    assert "".join(parts[:-1]) == status["report"]
    assert status["report"] == debug["synthesis_output"]
    assert debug["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert len(environment.batches[0][0]) == 40
    assert debug["kept_count"] == 8
    assert status["source_domains_count"] == debug["source_domains_count"]


def test_cli_uses_same_selection_and_releases_memory(environment):
    from debug_runner_clean_encoder import run_debug
    result = asyncio.run(run_debug("evidence retrieval"))
    assert result["pipeline_version"] == pipeline.PIPELINE_VERSION
    assert result["kept_count"] == 8
    assert result["synthesis_success"]
    assert len(environment.calls) == 3
    assert result["job_id"] not in services.JOB_CORPUS
    assert result["job_id"] not in services.SYNTHESIS_DEBUG


def test_failed_embedding_never_commits_lexical_corpus(environment, monkeypatch):
    def fail(*args):
        raise RuntimeError("disk full")
    monkeypatch.setattr(environment.vectors, "add_texts", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(pipeline.run_pipeline("evidence retrieval", "failed-job"))
    assert services.get_job_corpus("failed-job") == ([], [])
    assert "error_embedding" in services.get_synthesis_debug("failed-job")
    assert not environment.batches
    assert not environment.prompts


def test_web_and_cli_select_identical_evidence_for_identical_inputs(environment):
    from debug_runner_clean_encoder import run_debug

    async def run():
        cli = await run_debug("evidence retrieval")
        await services.init_research_job("web-job", "evidence retrieval")
        await agent.run_research_loop("evidence retrieval", "web-job")
        return cli, services.get_synthesis_debug("web-job")

    cli, web = asyncio.run(run())
    assert cli["chunks_kept_after_rerank"] == web["chunks_kept_after_rerank"]
    assert cli["synthesis_prompt"] == web["synthesis_prompt"]
    assert cli["dedup_stats"] == web["dedup_stats"]
    assert cli["config"] == web["config"]


@pytest.mark.parametrize("failure", ["search", "rerank", "synthesis"])
def test_failed_jobs_are_errors_and_terminate_stream(environment, monkeypatch, failure):
    if failure == "search":
        monkeypatch.setattr(searcher, "search", AsyncMock(return_value=[]))
    elif failure == "rerank":
        monkeypatch.setattr(reranker, "score_chunks_with_cross_encoder", AsyncMock(side_effect=RuntimeError("model failed")))
    else:
        async def fail(*args, **kwargs):
            raise RuntimeError("provider failed")
            yield  # This boundary is an async generator.
        monkeypatch.setattr(synthesizer, "groq_stream_content", fail)

    async def run():
        await services.init_research_job("failed-job", "evidence retrieval")
        await agent.run_research_loop("evidence retrieval", "failed-job")
        return await services.get_research_status("failed-job")

    status = asyncio.run(run())
    assert status["status"] == "error"
    assert status["report"] is None
    assert status["error"]
    assert services.report_stream_queues["failed-job"].get_nowait() is None
    assert not services.get_synthesis_debug("failed-job")["synthesis_success"]


def test_http_routes_keep_history_debug_and_completed_stream(environment):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        response = client.post("/research", json={"topic": "evidence retrieval"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        status = client.get(f"/status/{job_id}").json()
        assert status["status"] == "complete"
        debug = client.get(f"/debug/synthesis/{job_id}").json()
        assert debug["pipeline_version"] == pipeline.PIPELINE_VERSION
        assert debug["kept_count"] == 8
        assert client.get(f"/stream/report/{job_id}").text == status["report"]
        assert any(job["job_id"] == job_id for job in client.get("/jobs").json())
        assert client.delete(f"/jobs/{job_id}").status_code == 200
        assert client.get(f"/status/{job_id}").status_code == 404


def test_error_stream_opened_after_job_failure_does_not_wait(environment, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    monkeypatch.setattr(searcher, "search", AsyncMock(return_value=[]))
    with TestClient(app) as client:
        job_id = client.post("/research", json={"topic": "evidence retrieval"}).json()["job_id"]
        assert client.get(f"/status/{job_id}").json()["status"] == "error"
        assert client.get(f"/stream/report/{job_id}").text == ""


def test_cancelled_job_records_terminal_state_and_closes_stream(environment, monkeypatch):
    monkeypatch.setattr(pipeline, "run_pipeline", AsyncMock(side_effect=asyncio.CancelledError()))

    async def run():
        await services.init_research_job("cancelled", "topic")
        with pytest.raises(asyncio.CancelledError):
            await agent.run_research_loop("topic", "cancelled")
        return await services.get_research_status("cancelled")

    status = asyncio.run(run())
    assert status["status"] == "error"
    assert status["error"] == "Research was cancelled"
    assert services.report_stream_queues["cancelled"].get_nowait() is None


def test_followups_use_persisted_history_for_retrieval_and_generation(environment, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    resolved = "What are the limitations of hybrid evidence retrieval?"
    resolver = AsyncMock(return_value=resolved)
    monkeypatch.setattr(agent, "resolve_followup_topic", resolver)
    with TestClient(app) as client:
        first = client.post("/research", json={"topic": "evidence retrieval"}).json()
        second = client.post("/research", json={
            "topic": "What are its limitations?", "conversation_id": first["conversation_id"],
        }).json()
        assert second["conversation_id"] == first["conversation_id"]
        assert second["parent_job_id"] == first["job_id"]
        assert second["turn_index"] == 1
        assert environment.batches[1][1] == resolved
        assert resolver.await_args.args[0] == "What are its limitations?"
        context = resolver.await_args.args[1]
        assert [message["role"] for message in context] == ["user", "assistant"]
        assert context[0]["content"] == "evidence retrieval"
        assert "Evidence is explained" in context[1]["content"]
        assert "What are its limitations?" in environment.prompts[-1]
        assert "Conversation history" in environment.prompts[-1]

        # Rebuild transient state to verify that conversation continuity comes
        # from SQLite rather than a browser payload or in-memory message list.
        services.JOB_CORPUS.clear()
        services.JOB_CHUNK_META.clear()
        services.SYNTHESIS_DEBUG.clear()
        services._db_initialized = False
        history = client.get(f"/conversations/{first['conversation_id']}").json()
        assert [turn["topic"] for turn in history["turns"]] == ["evidence retrieval", "What are its limitations?"]
        assert all(turn["report"] for turn in history["turns"])
        third = client.post("/research", json={
            "topic": "Explain the second point", "conversation_id": first["conversation_id"],
        }).json()
        assert third["turn_index"] == 2
        assert third["parent_job_id"] == second["job_id"]
        assert len(resolver.await_args.args[1]) == 4


def test_new_research_does_not_inherit_another_conversation(environment, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    resolver = AsyncMock()
    monkeypatch.setattr(agent, "resolve_followup_topic", resolver)
    with TestClient(app) as client:
        first = client.post("/research", json={"topic": "first thread"}).json()
        second = client.post("/research", json={"topic": "second thread"}).json()
        assert first["conversation_id"] != second["conversation_id"]
        assert second["parent_job_id"] is None
        resolver.assert_not_awaited()
        assert len(client.get(f"/conversations/{second['conversation_id']}").json()["turns"]) == 1


def test_followup_rejects_unknown_or_busy_conversations(environment):
    from fastapi.testclient import TestClient
    from backend.main import app

    asyncio.run(services.init_research_job("busy", "still running"))
    with TestClient(app) as client:
        assert client.post("/research", json={"topic": "followup", "conversation_id": "missing"}).status_code == 404
        assert client.post("/research", json={"topic": "followup", "conversation_id": "busy"}).status_code == 409
        assert client.post("/research", json={"topic": "   "}).status_code == 422
        assert client.get("/conversations/missing").status_code == 404
        assert len(client.get("/jobs").json()) == 1


def test_conversation_context_is_bounded_without_losing_persisted_reports(environment):
    async def run():
        for index in range(9):
            await services.init_research_job(str(index), f"question-{index}", "0" if index else None)
            await services.update_research_status(str(index), status="complete", report="x" * 5000)
        await services.init_research_job("latest", "followup", "0")
        return await services.get_conversation_context("latest"), await services.get_conversation("0")

    context, history = asyncio.run(run())
    assert len(context) == 12
    assert sum(len(message["content"]) for message in context) <= 12000
    assert context[0]["content"] == "question-0"
    assert context[-2]["content"] == "question-8"
    assert all(len(turn["report"]) == 5000 for turn in history[:-1])


def test_conversation_migration_preserves_existing_job_reports(environment):
    import sqlite3
    connection = sqlite3.connect(services.DB_PATH)
    connection.execute("""CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY, topic TEXT, status TEXT, step TEXT, iteration INTEGER,
        report TEXT, error TEXT, timings_json TEXT, source_domains_count INTEGER,
        grounding_json TEXT, created_at TEXT, updated_at TEXT
    )""")
    connection.execute("""INSERT INTO jobs (job_id, topic, status, report, created_at, updated_at)
                           VALUES ('legacy', 'Original topic', 'complete', 'Saved report', 'now', 'now')""")
    connection.commit()
    connection.close()
    history = asyncio.run(services.get_conversation("legacy"))
    assert history[0]["report"] == "Saved report"
    assert history[0]["conversation_id"] == "legacy"
    assert history[0]["turn_index"] == 0
