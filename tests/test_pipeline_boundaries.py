import asyncio
import pickle
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import faiss
import numpy as np
import pytest

from backend import agent, chunker, evidence, groq_client, llm_fallback, reranker, retry, synthesizer
from backend.models import GroundingResult, ScoredChunk
from backend.vecstore import EMBED_DIM, LazyVectorStore, VectorStore


def bare_store(path):
    """Use real FAISS and checkpoint code without loading an embedding model."""
    store = VectorStore.__new__(VectorStore)
    store.path = str(path)
    store._lock = threading.RLock()
    store.model = SimpleNamespace(encode=lambda texts, **kwargs: np.zeros((len(texts), EMBED_DIM), dtype="float32"))
    store._ensure_index()
    return store


def test_vector_filter_expands_without_returning_duplicate_neighbors(tmp_path):
    store = bare_store(tmp_path / "vectors")
    vectors = np.zeros((100, EMBED_DIM), dtype="float32")
    vectors[:, 0] = np.arange(100)
    store.index.add(vectors)
    store.texts = [f"document-{i}" for i in range(100)]
    store.metadata = [{"job_id": "target" if i in {2, 70} else "other", "chunk_index": i} for i in range(100)]
    hits = store.query("query", top_k=2, job_id="target")
    assert [hit["text"] for hit in hits] == ["document-2", "document-70"]
    assert store.query("query", top_k=0) == []


def test_checkpoint_is_atomic_and_rolls_back_failed_insert(tmp_path, monkeypatch):
    path = tmp_path / "vectors"
    store = bare_store(path)
    store.add_texts(["original"], [{"job_id": "a"}])
    checkpoint = path.with_suffix(".store").read_bytes()

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr("backend.vecstore.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        store.add_texts(["uncommitted"], [{"job_id": "b"}])
    assert store.texts == ["original"]
    assert store.index.ntotal == 1
    assert path.with_suffix(".store").read_bytes() == checkpoint
    restored = bare_store(path)
    assert restored.texts == ["original"]
    assert restored.index.ntotal == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_vector_index_is_read_and_preserved_on_migration(tmp_path):
    path = tmp_path / "legacy"
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.zeros((1, EMBED_DIM), dtype="float32"))
    faiss.write_index(index, str(path) + ".index")
    with open(str(path) + ".meta", "wb") as output:
        pickle.dump((["old"], [{"job_id": "old-job"}]), output)
    original_index = path.with_suffix(".index").read_bytes()
    store = bare_store(path)
    store.add_texts(["new"], [{"job_id": "new-job"}])
    assert bare_store(path).texts == ["old", "new"]
    assert path.with_suffix(".index").read_bytes() == original_index


def test_vector_metadata_validation_does_not_mutate_store(tmp_path):
    store = bare_store(tmp_path / "vectors")
    with pytest.raises(ValueError, match="align"):
        store.add_texts(["text"], [])
    assert store.index.ntotal == 0
    assert store.texts == []


def test_lazy_store_does_not_load_model_at_construction():
    assert LazyVectorStore("unused")._store is None


def test_cross_encoder_applies_one_sigmoid_and_caches_model(monkeypatch):
    loaded = []
    predictions = []

    class FakeCrossEncoder:
        def __init__(self, name, device):
            loaded.append((name, device))
            self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
            self.max_length = 512

        def tokenizer(self, queries, documents, **kwargs):
            assert kwargs["padding"] is False
            return {"input_ids": [[1, 2], [1, 2, 3, 4, 5]]}

        def predict(self, pairs, **kwargs):
            predictions.append((pairs, kwargs))
            assert isinstance(kwargs["activation_fn"], Identity)
            return [0, 2]

    class Identity:
        pass

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), nn=SimpleNamespace(Identity=Identity))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    monkeypatch.setattr(reranker, "_CROSS_ENCODER", None)
    monkeypatch.setattr(reranker, "_CROSS_ENCODER_LOADED_MODEL", None)
    monkeypatch.setattr(reranker, "CROSS_ENCODER_MAX_CHARS", 20)
    items = [{"text": "a" * 100}, {"text": "second document"}]
    scores, diagnostics = reranker._cross_encoder_score_batch("fake", "topic", items)
    reranker._cross_encoder_score_batch("fake", "topic", items)
    assert scores == pytest.approx([0.5, 0.8807970779])
    assert len(loaded) == 1
    assert predictions[0][0][0] == ("topic", "a" * 20)
    assert diagnostics["min_tokens"] == 2
    assert diagnostics["max_tokens"] == 5
    assert diagnostics["total_ms"] >= diagnostics["inference_ms"]


def test_duplicate_copy_is_removed_across_sources_before_selection():
    copied = "This source explains evidence grounded research and the importance of accurate retrieval. " * 5
    first = {"text": copied, "url": "https://original.example"}
    second = {"text": copied, "url": "https://copy.example"}
    pairs = [(first, ScoredChunk(text=copied, score=0.9)), (second, ScoredChunk(text=copied, score=0.8))]
    kept, duplicates = evidence.deduplicate_rerank_candidates(pairs)
    assert [pair[0]["url"] for pair in kept] == ["https://original.example"]
    assert len(duplicates) == 1


def test_short_chunk_does_not_discard_following_long_unit():
    long_sentence = "Useful evidence " * 8 + "."
    chunks = chunker.chunk_text("Tiny. " + long_sentence, target_chunk_chars=80)
    assert long_sentence in chunks


def test_source_preparation_assigns_stable_ids_and_keeps_full_search_text():
    raw = ("Search evidence describes hybrid retrieval and lexical matching. " * 80
           + "The final passage must also survive text preparation.")
    items, groups = chunker.prepare_source_chunks([
        ({"url": "https://a.example", "raw_content": raw}, "truncated fallback"),
        ({"url": "https://b.example", "raw_content": ""},
         "A different source provides enough extracted text for a complete chunk."),
    ])
    assert "The final passage must also survive" in " ".join(item["text"] for item in items)
    assert [item["chunk_index"] for item in items] == list(range(len(items)))
    assert items[-1]["source_chunk_index"] == 0
    assert groups[-1]["chunks"][0]["chunk_index"] == len(items) - 1


def test_pipeline_snapshot_contract_does_not_share_mutable_job_fields():
    from backend.models import new_pipeline_snapshot
    settings = {"retrieval_top_k": 40}
    first = new_pipeline_snapshot("topic", "one", version="v1", timestamp="now", config=settings)
    second = new_pipeline_snapshot("topic", "two", version="v1", timestamp="now", config=settings)
    first["search_results"].append({"query": "first job only"})
    first["timings"]["searching_seconds"] = 1
    first["config"]["retrieval_top_k"] = 10
    assert second["search_results"] == []
    assert second["timings"] == {}
    assert second["config"]["retrieval_top_k"] == 40
    assert settings["retrieval_top_k"] == 40


@pytest.mark.parametrize("response", ['{"research_topic": "Limitations of hybrid retrieval"}', "invalid JSON"])
def test_followup_resolution_preserves_context_when_model_output_is_invalid(monkeypatch, response):
    generate = AsyncMock(return_value=response)
    monkeypatch.setattr(agent, "llm_generate", generate)
    history = [{"role": "user", "content": "Explain hybrid retrieval"},
               {"role": "assistant", "content": "It combines lexical and semantic search."}]
    question = "What are its limitations?"
    resolved = asyncio.run(agent.resolve_followup_topic(question, history))
    if response.startswith("{"):
        assert resolved == "Limitations of hybrid retrieval"
    else:
        assert "Explain hybrid retrieval" in resolved
        assert question in resolved
    assert "lexical and semantic search" in generate.await_args.args[0]
    assert generate.await_args.kwargs["task_type"] == "planning"


@pytest.mark.parametrize("error,falls_back", [(RuntimeError("RATE_LIMIT"), True), (RuntimeError("invalid request"), False)])
def test_provider_fallback_is_only_for_rate_limits(monkeypatch, error, falls_back):
    primary = AsyncMock(side_effect=error)
    secondary = AsyncMock(return_value="fallback text")
    monkeypatch.setattr(llm_fallback, "groq_generate", primary)
    monkeypatch.setattr(llm_fallback, "gemini_generate", secondary)
    if falls_back:
        text, provider, model, fallback = asyncio.run(llm_fallback.generate_with_details("prompt"))
        assert (text, provider, fallback) == ("fallback text", "gemini", True)
        assert model == llm_fallback.DEFAULT_GEMINI_MODEL
    else:
        with pytest.raises(RuntimeError, match="invalid request"):
            asyncio.run(llm_fallback.generate_with_details("prompt"))
        secondary.assert_not_awaited()
    primary.assert_awaited_once()


def test_retry_does_not_sleep_after_last_attempt(monkeypatch):
    sleep = AsyncMock()
    factory = AsyncMock(side_effect=RuntimeError("RATE_LIMIT"))
    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="RATE_LIMIT"):
        asyncio.run(retry.async_retry(factory))
    assert factory.await_count == 5
    assert [call.args[0] for call in sleep.await_args_list] == [2, 5, 10, 20]


@pytest.mark.parametrize("termination", ["exception", "eof", "done", "finish_reason"])
def test_stream_client_never_retries_after_emitting_tokens(monkeypatch, termination):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield 'data: {"choices": [{"delta": {"content": "partial"}}]}'
            if termination == "exception":
                raise RuntimeError("connection lost")
            if termination == "done":
                yield "data: [DONE]"
            if termination == "finish_reason":
                yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            return False

    class Client:
        def stream(self, *args, **kwargs):
            calls.append(1)
            return Stream()

    monkeypatch.setattr(groq_client, "CLIENT", Client())
    monkeypatch.setattr(groq_client, "GROQ_API_KEY", "fake-for-test")
    monkeypatch.setattr(groq_client, "GROQ_SEMAPHORE", asyncio.Semaphore(2))
    parts = []

    async def run():
        async for part in groq_client.groq_stream_content("prompt"):
            parts.append(part)

    if termination == "exception":
        with pytest.raises(RuntimeError, match="connection lost"):
            asyncio.run(run())
    elif termination == "eof":
        with pytest.raises(RuntimeError, match="completion marker"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert parts == ["partial"]
    assert len(calls) == 1


@pytest.mark.parametrize("partial", [False, True])
def test_synthesis_stream_fallback_does_not_mix_reports(monkeypatch, partial):
    async def stream(*args, **kwargs):
        if partial:
            yield "partial primary answer"
        raise RuntimeError("RATE_LIMIT")

    monkeypatch.setattr(synthesizer, "SYNTHESIS_PROVIDER", "groq")
    monkeypatch.setattr(synthesizer, "groq_stream_content", stream)
    fallback = AsyncMock(return_value=("secondary answer", "gemini", "model", False))
    monkeypatch.setattr(synthesizer, "generate_with_details", fallback)
    monkeypatch.setattr(synthesizer, "ground_report", AsyncMock(return_value=GroundingResult(
        claims=[], verified_count=0, unverified_count=0, overall_score=0,
    )))
    monkeypatch.setattr(synthesizer.services, "SYNTHESIS_DEBUG", {})
    queue = asyncio.Queue()
    debug = {}

    async def run():
        return await synthesizer.generate(
            "topic", "job", None, stream_queue=queue, snapshot=debug,
            selected_chunks=[{"text": "source evidence", "url": "https://a.example"}],
        )

    if partial:
        with pytest.raises(RuntimeError, match="RATE_LIMIT"):
            asyncio.run(run())
        fallback.assert_not_awaited()
        assert queue.get_nowait() == "partial primary answer"
        assert not debug["synthesis_success"]
    else:
        report, count, _ = asyncio.run(run())
        assert report == "secondary answer"
        assert debug["synthesis_fallback_triggered"]
        assert count == 1
        assert queue.get_nowait() == "secondary answer"


def test_empty_synthesis_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(synthesizer, "generate_with_details", AsyncMock(return_value=("", "groq", "model", False)))
    monkeypatch.setattr(synthesizer.services, "SYNTHESIS_DEBUG", {})
    debug = {}
    with pytest.raises(RuntimeError, match="empty"):
        asyncio.run(synthesizer.generate("topic", "job", None, snapshot=debug,
                                        selected_chunks=[{"text": "evidence"}]))
    assert not debug["synthesis_success"]
