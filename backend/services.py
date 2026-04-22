# backend/services.py — Redis/JSON status, vector store, job corpus, streams, job index
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .vecstore import VectorStore

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
logger = logging.getLogger("research_agent")

if REDIS_AVAILABLE:
    try:
        redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    except Exception as e:
        redis_client = None
        logger.warning("Redis client not configured: %s", e)
else:
    redis_client = None

vec = VectorStore(path=os.getenv("VEC_PATH", "data/faiss_index"))

RESEARCH_KEY_PREFIX = "research:"
DB_PATH = Path(os.getenv("DB_PATH", "data/lumen.db"))
LEGACY_JOBS_INDEX_PATH = Path("data/jobs_index.json")
LEGACY_DOCS_DIR = Path("data/docs")

# Per-job lexical corpus (aligns with metadata chunk_index on each row)
JOB_CORPUS: dict[str, list[str]] = {}
JOB_CHUNK_META: dict[str, list[dict]] = {}

report_stream_queues: dict[str, asyncio.Queue[str | None]] = {}
_db_lock = asyncio.Lock()
_db_initialized = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_record(job_id: str, topic: str = "") -> dict[str, Any]:
    ts = _now_iso()
    return {
        "job_id": job_id,
        "topic": topic,
        "status": "processing",
        "step": None,
        "iteration": None,
        "report": None,
        "error": None,
        "timings": {},
        "source_domains_count": None,
        "grounding": None,
        "created_at": ts,
        "updated_at": ts,
    }


def _connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_legacy_files_unlocked(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT COUNT(*) AS n FROM jobs")
    row = cur.fetchone()
    if row and int(row["n"]) > 0:
        return

    rows: list[dict[str, Any]] = []
    if LEGACY_JOBS_INDEX_PATH.exists():
        try:
            parsed = json.loads(LEGACY_JOBS_INDEX_PATH.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                rows = [r for r in parsed if isinstance(r, dict)]
        except Exception as e:
            logger.warning("legacy jobs index migration failed: %s", e)

    by_id: dict[str, dict[str, Any]] = {}
    for row_data in rows:
        job_id = str(row_data.get("job_id") or "").strip()
        if not job_id:
            continue
        by_id[job_id] = _default_record(job_id, str(row_data.get("topic") or ""))
        by_id[job_id]["status"] = str(row_data.get("status") or "processing")
        by_id[job_id]["created_at"] = str(row_data.get("created_at") or _now_iso())
        by_id[job_id]["updated_at"] = str(row_data.get("updated_at") or _now_iso())

    if LEGACY_DOCS_DIR.exists():
        for file in LEGACY_DOCS_DIR.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            job_id = str(data.get("job_id") or file.stem).strip()
            if not job_id:
                continue
            base = by_id.get(job_id, _default_record(job_id, str(data.get("topic") or "")))
            base.update(data)
            by_id[job_id] = base

    if not by_id:
        return

    for data in by_id.values():
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs (
                job_id, topic, status, step, iteration, report, error,
                timings_json, source_domains_count, grounding_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["job_id"],
                data.get("topic") or "",
                data.get("status") or "processing",
                data.get("step"),
                data.get("iteration"),
                data.get("report"),
                data.get("error"),
                json.dumps(data.get("timings") or {}, ensure_ascii=False),
                data.get("source_domains_count"),
                json.dumps(data.get("grounding"), ensure_ascii=False)
                if data.get("grounding") is not None
                else None,
                data.get("created_at") or _now_iso(),
                data.get("updated_at") or _now_iso(),
            ),
        )
    conn.commit()


async def _ensure_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
    async with _db_lock:
        if _db_initialized:
            return

        def _init() -> None:
            conn = _connect_db()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'processing',
                        step TEXT,
                        iteration INTEGER,
                        report TEXT,
                        error TEXT,
                        timings_json TEXT,
                        source_domains_count INTEGER,
                        grounding_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at DESC)"
                )
                _migrate_legacy_files_unlocked(conn)
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(_init)
        _db_initialized = True


def ensure_report_queue(job_id: str) -> asyncio.Queue[str | None]:
    if job_id not in report_stream_queues:
        report_stream_queues[job_id] = asyncio.Queue()
    return report_stream_queues[job_id]


async def finish_report_stream(job_id: str) -> None:
    q = report_stream_queues.get(job_id)
    if q is not None:
        await q.put(None)


def job_chunk_count(job_id: str) -> int:
    return len(JOB_CORPUS.get(job_id, []))


def append_job_chunks(job_id: str, texts: list[str], metadatas: list[dict]) -> None:
    """Extend in-memory BM25 corpus; ``metadatas`` must already include correct ``chunk_index``."""
    if job_id not in JOB_CORPUS:
        JOB_CORPUS[job_id] = []
        JOB_CHUNK_META[job_id] = []
    JOB_CORPUS[job_id].extend(texts)
    JOB_CHUNK_META[job_id].extend(metadatas)


def get_job_corpus(job_id: str) -> tuple[list[str], list[dict]]:
    return JOB_CORPUS.get(job_id, []), JOB_CHUNK_META.get(job_id, [])


def init_job_chunk_storage(job_id: str) -> None:
    JOB_CORPUS[job_id] = []
    JOB_CHUNK_META[job_id] = []


async def register_job_in_index(job_id: str, topic: str) -> None:
    await _ensure_db()
    ts = _now_iso()

    def _insert() -> None:
        conn = _connect_db()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, topic, status, created_at, updated_at, timings_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, topic or "", "processing", ts, ts, "{}"),
            )
            conn.commit()
        finally:
            conn.close()

    async with _db_lock:
        await asyncio.to_thread(_insert)


async def sync_job_index_status(
    job_id: str, status: str | None = None, topic: str | None = None
) -> None:
    await _ensure_db()
    now = _now_iso()

    def _upsert() -> None:
        conn = _connect_db()
        try:
            existing = conn.execute(
                "SELECT job_id, topic, created_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = COALESCE(?, status),
                        topic = COALESCE(NULLIF(?, ''), topic),
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (status, topic if topic is not None else "", now, job_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, topic, status, created_at, updated_at, timings_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, topic or "", status or "processing", now, now, "{}"),
                )
            conn.commit()
        finally:
            conn.close()

    async with _db_lock:
        await asyncio.to_thread(_upsert)


async def list_jobs() -> list[dict[str, Any]]:
    await _ensure_db()

    def _list() -> list[dict[str, Any]]:
        conn = _connect_db()
        try:
            rows = conn.execute(
                """
                SELECT job_id, topic, status, created_at, updated_at
                FROM jobs
                ORDER BY datetime(updated_at) DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    async with _db_lock:
        return await asyncio.to_thread(_list)


async def delete_research_job(job_id: str) -> bool:
    """Remove job from SQLite, Redis, and in-memory corpus/streams. Returns True if anything existed."""
    await _ensure_db()
    existed = False

    if redis_client:
        try:
            n = await redis_client.delete(f"{RESEARCH_KEY_PREFIX}{job_id}")
            if n:
                existed = True
        except Exception as e:
            logger.warning("redis delete research %s: %s", job_id, e)

    def _delete_db() -> int:
        conn = _connect_db()
        try:
            cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    async with _db_lock:
        if await asyncio.to_thread(_delete_db):
            existed = True

    if job_id in JOB_CORPUS or job_id in JOB_CHUNK_META:
        existed = True
    JOB_CORPUS.pop(job_id, None)
    JOB_CHUNK_META.pop(job_id, None)
    report_stream_queues.pop(job_id, None)

    return existed


async def update_research_status(job_id: str, **kwargs: Any) -> None:
    await _ensure_db()
    allowed = (
        "status",
        "step",
        "iteration",
        "report",
        "error",
        "timings",
        "topic",
        "source_domains_count",
        "grounding",
        "created_at",
        "updated_at",
    )
    patch = {k: v for k, v in kwargs.items() if k in allowed}
    if not patch:
        return

    data = await get_research_status(job_id)
    if not data:
        data = _default_record(job_id)
    patch.setdefault("updated_at", _now_iso())

    data.update(patch)
    data["job_id"] = job_id

    def _write_db() -> None:
        conn = _connect_db()
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, topic, status, step, iteration, report, error, timings_json,
                    source_domains_count, grounding_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    topic = excluded.topic,
                    status = excluded.status,
                    step = excluded.step,
                    iteration = excluded.iteration,
                    report = excluded.report,
                    error = excluded.error,
                    timings_json = excluded.timings_json,
                    source_domains_count = excluded.source_domains_count,
                    grounding_json = excluded.grounding_json,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    data["job_id"],
                    data.get("topic") or "",
                    data.get("status") or "processing",
                    data.get("step"),
                    data.get("iteration"),
                    data.get("report"),
                    data.get("error"),
                    json.dumps(data.get("timings") or {}, ensure_ascii=False),
                    data.get("source_domains_count"),
                    json.dumps(data.get("grounding"), ensure_ascii=False)
                    if data.get("grounding") is not None
                    else None,
                    data.get("created_at") or _now_iso(),
                    data.get("updated_at") or _now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    async with _db_lock:
        await asyncio.to_thread(_write_db)

    if redis_client:
        mapping: dict[str, str] = {}
        for k, v in data.items():
            if k in ("timings",) and isinstance(v, dict):
                mapping[k] = json.dumps(v)
            elif k == "iteration" and v is not None:
                mapping[k] = str(int(v))
            elif k == "source_domains_count" and v is not None:
                mapping[k] = str(int(v))
            elif v is None:
                mapping[k] = ""
            elif isinstance(v, (dict, list)):
                mapping[k] = json.dumps(v)
            else:
                mapping[k] = str(v)
        try:
            await redis_client.hset(f"{RESEARCH_KEY_PREFIX}{job_id}", mapping=mapping)
        except Exception as e:
            logger.warning("redis hset research %s: %s", job_id, e)

    await sync_job_index_status(job_id, status=data.get("status"), topic=data.get("topic"))


async def init_research_job(job_id: str, topic: str) -> None:
    init_job_chunk_storage(job_id)
    ensure_report_queue(job_id)
    await register_job_in_index(job_id, topic)
    await update_research_status(
        job_id,
        topic=topic,
        status="processing",
        step="starting",
        iteration=0,
        report=None,
        error=None,
        timings={},
        source_domains_count=None,
        grounding=None,
    )


def _hash_to_status(job_id: str, h: dict[str, str]) -> dict[str, Any]:
    data: dict[str, Any] = {"job_id": job_id}
    for k, v in h.items():
        if k == "timings":
            try:
                data[k] = json.loads(v) if v else {}
            except json.JSONDecodeError:
                data[k] = {}
        elif k == "iteration":
            if v == "" or v is None:
                data[k] = None
            else:
                try:
                    data[k] = int(v)
                except ValueError:
                    data[k] = None
        elif k == "source_domains_count":
            if v == "" or v is None:
                data[k] = None
            else:
                try:
                    data[k] = int(v)
                except ValueError:
                    data[k] = None
        elif k == "grounding":
            try:
                data[k] = json.loads(v) if v else None
            except json.JSONDecodeError:
                data[k] = None
        elif k in ("report", "error", "step") and v == "":
            data[k] = None
        else:
            data[k] = v
    return data


async def get_research_status(job_id: str) -> Optional[dict[str, Any]]:
    await _ensure_db()
    if redis_client:
        try:
            h = await redis_client.hgetall(f"{RESEARCH_KEY_PREFIX}{job_id}")
            if h:
                return _hash_to_status(job_id, h)
        except Exception as e:
            logger.warning("redis get research %s: %s", job_id, e)

    def _get_db() -> Optional[dict[str, Any]]:
        conn = _connect_db()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            timings_raw = data.pop("timings_json", None)
            grounding_raw = data.pop("grounding_json", None)
            try:
                data["timings"] = json.loads(timings_raw) if timings_raw else {}
            except json.JSONDecodeError:
                data["timings"] = {}
            try:
                data["grounding"] = json.loads(grounding_raw) if grounding_raw else None
            except json.JSONDecodeError:
                data["grounding"] = None
            return data
        finally:
            conn.close()

    async with _db_lock:
        return await asyncio.to_thread(_get_db)


async def run_research_task(topic: str, job_id: str) -> None:
    from .agent import run_research_loop

    await run_research_loop(topic, job_id)


def get_logger(name: str = "research_agent") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
