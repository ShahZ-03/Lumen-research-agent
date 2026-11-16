# backend/main.py — FastAPI research API
import uuid
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

from backend.models import GroundingResult, JobSummary, ResearchRequest, ResearchStatus
from backend import services

logger = services.get_logger()

app = FastAPI(title="Research Agent API")


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        # Safe local defaults for development.
        return ["http://localhost:8080", "http://127.0.0.1:8080"]
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    # Never allow wildcard when credentials are enabled.
    return [o for o in origins if o != "*"]


allowed_origins = _parse_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("data").mkdir(exist_ok=True)
Path("data/docs").mkdir(parents=True, exist_ok=True)


@app.post("/research")
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    try:
        turn = await services.init_research_job(job_id, req.topic, req.conversation_id)
    except services.ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except services.ConversationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(services.run_research_task, req.topic, job_id)
    logger.info("scheduled research job %s topic=%r", job_id, req.topic[:80])
    return {"job_id": job_id, "status": "processing", **turn}


def _status_to_model(job_id: str, info: dict) -> ResearchStatus:
    g_raw = info.get("grounding")
    grounding = None
    if isinstance(g_raw, dict):
        try:
            grounding = GroundingResult.model_validate(g_raw)
        except Exception:
            grounding = None
    return ResearchStatus(
        job_id=job_id,
        topic=info.get("topic") or "",
        conversation_id=info.get("conversation_id") or job_id,
        parent_job_id=info.get("parent_job_id") or None,
        turn_index=int(info.get("turn_index") or 0),
        status=str(info.get("status", "processing")),
        step=info.get("step"),
        iteration=info.get("iteration"),
        report=info.get("report"),
        error=info.get("error"),
        timings=info.get("timings") if isinstance(info.get("timings"), dict) else None,
        source_domains_count=info.get("source_domains_count"),
        created_at=info.get("created_at"),
        updated_at=info.get("updated_at"),
        grounding=grounding,
    )


@app.get("/status/{job_id}", response_model=ResearchStatus)
async def research_status(job_id: str):
    info = await services.get_research_status(job_id)
    if not info or not info.get("status"):
        raise HTTPException(status_code=404, detail="job_id not found")
    return _status_to_model(job_id, info)


@app.get("/jobs", response_model=list[JobSummary])
async def list_research_jobs():
    rows = await services.list_jobs()
    out: list[JobSummary] = []
    for row in rows:
        try:
            out.append(
                JobSummary(
                    job_id=str(row.get("job_id", "")),
                    topic=str(row.get("topic", "")),
                    status=str(row.get("status", "")),
                    created_at=str(row.get("created_at", "")),
                    updated_at=str(row.get("updated_at", "")),
                    conversation_id=row.get("conversation_id") or row["job_id"],
                    turn_index=int(row.get("turn_index") or 0),
                )
            )
        except Exception:
            continue
    return out


@app.get("/conversations/{conversation_id}")
async def conversation_history(conversation_id: str):
    turns = await services.get_conversation(conversation_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation_id": conversation_id,
        "turns": [_status_to_model(turn["job_id"], turn) for turn in turns],
    }


@app.get("/debug/synthesis/{job_id}")
async def synthesis_debug(job_id: str):
    info = services.get_synthesis_debug(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="debug snapshot not found for job_id")
    return info


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    deleted = await services.delete_research_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {"ok": True, "job_id": job_id}


@app.get("/stream/report/{job_id}")
async def stream_report(job_id: str):
    info = await services.get_research_status(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job_id not found")

    if info.get("status") in {"complete", "error"}:
        report = str(info.get("report") or "")

        async def replay():
            yield report.encode("utf-8")

        return StreamingResponse(replay(), media_type="text/plain; charset=utf-8")

    q = services.ensure_report_queue(job_id)

    async def gen():
        while True:
            item = await q.get()
            if item is None:
                break
            yield item.encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.get("/health")
async def health():
    return {"ok": True}
