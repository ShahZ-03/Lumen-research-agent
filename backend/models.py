# backend/models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=8000)
    conversation_id: Optional[str] = Field(default=None, min_length=1, max_length=128)

    @field_validator("topic")
    @classmethod
    def nonempty_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A question is required")
        return value


class GroundedClaim(BaseModel):
    claim: str
    score: float
    verified: bool
    best_source_url: Optional[str] = None
    best_source_title: Optional[str] = None


class GroundingResult(BaseModel):
    claims: List[GroundedClaim]
    verified_count: int
    unverified_count: int
    overall_score: float


class ResearchStatus(BaseModel):
    job_id: str
    topic: str = ""
    conversation_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    turn_index: int = 0
    status: str  # processing | complete | error
    step: Optional[str] = None
    iteration: Optional[int] = None
    report: Optional[str] = None
    error: Optional[str] = None
    timings: Optional[Dict[str, Any]] = None
    source_domains_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    grounding: Optional[GroundingResult] = None


class JobSummary(BaseModel):
    job_id: str
    topic: str
    status: str
    created_at: str
    updated_at: str
    conversation_id: Optional[str] = None
    turn_index: int = 0


class ScoredChunk(BaseModel):
    text: str
    score: float
    url: Optional[str] = None
    title: Optional[str] = None


class UploadResponse(BaseModel):
    doc_id: str
    message: str

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    # if true, server will stream the LLM response instead of returning
    # a single JSON object; defaults to False for backwards compatibility
    stream: Optional[bool] = False

class StatusResponse(BaseModel):
    doc_id: str
    status: str
    error: Optional[str] = None
    timings: Dict[str, Any] = {}


def new_pipeline_snapshot(topic: str, job_id: str, *, version: str,
                          timestamp: str, config: dict) -> dict:
    """Construct the API/CLI diagnostic data contract with fresh mutable fields."""
    return {
        "pipeline_version": version, "job_id": job_id, "topic": topic,
        "timestamp": timestamp, "timings": {},
        "config": dict(config),
        "sub_questions": [], "sub_question_strategy": "",
        "search_results": [], "chunks_by_source": [], "scored_chunks": [],
        "chunks_kept_after_rerank": [], "retrieved_chunks": [],
        "rerank_method": "cross_encoder", "rerank_latency_ms": {},
        "scoring_fallback_triggered": False, "kept_count": 0, "dropped_count": 0,
        "dedup_stats": {}, "synthesis_prompt": "", "synthesis_context": [],
        "synthesis_output": "", "synthesis_success": False,
        "synthesis_provider": "", "synthesis_fallback_triggered": False,
        "source_domains_count": 0, "grounding": None,
    }
