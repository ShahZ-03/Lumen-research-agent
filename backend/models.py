# backend/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ResearchRequest(BaseModel):
    topic: str


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

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

class GenerateNotesRequest(BaseModel):
    doc_id: str
    stream: Optional[bool] = False


class StatusResponse(BaseModel):
    doc_id: str
    status: str
    error: Optional[str] = None
    timings: Dict[str, Any] = {}
