"""Pydantic request / response schemas for the Egyptian Legal RAG API.

Includes models for synchronous ``/ask``, streaming ``/ask/stream``,
health checks, and metrics.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="السؤال القانوني باللغة العربية أو الإنجليزية",
        json_schema_extra={"example": "ما هي شروط صحة عقد البيع؟"},
    )
    top_k: int = Field(default=3, ge=1, le=10, description="عدد المواد المسترجعة")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class SourceItem(BaseModel):
    article_number: int = Field(..., description="رقم المادة في القانون المدني المصري")
    citation: str = Field(..., description="اسم الوثيقة ورقم المادة الرسمي")
    text_snippet: str = Field(..., description="مقتطف من نص المادة")
    score: float | None = Field(
        default=None, description="درجة المطابقة أو إعادة الترتيب"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    model_version: str = "v0.2.0"
    latency_ms: float | None = Field(
        default=None, description="End-to-end latency in milliseconds"
    )


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    version: str
    uptime_seconds: float | None = None


# ---------------------------------------------------------------------------
# SSE streaming event types
# ---------------------------------------------------------------------------
class StreamEventType(str, Enum):
    """Server-Sent Event types for ``/ask/stream``."""

    SOURCES = "sources"
    CHUNK = "chunk"
    DONE = "done"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single Server-Sent Event payload."""

    event: StreamEventType
    data: str  # JSON-encoded payload for the event type


# ---------------------------------------------------------------------------
# OOD rejection response
# ---------------------------------------------------------------------------
class OODResponse(BaseModel):
    """Response for Out-of-Domain queries."""

    rejected: bool = True
    reason: str = "السؤال خارج نطاق القانون المدني المصري — The query is outside the Egyptian Civil Code domain."
