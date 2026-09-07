from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    """Request model for submitting a search query."""
    question: str = Field(...,min_length=3, 
    example="ما هي عقوبة السرقة؟", 
    json_schema_extra={"example": "ما هي شروط صحة عقد البيع؟"}
    )
    
    top_k: int = Field(default=3, ge=1, le=10, 
    description="Number of relevant articles to return", 
    )

class SourceItem(BaseModel):
    """Source item for a single search result."""
    article_number: int = Field(..., description="رقم الماده في القانون المصري")
    citation: str = Field(..., description="اسم الوثيقه ورقم الماده")
    score: float = Field(..., ge=0.0, le=1.0, example=0.85)
    text_snippet: str = Field(..., description="مقتطف من نص المادة")


class AskResponse(BaseModel):
    """Response model for the final answer"""
    question: str
    answer: str
    sources: list[SourceItem]
    model_version: str = "v0.1.0"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    documents_indexed: int
    version: str

    