from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="السؤال القانوني باللغة العربية أو الإنجليزية",
        json_schema_extra={"example": "ما هي شروط صحة عقد البيع؟"},
    )
    top_k: int = Field(default=3, ge=1, le=10, description="عدد المواد المسترجعة")


class SourceItem(BaseModel):
    article_number: int = Field(..., description="رقم المادة في القانون المدني المصري")
    citation: str = Field(..., description="اسم الوثيقة ورقم المادة الرسمي")
    text_snippet: str = Field(..., description="مقتطف من نص المادة")
    score: Optional[float] = Field(
        default=None, description="درجة المطابقة أو إعادة الترتيب"
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    model_version: str = "v0.1.0"


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    version: str
