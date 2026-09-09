from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from src.api.schemas import AskResponse, HealthResponse, QuestionRequest, SourceItem
from src.rag.generation import LegalGenerator
from src.rag.guardrails import LegalGuardrails
from src.rag.reranking import LegalReranker
from src.rag.retrieval import LegalRetriever

# متغيرات على مستوى التطبيق لتحميل النماذج مرة واحدة فقط
RETRIEVER: LegalRetriever | None = None
RERANKER: LegalReranker | None = None
GENERATOR: LegalGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RETRIEVER, RERANKER, GENERATOR
    print("[STARTUP] Initializing RAG components...")
    RETRIEVER = LegalRetriever()
    RERANKER = LegalReranker()
    GENERATOR = LegalGenerator()
    print("[STARTUP] RAG Engine ready for inference.")
    yield
    RETRIEVER = None
    RERANKER = None
    GENERATOR = None
    print("[SHUTDOWN] Cleared RAG pipeline state.")


app = FastAPI(
    title="Egyptian Civil Code Legal RAG API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    is_ready = RETRIEVER is not None and len(RETRIEVER.chunks) > 0
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        documents_indexed=len(RETRIEVER.chunks) if RETRIEVER else 0,
        version="0.1.0",
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest):
    if not RETRIEVER or not RERANKER or not GENERATOR:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )

    # 1. فحص حواجز الأمان (Guardrails)
    is_valid, validation_msg = LegalGuardrails.validate_input(request.question)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=validation_msg,
        )

    # 2. الاسترجاع الدلالي (Dense Retrieval)
    candidate_chunks = RETRIEVER.retrieve(
        query=request.question,
        top_k=max(request.top_k * 2, 6),
        language="ar",
    )

    # 3. إعادة الترتيب (Cross-Encoder Reranking)
    reranked_chunks = RERANKER.rerank(
        query=request.question,
        candidate_chunks=candidate_chunks,
        top_n=request.top_k,
    )

    # 4. التوليد القانوني المؤطر بالاستشهادات (Generation)
    answer = GENERATOR.generate(request.question, reranked_chunks)

    # 5. فحص مخرجات التوليد عبر Guardrails
    is_out_valid, out_msg = LegalGuardrails.validate_output(answer, reranked_chunks)
    if not is_out_valid:
        answer = out_msg

    sources = [
        SourceItem(
            article_number=c["metadata"]["article_number"],
            citation=c["metadata"]["citation"],
            text_snippet=c["text"][:150] + "...",
            score=round(c.get("rerank_score", c.get("score", 0.0)), 4),
        )
        for c in reranked_chunks
    ]

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        model_version="0.1.0",
    )
