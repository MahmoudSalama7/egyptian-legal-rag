from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from src.api.schemas import AskResponse, HealthResponse, QuestionRequest, SourceItem

# path to the chunk file
DATA_PATH = Path(__file__).parent.parent / "data"
CHUNKS_PATH = DATA_PATH / "processed" / "civil_code_chunks.json"

# cache for the chunks in RAM
CORPUS_CACHE: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """manages the application lifecycle (loads data at startup)"""
    global CORPUS_CACHE
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"لم يتم العثور على ملف البيانات: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        CORPUS_CACHE = json.load(f)
        print(f"[STARTUP] Successfully loaded {len(CORPUS_CACHE)} chunks into memory.")
    yield

    CORPUS_CACHE.clear()
    print("[SHUTDOWN] Corpus cache cleared.")

# initialize the app
app = FastAPI(
    title="Egyptian Civil Code RAG API",
    description="خدمة استرجاع وإجابة استفسارات القانون المدني المصري",
    version="0.1.0",    
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of the RAG system.
    Returns:
        HealthResponse: health status and number of indexed documents.
    """
    if len(CORPUS_CACHE) == 0:
        raise HTTPException(status_code=404, detail="No documents loaded into memory.") 
        
    return HealthResponse(      
        status="healthy",
        documents_indexed=len(CORPUS_CACHE),
        version="0.1.0"
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: QuestionRequest):
    """
    Process the question, retrieve the relevant chunks from memory,
    and generate a text answer through an internal LLM.
    
    Args:
        request (QuestionRequest): the user's question and the number of chunks to return.
    
    Returns:
        AskResponse: the answer with the list of source chunks.
    """
    from src.rag.reranker import rerank_query
    from src.rag.llm_provider import LLMProvider
    
    try:
        query_text = request.question
        top_k = request.top_k

        # 1) retrieve the relevant chunks 
        results = rerank_query(query_text, CORPUS_CACHE, top_k=top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant chunks found.")

        # 2) بناء السياق
        context_chunks = []
        for item in results:
            context_chunks.append({
                "article_number": item["article_number"],
                "citation": item["citation"],
                "content": item["text_snippet"],
                "score": float(item["score"])
            })

        # 3) إنشاء الإجابة عبر LLM داخلي
        llm_provider = LLMProvider()
        answer = llm_provider.generate_answer(
            query=query_text, 
            context_chunks=context_chunks
        )
        
        # 4) تحويل النتائج إلى تنسيق JSON المطلوب
        sources = [
            SourceItem(
                article_number=int(item["article_number"]),
                citation=str(item["citation"]),
                score=float(item["score"]),
                text_snippet=str(item["text_snippet"])
            )
            for item in results
        ]
        
        return AskResponse(
            question=query_text,
            answer=answer,
            sources=sources,
            model_version="0.1.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing the request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")