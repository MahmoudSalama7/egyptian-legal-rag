from __future__ import annotations

from typing import Any
from sentence_transformers import CrossEncoder

DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class LegalReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        # Cross-Encoder متعدد اللغات يدعم العربية
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidate_chunks: list[dict[str, Any]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not candidate_chunks:
            return []

        pairs = [[query, c["text"]] for c in candidate_chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(candidate_chunks, scores):
            chunk["rerank_score"] = float(score)

        # ترتيب النتائج بناءً على تقييم الـ Cross-Encoder
        sorted_chunks = sorted(
            candidate_chunks, key=lambda x: x["rerank_score"], reverse=True
        )
        return sorted_chunks[:top_n]
