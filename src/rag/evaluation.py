from __future__ import annotations

from typing import Any


class RAGEvaluator:
    @staticmethod
    def evaluate_retrieval(
        retrieved_chunks: list[dict[str, Any]], expected_article_number: int
    ) -> float:
        """حساب هل تم استرجاع المادة الصحيحة ضمن الـ Top-k (Hit Rate)."""
        if not retrieved_chunks:
            return 0.0
        for c in retrieved_chunks:
            if c.get("metadata", {}).get("article_number") == expected_article_number:
                return 1.0
        return 0.0

    @staticmethod
    def evaluate_citation_groundedness(
        answer: str, sources: list[dict[str, Any]]
    ) -> float:
        """فحص هل الإجابة استشهدت فعلياً بأرقام المواد الموجودة في المصادر."""
        if not sources:
            return 0.0
        valid_citations = 0
        for s in sources:
            art_no = str(s.get("metadata", {}).get("article_number", ""))
            if art_no and art_no in answer:
                valid_citations += 1
        return valid_citations / len(sources) if sources else 0.0
