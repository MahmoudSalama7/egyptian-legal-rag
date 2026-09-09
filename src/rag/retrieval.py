from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from src.rag.embeddings import EmbeddingService

INDEX_STORE_DIR = Path("data/processed/index_store")
CHUNKS_PATH = Path("data/processed/civil_code_chunks.json")


class LegalRetriever:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embed_service = embedding_service or EmbeddingService()
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray | None = None
        self._load_or_build_index()

    def _load_or_build_index(self):
        INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vectors_file = INDEX_STORE_DIR / "vectors.npy"

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chunks file missing: {CHUNKS_PATH}")

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if vectors_file.exists():
            self.vectors = np.load(vectors_file)
        else:
            texts = [c["text"] for c in self.chunks]
            print(f"Generating embeddings for {len(texts)} chunks...")
            self.vectors = self.embed_service.embed_texts(texts)
            np.save(vectors_file, self.vectors)
            print(f"Saved vectors to {vectors_file}")

    def retrieve(
        self, query: str, top_k: int = 5, language: str | None = None
    ) -> list[dict[str, Any]]:
        """استرجاع أفضل Chunks مطابقة للاستعلام مع إمكانية التصفية باللغة."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = np.array(self.embed_service.embed_query(query))

        # حساب Cosine Similarity (المتجهات مطبّع عليها Normalize مسبقاً)
        scores = np.dot(self.vectors, query_vec)

        # ترتيب النتائج تنازلياً
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if language and chunk.get("language") != language:
                continue

            results.append(
                {
                    **chunk,
                    "score": float(scores[idx]),
                }
            )

            if len(results) >= top_k:
                break

        return results
