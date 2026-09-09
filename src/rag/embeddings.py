from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# نموذج خفيف وسريع ويدعم العربية والإنجليزية بامتياز للبيئة المحلية
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """تضمين قائمة نصوص وإرجاع مصفوفة numpy."""
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> list[float]:
        """تضمين سؤال أو استعلام فردي."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.tolist()
