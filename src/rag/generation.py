"""Egyptian Legal RAG — Legal Answer Generator with Ollama (qwen2.5:7b) & Langfuse Cloud Tracing.

Supports:
- Local LLM generation via Ollama (qwen2.5:7b) OpenAI-compatible /v1/chat/completions endpoint
- Automatic fallback to deterministic legal citation synthesis when Ollama is offline (e.g., CI/CD)
- Langfuse Cloud distributed tracing when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("egyptian_legal_rag.generation")


class LegalGenerator:
    """Legal Generator synthesizing citation-grounded answers via Ollama (qwen2.5:7b) & Langfuse."""

    def __init__(self, llm_client: Any = None):
        self.client = llm_client
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

        # Langfuse Cloud keys
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse = None

        if self.langfuse_public_key and self.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self.langfuse = Langfuse(
                    public_key=self.langfuse_public_key,
                    secret_key=self.langfuse_secret_key,
                    host=self.langfuse_host,
                )
                logger.info(
                    "Langfuse Cloud tracing initialized for host: %s",
                    self.langfuse_host,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse initialization skipped: %s", exc)

    def build_prompt(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """Build formal legal prompt framed by retrieved article citations."""
        context_blocks = []
        for c in context_chunks:
            meta = c.get("metadata", {})
            art_no = meta.get("article_number", c.get("article_number", "غير محدد"))
            citation = meta.get("citation", f"المادة {art_no}")
            context_blocks.append(f"[{citation}]\n{c['text']}")

        joined_context = "\n\n".join(context_blocks)

        prompt = f"""أنت مستشار قانوني خبير في القانون المدني المصري.
أجب عن سؤال المستخدم بدقة وأمانة قانونية كاملة مستنداً فقط إلى المواد القانونية المرفقة أدناه.
يجب عليك ذكر رقم المادة والاستشهاد الرسمي (Citation) لكل نقطة قانونية تذكرها في إجابتك.
إذا لم تجد الإجابة صراحة في المواد المرفقة، قل بوضوح: "لا تتوفر مادة كافية في السياق المرفق للإجابة عن هذا السؤال".

المواد القانونية المتاحة:
---------------------
{joined_context}
---------------------

سؤال المستخدم:
{query}

الإجابة القانونية المدعومة بالاستشهادات:"""
        return prompt

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """Generate citation-grounded answer using Ollama (qwen2.5:7b) or citation fallback."""
        prompt = self.build_prompt(query, context_chunks)
        answer = None

        # 1. Custom client provided
        if self.client:
            try:
                response = self.client.complete(prompt)
                answer = str(response)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Custom LLM client failed: %s", exc)

        # 2. Query local Ollama endpoint (qwen2.5:7b)
        if not answer:
            try:
                endpoint = f"{self.ollama_url.rstrip('/')}/v1/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "أنت مستشار قانوني خبير في القانون المدني المصري.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(endpoint, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001
                answer = None

        # 3. Deterministic baseline citation fallback (CI / offline)
        if not answer:
            citations = []
            for c in context_chunks:
                meta = c.get("metadata", {})
                art_no = meta.get("article_number", c.get("article_number"))
                cit = meta.get(
                    "citation", f"المادة {art_no}" if art_no else "المادة المرفقة"
                )
                citations.append(cit)

            answer = (
                f"استناداً إلى نصوص القانون المدني المصري الواردة في ({', '.join(citations)})، "
                f"فإن أحكام المسألة تتحدد بالضوابط المقررة في نصوص المواد المرفقة طيه."
            )

        # 4. Log trace to Langfuse Cloud if enabled
        if self.langfuse:
            try:
                self.langfuse.trace(
                    name="legal_rag_generation",
                    input={"question": query, "chunks_count": len(context_chunks)},
                    output={"answer": answer},
                    metadata={"model": self.model_name},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse trace logging skipped: %s", exc)

        return answer
