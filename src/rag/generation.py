from __future__ import annotations

from typing import Any


class LegalGenerator:
    def __init__(self, llm_client: Any = None):
        self.client = llm_client

    def build_prompt(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        """بناء البرومبت القانوني المؤطر بنصوص المواد المسترجعة."""
        context_blocks = []
        for c in context_chunks:
            meta = c.get("metadata", {})
            art_no = meta.get("article_number", "غير محدد")
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
        """توليد الإجابة (مؤطرة بالنصوص ومجهزة للعمل محلياً أو مع أي LLM)."""
        prompt = self.build_prompt(query, context_chunks)

        # إذا كان هناك اتصال مفعّل بـ LLM (مثل vLLM أو API):
        if self.client:
            # مثال لنداء الـ API (OpenAI-compatible أو HuggingFace)
            response = self.client.complete(prompt)
            return str(response)

        # Baseline Generator محلي يجمع الاستشهادات القانونية بدقة حتى ربط vLLM في Module 3
        citations = [
            c["metadata"]["citation"] for c in context_chunks if "metadata" in c
        ]
        return (
            f"استناداً إلى نصوص القانون المدني المصري الواردة في ({', '.join(citations)})، "
            f"فإن أحكام المسألة تتحدد بالضوابط المقررة في نصوص المواد المرفقة طيه."
        )
