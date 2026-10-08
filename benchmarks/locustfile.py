"""Locust load & stress test for the Egyptian Legal RAG API.

Simulates concurrent legal query traffic covering:
- Synchronous ``/ask`` endpoint
- SSE streaming ``/ask/stream`` endpoint
- Out-of-Domain (OOD) queries (expected graceful 422 rejection)
- Health check probes

Usage
-----
    locust -f benchmarks/locustfile.py --host http://localhost:8000

Or headless mode for CI:
    locust -f benchmarks/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 -t 60s --csv=benchmarks/results
"""

from __future__ import annotations

import random

from locust import HttpUser, between, tag, task

# ---------------------------------------------------------------------------
# Legal query pools — realistic Egyptian Civil Code questions
# ---------------------------------------------------------------------------
LEGAL_QUESTIONS_AR = [
    "ما هي شروط صحة عقد البيع في القانون المدني المصري؟",
    "ما حكم القوة القاهرة في المسئولية المدنية؟",
    "ما هي أحكام فسخ العقد وفقاً للمادة 157؟",
    "ما هو سن الرشد في القانون المدني المصري؟",
    "ما هي شروط الأهلية في التعاقد؟",
    "ما حكم الغلط في العقود المدنية؟",
    "ما هي أحكام التعويض عن الضرر الأدبي؟",
    "ما هي مدة التقادم في الدعاوى المدنية؟",
    "ما هي شروط صحة الوكالة؟",
    "ما هي أحكام حق الملكية؟",
    "ما هو حكم الإثراء بلا سبب؟",
    "ما هي أحكام الكفالة في القانون المدني؟",
    "ما هي شروط الإيجار وحقوق المستأجر؟",
    "ما هي أحكام المادة 147 بشأن العقد شريعة المتعاقدين؟",
    "كيف ينتقل الالتزام بالضمان في عقد البيع؟",
]

LEGAL_QUESTIONS_EN = [
    "What are the conditions for contract validity under Egyptian Civil Code?",
    "How does Article 147 apply to contractual obligations?",
    "What is the legal age of majority in Egyptian Civil Law?",
    "What are the provisions for force majeure under civil liability?",
    "What are the rules regarding contract rescission?",
]

# Out-of-domain questions — should be rejected
OOD_QUESTIONS = [
    "What is the weather in Cairo today?",
    "How to make koshari?",
    "Tell me about quantum physics.",
    "What is the capital of France?",
    "كيف أطبخ المحشي؟",
]


class LegalRAGUser(HttpUser):
    """Simulates a legal professional querying the RAG system."""

    wait_time = between(0.5, 2.0)

    @tag("ask")
    @task(10)
    def ask_legal_question_ar(self):
        """Synchronous Arabic legal question."""
        q = random.choice(LEGAL_QUESTIONS_AR)
        with self.client.post(
            "/ask",
            json={"question": q, "top_k": 3},
            name="/ask [ar]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @tag("ask")
    @task(3)
    def ask_legal_question_en(self):
        """Synchronous English legal question."""
        q = random.choice(LEGAL_QUESTIONS_EN)
        with self.client.post(
            "/ask",
            json={"question": q, "top_k": 3},
            name="/ask [en]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @tag("stream")
    @task(5)
    def ask_stream(self):
        """SSE streaming legal question."""
        q = random.choice(LEGAL_QUESTIONS_AR)
        with self.client.post(
            "/ask/stream",
            json={"question": q, "top_k": 3},
            name="/ask/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                # Consume the stream
                for _line in resp.iter_lines():
                    pass
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @tag("ood")
    @task(2)
    def ask_ood_question(self):
        """Out-of-domain question — should return 422."""
        q = random.choice(OOD_QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": q, "top_k": 3},
            name="/ask [ood]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 422:
                resp.success()  # Expected rejection
            else:
                resp.failure(f"OOD not rejected: {resp.status_code}")

    @tag("health")
    @task(1)
    def health_check(self):
        """Health check probe."""
        self.client.get("/health", name="/health")
