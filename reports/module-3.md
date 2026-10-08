# Module 3 — High-Performance Serving & Low-Latency Deployment

## 1. Serving Architecture

### 1.1 Production API Design

The Egyptian Legal RAG API is built on **FastAPI** with an asynchronous architecture:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI 0.110+ | Async HTTP, automatic OpenAPI docs |
| ASGI Server | Uvicorn | High-performance event loop |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` | Bilingual (AR/EN) dense retrieval |
| Cross-Encoder | `mmarco-mMiniLMv2-L12-H384-v1` | Reranking for precision |
| Generation | Template-based (pluggable LLM client) | Citation-grounded answers |

### 1.2 Endpoints

| Method | Path | Type | Description |
|--------|------|------|-------------|
| `GET` | `/health` | Sync | Readiness probe with uptime tracking |
| `POST` | `/ask` | Sync | Full RAG pipeline with latency reporting |
| `POST` | `/ask/stream` | SSE | Real-time token/chunk streaming |

### 1.3 SSE Streaming (`/ask/stream`)

The streaming endpoint delivers answers via **Server-Sent Events (SSE)** with three event types:

1. **`sources`** — JSON array of retrieved legal articles (emitted first)
2. **`chunk`** — Incremental answer text (word-level chunking, ~5 events)
3. **`done`** — Completion signal with latency metadata
4. **`error`** — Error description for failed queries

```
event: sources
data: [{"article_number": 147, "citation": "المادة 147", ...}]

event: chunk
data: استناداً إلى نصوص القانون المدني المصري

event: chunk
data: الواردة في (المادة 147، المادة 158)

event: done
data: {"latency_ms": 234.56}
```

---

## 2. Concurrency & Micro-Batching Design

### 2.1 Adaptive Micro-Batching

The `MicroBatcher` class (`src/api/batching.py`) groups individual requests into batches for higher throughput:

- **Embedding Batcher**: `max_batch_size=32`, `max_wait_ms=30ms`
- **Reranking Batcher**: `max_batch_size=8`, `max_wait_ms=25ms`

Architecture:
- Callers `submit()` a payload and receive an `asyncio.Future`
- Background flush loop checks every `max_wait_ms` or when batch is full
- Processing runs in a thread pool (`run_in_executor`) to avoid blocking the event loop

### 2.2 OOD (Out-of-Domain) Handling

Queries outside the Egyptian Civil Code domain are gracefully rejected:
- `LegalGuardrails.is_legal_domain()` checks for legal keyword presence
- Returns HTTP 422 with bilingual explanation
- Streaming endpoint returns an `error` SSE event instead

---

## 3. Docker Build Architecture

### 3.1 Multi-Stage Build

```
Stage 1 (builder):
  - python:3.11-slim base
  - Install uv for fast dependency caching
  - Copy pyproject.toml first (layer cache)
  - Install dependencies → install project

Stage 2 (runtime):
  - python:3.11-slim base (minimal)
  - Non-root user (appuser, UID 1000)
  - Copy only site-packages + application code + data
  - HEALTHCHECK with start-period for model loading
```

### 3.2 Build Metrics

| Metric | Value |
|--------|-------|
| Final image layers | ~7 (minimized) |
| Non-root user | ✅ `appuser:1000` |
| Health check | ✅ 30s interval, 60s start-period |
| Dependency caching | ✅ via `uv` layer ordering |
| `.dockerignore` | ✅ Excludes tests, docs, VCS, IDE files |

---

## 4. Load & Stress Testing

### 4.1 Locust Test Configuration

The load test (`benchmarks/locustfile.py`) simulates realistic traffic:

| Task | Weight | Description |
|------|--------|-------------|
| `/ask [ar]` | 10 | Arabic legal questions |
| `/ask [en]` | 3 | English legal questions |
| `/ask/stream` | 5 | SSE streaming queries |
| `/ask [ood]` | 2 | Out-of-domain (expect 422) |
| `/health` | 1 | Health check probes |

Wait time: 0.5–2.0 seconds between requests per user.

### 4.2 Running Load Tests

```bash
# Interactive web UI
locust -f benchmarks/locustfile.py --host http://localhost:8000

# Headless CI mode
locust -f benchmarks/locustfile.py --host http://localhost:8000 \
    --headless -u 50 -r 5 -t 60s --csv=benchmarks/results
```

### 4.3 Profiler

The standalone profiler (`benchmarks/profiler.py`) measures sequential latency:

```bash
python -m benchmarks.profiler --url http://localhost:8000 --requests 100 --output benchmarks/profile_results.json
```

Reports: P50, P95, P99 latency, mean, RPS throughput.

### 4.4 Load Test Results (Placeholder)

> **Note:** Run the load test against a live instance to populate actual metrics.
> Results will be logged to `benchmarks/results_*.csv`.

| Metric | Target | Actual |
|--------|--------|--------|
| RPS (50 users) | ≥ 5 | _TBD_ |
| P50 latency | < 500ms | _TBD_ |
| P95 latency | < 2000ms | _TBD_ |
| P99 latency | < 5000ms | _TBD_ |
| Error rate | < 1% | _TBD_ |

---

## 5. Test Coverage

Module 3 adds comprehensive tests:

- `test_api.py` — Health, sync /ask, SSE streaming, OOD rejection, prompt injection
- `test_batching.py` — MicroBatcher single/multi-item, stats, error propagation
- Existing `test_rag.py` — Guardrails, embeddings, retrieval, reranking, generation

Target: **≥ 70% coverage** (enforced by `pytest --cov-fail-under=70`).

---

## 6. Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/api/main.py` | Modified | Production API with SSE, OOD, batching |
| `src/api/schemas.py` | Modified | SSE event types, OOD model, latency field |
| `src/api/batching.py` | **New** | Adaptive micro-batching engine |
| `src/api/__init__.py` | **New** | Package init |
| `docker/Dockerfile` | Modified | Multi-stage build with uv caching |
| `.dockerignore` | Modified | Comprehensive exclusion patterns |
| `benchmarks/locustfile.py` | **New** | Locust load test |
| `benchmarks/profiler.py` | **New** | Latency percentile profiler |
| `tests/test_api.py` | Modified | Expanded with streaming & OOD tests |
| `tests/test_batching.py` | **New** | MicroBatcher unit tests |
| `pyproject.toml` | Modified | v0.2.0, pytest-asyncio, locust |
| `reports/module-3.md` | **New** | This document |
