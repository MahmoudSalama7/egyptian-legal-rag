# ⚖️ Egyptian Legal RAG (محرك الاسترجاع والتوليد المعزز للقانون المدني المصري)

[![CI/CD Pipeline](https://github.com/MahmoudSalama7/egyptian-legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/MahmoudSalama7/egyptian-legal-rag/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Observability-F46800.svg)](https://grafana.com/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **production-ready, bilingual (Arabic & English)** Retrieval-Augmented Generation (RAG) system specifically engineered for the **Egyptian Civil Code (القانون المدني المصري)**. This system delivers precise legal article retrieval, cross-encoder re-ranking, real-time SSE token streaming, input/output security guardrails, automated RAGAS quality evaluation, Prometheus/Grafana observability, and enterprise ML lifecycle management via DVC and MLflow.

---

## ⚡ 3-Command Quickstart

Run the complete production pipeline from scratch in three terminal commands:

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/MahmoudSalama7/egyptian-legal-rag.git && cd egyptian-legal-rag && uv sync --extra dev

# 2. Run reproducible DVC data pipeline & pytest quality suite (>=70% coverage gate)
uv run dvc repro && uv run pytest

# 3. Launch full production stack (API + Prometheus + Grafana + MLflow UI) via Docker Compose
docker compose up --build -d
```

---

## 🌟 Key System Capabilities

* **📜 Legal Data Ingestion & Normalization**: Automated PDF parsing (`PyMuPDF`/`pypdf`), Arabic text normalization, and article-preserving chunking.
* **🔍 Multilingual Vector Retrieval**: Dense embedding search tailored for Arabic/English legal queries (`paraphrase-multilingual-MiniLM-L12-v2`).
* **🎯 Cross-Encoder Reranking**: Re-ranking stage using multilingual Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) maximizing precision.
* **⚡ Real-Time SSE Token Streaming**: Server-Sent Events endpoint (`/ask/stream`) delivering real-time answer token chunks and source articles.
* **⚙️ Adaptive Micro-Batching**: Asynchronous request batching (`MicroBatcher`) optimizing embedding & reranking GPU/CPU throughput.
* **🛡️ Security Guardrails**: Prompt injection detection, out-of-domain (OOD) query rejection, and citation grounding checks.
* **📊 Official RAGAS Quality Evaluation**: 20-question golden dataset evaluation (`civil_code_eval_20.json`) measuring Faithfulness, Relevance, Context Recall, and Context Precision logged to MLflow.
* **📈 Prometheus & Grafana Observability**: Live metrics exporter (`/metrics`) exposing P95/P99 latencies, guardrail rejections, and context recall gauge in pre-provisioned Grafana dashboards.

---

## 🌐 Production Ecosystem Services

| Service | Endpoint / URL | Purpose | Credentials |
|---------|---------------|---------|-------------|
| **FastAPI Serving API** | `http://localhost:8000` | Sync Q&A, SSE Streaming, Health, Metrics | N/A |
| **Interactive API Docs** | `http://localhost:8000/docs` | Swagger UI for testing endpoints | N/A |
| **Prometheus Server** | `http://localhost:9090` | Time-series metrics scraping engine | N/A |
| **Grafana Dashboards** | `http://localhost:3000` | Live visual observability dashboards | `admin` / `admin` |
| **MLflow Tracking UI** | `http://localhost:5000` | Experiment tracking & model registry | N/A |

---

## 📡 API Reference & Example `curl` Commands

### 1. Health Probe (`GET /health`)

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "documents_indexed": 2296,
  "version": "1.0.0",
  "uptime_seconds": 124.5
}
```

### 2. Synchronous Legal Q&A (`POST /ask`)

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "ما هي أحكام فسخ العقد في القانون المدني المصري؟", "top_k": 3}'
```

**Response:**
```json
{
  "question": "ما هي أحكام فسخ العقد في القانون المدني المصري؟",
  "answer": "استناداً إلى نصوص القانون المدني المصري الواردة في (المادة 157، المادة 158)، فإن أحكام المسألة تتحدد بالضوابط المقررة في نصوص المواد المرفقة طيه.",
  "sources": [
    {
      "article_number": 157,
      "citation": "المادة 157",
      "text_snippet": "في العقود الملزمة للجانبين، إذا لم يف أحد المتعاقدين بالتزامه جاز للمتعاقد الآخر...",
      "score": 0.8942
    }
  ],
  "model_version": "1.0.0",
  "latency_ms": 142.18
}
```

### 3. SSE Real-Time Streaming (`POST /ask/stream`)

```bash
curl -N -X POST "http://localhost:8000/ask/stream" \
     -H "Content-Type: application/json" \
     -d '{"question": "ما هي شروط صحة عقد البيع؟", "top_k": 3}'
```

**Streamed Response Frames:**
```
event: sources
data: [{"article_number": 418, "citation": "المادة 418", ...}]

event: chunk
data: استناداً إلى نصوص القانون المدني المصري

event: chunk
data: الواردة في (المادة 418)...

event: done
data: {"latency_ms": 156.42}
```

### 4. Prometheus Metrics Exporter (`GET /metrics`)

```bash
curl -X GET "http://localhost:8000/metrics"
```

---

## 📁 Repository Structure

```
egyptian-legal-rag/
├── benchmarks/               # Load & stress testing (Locust + Latency profiler)
│   ├── locustfile.py         # Concurrent user traffic benchmark
│   └── profiler.py           # Latency percentile profiler (P50, P95, P99)
├── data/
│   ├── evaluation/           # 20-question Golden Dataset (civil_code_eval_20.json)
│   ├── raw/                  # Raw Legal PDF documents (Law.pdf)
│   └── processed/            # Structured JSON data & vector embeddings store
├── docker/                   # Container & Observability configurations
│   ├── Dockerfile            # Multi-stage production build using uv
│   ├── prometheus.yml        # Prometheus scrape target configuration
│   └── grafana/              # Pre-provisioned Grafana datasource & dashboard JSON
├── reports/                  # Engineering & Evaluation reports (Modules 1–5)
│   ├── module-1.md           # Module 1: Packaging & Baseline report
│   ├── module-2.md           # Module 2: Tracking & DVC report
│   ├── module-3.md           # Module 3: High-Performance Serving report
│   ├── module-4.md           # Module 4: Golden Dataset & Ragas Eval report
│   └── module-5.md           # Module 5: Observability & Operational Playbook
├── src/
│   ├── api/                  # FastAPI Application Layer
│   │   ├── batching.py       # Adaptive micro-batching engine
│   │   ├── main.py           # Production REST server & SSE streaming
│   │   └── schemas.py        # Request/Response Pydantic schemas
│   └── rag/                  # Core RAG Architecture
│       ├── chunking.py       # Legal article chunking logic
│       ├── clean_dataset.py  # Arabic text normalization
│       ├── embeddings.py     # Multilingual embedding service
│       ├── guardrails.py     # Input/Output security & OOD guardrails
│       ├── ingestion.py      # PDF parsing & text extraction
│       ├── ragas_eval.py     # Official Ragas evaluation pipeline
│       ├── registry.py       # MLflow Model Registry helpers
│       ├── reranking.py      # Cross-encoder re-ranking module
│       └── retrieval.py      # Dense vector retriever
├── tests/                    # Unit and integration test suite (pytest)
├── docker-compose.yml        # Full ecosystem orchestration (API, Prometheus, Grafana, MLflow)
├── dvc.yaml                  # DVC data pipeline specification
├── pyproject.toml            # Project metadata & dependency definitions
└── README.md                 # Project documentation & quickstart
```

---

## 🧪 Testing & Quality Assurance

Run the test suite with coverage reporting (enforcing `>= 70%` coverage gate):

```bash
uv run pytest
```

Run Ruff linter and formatter:

```bash
uv run ruff check .
uv run ruff format .
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
