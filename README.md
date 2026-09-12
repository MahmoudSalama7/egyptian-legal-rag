# ⚖️ Egyptian Legal RAG (محرك الاسترجاع والتوليد المعزز للقانون المدني المصري)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **production-ready, bilingual (Arabic & English)** Retrieval-Augmented Generation (RAG) system specifically designed for the **Egyptian Civil Code (القانون المدني المصري)**.

This system provides precise legal document retrieval, cross-encoder re-ranking, citation-grounded generation, input/output security guardrails, and enterprise tracking via MLflow & DVC.

---

## 🌟 Key Features

* **📜 Legal Data Ingestion & Normalization**: Automated PDF ingestion (`PyMuPDF`/`pypdf`), text normalization, and article-aware chunking preserving legal article numbers (`المادة`).
* **🔍 Dense Multilingual Vector Retrieval**: Sentence-transformer based retrieval tailored for legal domain Arabic & English queries (`paraphrase-multilingual-MiniLM-L12-v2`).
* **🎯 Cross-Encoder Reranking**: Re-ranking stage using multilingual Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) to maximize relevance and minimize precision loss.
* **🛡️ Security & Anti-Hallucination Guardrails**:
  * **Input Validation**: Prompt injection detection and domain scope checking.
  * **Output Validation**: Strict citation verification ensuring response grounding.
* **⚖️ Citation-Grounded Legal Generator**: Formulates formal legal answers linked directly to Egyptian Civil Code articles.
* **🚀 Production REST API**: FastAPI backend with async lifecycle management, `/ask` and `/health` endpoints.
* **📊 MLOps & Experimentation**:
  * **DVC Pipeline**: Reproducible ingestion, cleaning, chunking, and indexing.
  * **MLflow Tracking**: Experiment logging for chunking strategies, embeddings, and retrieval metrics.
* **🧪 Comprehensive Testing**: Full test coverage suite using `pytest` and code linting via `ruff`.

---

## 📁 Repository Structure

```
egyptian-legal-rag/
├── data/
│   ├── raw/                  # Raw Legal PDF documents (Law.pdf)
│   └── processed/            # Structured JSON data & vector embeddings index
├── docker/                   # Deployment & container configurations
├── serving/                  # Serving orchestration (BentoML, Nginx, vLLM)
├── reports/                  # Pipeline validation & evaluation reports
├── src/
│   ├── api/                  # FastAPI Application
│   │   ├── main.py           # API endpoints & app lifespan
│   │   └── schemas.py        # Request & Response Pydantic models
│   └── rag/                  # Core RAG Engine
│       ├── chunking.py       # Legal article chunking logic
│       ├── clean_dataset.py  # Arabic text normalization & cleanup
│       ├── config.py         # System configuration & environment vars
│       ├── embeddings.py     # SentenceTransformers embedding service
│       ├── evaluation.py     # Ragas evaluation integration
│       ├── experiments.py    # MLflow experiment tracking runner
│       ├── generation.py     # Legal prompt synthesis & LLM generator
│       ├── guardrails.py     # Security guardrails & domain filters
│       ├── ingestion.py      # PDF parsing & text extraction
│       ├── registry.py       # Model & artifact registry helpers
│       ├── reranking.py      # Cross-encoder re-ranking module
│       ├── retrieval.py      # Dense retriever & vector search
│       ├── server.py         # MLflow UI runner script
│       └── validate.py       # Schema & data pipeline validators
├── tests/                    # Unit and integration test suite
├── dvc.yaml                  # DVC data pipeline stages
├── pyproject.toml            # Dependencies and project metadata
└── .env.example              # Environment variable configuration template
```

---

## ⚙️ Quick Start & Installation

### Prerequisites

* Python `>= 3.10`
* [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/MahmoudSalama7/egyptian-legal-rag.git
cd egyptian-legal-rag
```

### 2. Environment Setup

Copy `.env.example` to `.env` and set any relevant API keys or server configuration:

```bash
cp .env.example .env
```

### 3. Install Dependencies

Using **`uv`**:
```bash
uv sync --extra dev
```

Or using **`pip`**:
```bash
pip install -e ".[dev]"
```

---

## 🔄 Running the DVC Data Pipeline

Run the end-to-end data ingestion, dataset cleaning, chunking, and vector indexing pipeline:

```bash
uv run dvc repro
```

This runs the following sequential pipeline stages:
1. `ingest`: Extracts raw legal text from `data/raw/Law.pdf`.
2. `clean`: Normalizes Arabic text and strips noise into `data/processed/civil_code_ready.json`.
3. `chunk`: Splits text into legal article units in `data/processed/civil_code_chunks.json`.
4. `index`: Generates vector embeddings stored in `data/processed/index_store/vectors.npy`.

---

## 🚀 Running the API & MLflow UI

### Start the FastAPI REST Server

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
* Swagger UI: `http://localhost:8000/docs`
* Redoc: `http://localhost:8000/redoc`

### Start MLflow Experiment UI

```bash
uv run python -m src.rag.server
```
Access the tracking dashboard at `http://localhost:5000`.

---

## 📡 API Reference

### Health Check (`GET /health`)

**Response:**
```json
{
  "status": "healthy",
  "documents_indexed": 1050,
  "version": "0.1.0"
}
```

### Ask Legal Question (`POST /ask`)

**Request Payload:**
```json
{
  "question": "ما هي أحكام فسخ العقد في القانون المدني المصري؟",
  "top_k": 3
}
```

**Response Payload:**
```json
{
  "question": "ما هي أحكام فسخ العقد في القانون المدني المصري؟",
  "answer": "استناداً إلى نصوص القانون المدني المصري الواردة في (المادة 157، المادة 158)...",
  "sources": [
    {
      "article_number": "157",
      "citation": "المادة 157",
      "text_snippet": "في العقود الملزمة للجانبين، إذا لم يف أحد المتعاقدين بالتزامه جاز للمتعاقد الآخر...",
      "score": 0.8942
    }
  ],
  "model_version": "0.1.0"
}
```

---

## 🧪 Testing & Quality Assurance

Run the test suite with coverage reporting:

```bash
uv run pytest
```

Run linting checks with **Ruff**:

```bash
uv run ruff check .
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
