# Module 5 — Production Observability, Monitoring & Final Delivery

## 1. Observability Architecture & Stack

The Egyptian Legal RAG monitoring ecosystem provides real-time observability across system performance, guardrail rejections, and legal retrieval quality metrics.

```
+------------------+         +------------------+         +-------------------+
|  RAG Serving API | ------> |    Prometheus    | ------> |      Grafana      |
|   (:8000/metrics)|  scrape |      (:9090)     |         |      (:3000)      |
+------------------+         +------------------+         +-------------------+
          |                                                         |
          v                                                         v
+------------------+                                      +-------------------+
|    MLflow UI     |                                      | Pre-provisioned   |
|     (:5000)      |                                      | RAG Dashboard     |
+------------------+                                      +-------------------+
```

---

## 2. Prometheus Metrics Instrumentations

The RAG API exports standard Prometheus metrics at `GET /metrics`:

| Metric Name | Type | Labels / Buckets | Operational Purpose |
|-------------|------|------------------|---------------------|
| `http_request_duration_seconds` | Histogram | `le="0.05", "0.1", "0.25", "0.5", "1.0", "2.5", "5.0"` | Tracks end-to-end P50, P95, P99 request latencies. |
| `rag_guardrail_rejections_total` | Counter | N/A | Counts prompt injections and out-of-domain (OOD) rejections. |
| `rag_context_recall_estimate` | Gauge | N/A | Monitors real-time estimated context recall metric (target ≥ 0.80). |

---

## 3. Pre-Provisioned Grafana Dashboard (`docker/grafana/`)

The Grafana instance automatically loads the **Egyptian Legal RAG — Production Observability Dashboard** (`rag_dashboard.json`) containing 4 core visual panels:

1. **API Request Throughput (RPS)** — `rate(http_request_duration_seconds_count[1m])`
2. **P95 & P99 Latency (Seconds)** — `histogram_quantile(0.95, ...)` and `histogram_quantile(0.99, ...)`
3. **Guardrail & OOD Rejections Total** — `rag_guardrail_rejections_total`
4. **Estimated Context Recall Gauge** — `rag_context_recall_estimate` with threshold colors (Green ≥ 0.85, Yellow ≥ 0.70, Red < 0.70)

---

## 4. Operational Playbook & Drift Detection Plan

### 4.1 Data & Model Drift Detection

1. **Concept Drift**: Legal amendments to the Egyptian Civil Code or newly enacted complementary laws (e.g., NGO law replacing Articles 54–80).
   - *Detection Strategy*: Automated weekly scheduled run of `src/rag/ragas_eval.py`. If Context Recall falls below 0.80 or Faithfulness below 0.85, an alert is logged.
   - *Mitigation Plan*: Trigger DVC pipeline `dvc repro` to re-ingest, re-chunk, and re-index updated legal texts.

2. **Query Distribution Drift**: Influx of new legal terminology or non-civil law inquiries.
   - *Detection Strategy*: Monitor `rag_guardrail_rejections_total` spike in Grafana. If rejection rate exceeds 15% over a 10-minute window, investigate query logs.

### 4.2 Incident Response Matrix

| Alert Trigger | Root Cause | Immediate Remediation Action |
|---------------|------------|------------------------------|
| P95 Latency > 2.0s | High concurrent traffic or large cross-encoder batching queues | Scale replica count in Docker/K8s; adjust micro-batch `max_wait_ms` from 30ms to 15ms. |
| Health Check Failing | Unhandled memory pressure or vector store loading error | Container automatically restarts via `HEALTHCHECK` policy; inspect container logs via `docker logs`. |
| Context Recall < 0.80 | Reranking truncation or vocabulary mismatch | Re-tune dense retriever `top_k` candidates from 6 to 10; trigger model evaluation run. |

---

## 5. End-to-End Release & Verification Checklist

- [x] **Module 3**: FastAPI async serving, SSE streaming `/ask/stream`, micro-batching engine, multi-stage Dockerfile, Locust load benchmark, report `reports/module-3.md`.
- [x] **Module 4**: 20-question golden dataset `civil_code_eval_20.json`, official Ragas evaluation pipeline `src/rag/ragas_eval.py`, MLflow metric logging, report `reports/module-4.md`.
- [x] **Module 5**: Prometheus metrics exporter `/metrics`, Grafana provisioning & dashboard JSON, full ecosystem orchestration via `docker-compose.yml`, report `reports/module-5.md`.
- [x] **Testing & Coverage**: All unit and integration tests passing with `>= 70%` coverage.
- [x] **Release Tagging**: Code merged to `main` and tagged as release `v1.0.0`.
