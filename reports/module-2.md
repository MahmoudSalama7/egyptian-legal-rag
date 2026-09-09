# Module 2 Report: Tracking, Versioning & Automation

## 1. DVC Data Lineage
- **Tracked Artifacts**:
  - `data/raw/egyptian_civil_code.pdf.dvc`
  - `data/processed/civil_code_ready.json.dvc`
  - `data/processed/civil_code_chunks.json.dvc`
- **Reproducibility**: Entire pipeline from raw PDF to vector store index executes with `dvc repro`.

## 2. MLflow Experimentation Summary
- **Experiment**: `legal_rag_retrieval_and_eval`
- **Evaluated Parameters**: `top_k`, `use_reranker`, `embedding_model`, `prompt_version`.
- **Primary Metric**: Context Recall (Hit Rate against ground truth legal queries).
- **Outcome**: Combining dense retrieval (`top_k=5`) with cross-encoder reranking achieved the highest hit rate while keeping latency within acceptable thresholds.
- **Registry**: Best pipeline candidate registered under `EgyptianCivilCodeRAG` and transitioned to `Production`.

*(Insert screenshot of MLflow comparison table at `reports/mlflow_comparison.png`)*
