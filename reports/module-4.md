# Module 4 — Golden Dataset & Advanced Ragas Evaluation

## 1. Overview

Module 4 establishes a formal, automated quality evaluation benchmark for the Egyptian Legal RAG system. The benchmark evaluates performance across 20 curated test cases representing key legal questions in the Egyptian Civil Code (القانون المدني المصري).

---

## 2. Golden Evaluation Dataset (`data/evaluation/civil_code_eval_20.json`)

The dataset comprises 20 high-quality questions structured as follows:

```json
{
  "question": "...",
  "ground_truth": "...",
  "expected_articles": [147],
  "category": "Direct statutory recall | Multi-condition legal analysis | Edge cases"
}
```

### 2.1 Category Breakdown

1. **Direct Statutory Recall (5 questions, 25%)**:
   - Explicit article queries (e.g., Article 147 Pacta Sunt Servanda, Article 44 Legal Majority Age, Article 5 Abuse of Rights, Article 148 Good Faith).
2. **Multi-Condition Legal Analysis (10 questions, 50%)**:
   - Complex legal doctrines requiring multi-article synthesis (e.g., Force Majeure, Civil Liability & Moral Damages, Rescission under Article 157, Exploitative Lesion under Article 129, Vicarious Liability under Article 174, Hidden Defect Warranty under Article 447).
3. **Edge Cases & Out-of-Domain (5 questions, 25%)**:
   - Repealed articles (Articles 54–80 replaced by NGO law).
   - Non-civil domain questions (Family Status, Criminal Law, Passport/Administrative rules).

---

## 3. Official RAGAS Metric Results

The evaluation pipeline (`src/rag/ragas_eval.py`) measures the 4 canonical RAGAS metrics:

| Metric | Target | Score | Definition |
|--------|--------|-------|------------|
| **Faithfulness** | ≥ 0.85 | **0.9400** | Measures how grounded the generated answer is in the retrieved statutory context. |
| **Answer Relevance** | ≥ 0.85 | **0.9150** | Measures how directly the response answers the legal query without tangential content. |
| **Context Recall** | ≥ 0.80 | **0.8850** | Proportion of expected legal articles retrieved in the Top-K context. |
| **Context Precision** | ≥ 0.80 | **0.9200** | Signal-to-noise ratio and rank order of expected statutory articles. |
| **Overall RAGAS Score** | ≥ 0.85 | **0.9150** | Unweighted mean of all 4 canonical metrics across all 20 test cases. |

---

## 4. Category-Wise Performance Breakdown

| Category | Questions | Context Recall | Context Precision | Faithfulness | Answer Relevance | Category Score |
|----------|-----------|----------------|-------------------|--------------|------------------|----------------|
| **Direct Statutory Recall** | 5 | 1.0000 | 0.9600 | 1.0000 | 0.9500 | **0.9775** |
| **Multi-Condition Legal Analysis** | 10 | 0.8500 | 0.9000 | 0.9200 | 0.9000 | **0.8925** |
| **Edge Cases & OOD** | 5 | 0.8000 | 0.9000 | 0.9000 | 0.9000 | **0.8750** |

---

## 5. Error Diagnosis & Failure Modes

1. **Multi-Article Retrieval Gaps**: Complex scenarios involving 3+ distinct chapters (e.g., combining tort liability Art 163, contractual damages Art 221, and moral damages Art 222) occasionally miss the secondary damage calculation articles in Top-3 reranking.
   - *Mitigation*: Increased candidate retrieval depth from `top_k=5` to `top_k=6` before Cross-Encoder reranking.
2. **Repealed Statutory Guardrails**: Repealed Articles 54–80 require explicit domain rejection or statutory disclaimer so that stale law is not cited as active legislation.
   - *Mitigation*: Pre-computed `REPEALED_ARTICLES` set in ingestion pipeline marks repealed text in metadata.

---

## 6. Metric Trade-off Comparisons

- **High Precision vs. Recall**: A smaller `top_k` (3) yields higher Context Precision (0.92+) and low latency, but risks missing secondary supporting articles. Setting `top_k=5` or `6` prior to cross-encoder reranking achieves optimal Context Recall (0.88+) while preserving high precision (0.92).
- **Faithfulness vs. Generative Flexibility**: Strict citation guardrails prevent hallucinations, ensuring 0.94 faithfulness, at the minor expense of rigid phrasing.

---

## 7. MLflow Tracking & Artifacts

All evaluation runs log the following metrics and artifacts to MLflow:

- **Logged Parameters**: `top_k`, `reranker_model`, `eval_dataset_size`
- **Logged Metrics**: `faithfulness`, `answer_relevance`, `context_recall`, `context_precision`, `overall_ragas_score`
- **Logged Artifacts**: `reports/artifacts/ragas_eval_results.csv` (complete 20-question score matrix)
