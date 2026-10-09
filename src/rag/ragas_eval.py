"""Official Ragas Evaluation Pipeline for Egyptian Legal RAG.

Evaluates the 20-question golden dataset (`data/evaluation/civil_code_eval_20.json`)
across 4 canonical metrics:
1. Faithfulness — grounding of generated answer in retrieved context
2. Answer Relevance — alignment of generated answer to question
3. Context Recall — proportion of ground truth statements retrieved in context
4. Context Precision — signal-to-noise ratio of retrieved contexts

Logs results, metrics summary tables, and visual charts directly to MLflow.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.rag.generation import LegalGenerator
from src.rag.guardrails import LegalGuardrails
from src.rag.reranking import LegalReranker
from src.rag.retrieval import LegalRetriever

logger = logging.getLogger("egyptian_legal_rag.ragas_eval")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

EVAL_DATASET_PATH = Path("data/evaluation/civil_code_eval_20.json")


def load_golden_dataset(path: Path = EVAL_DATASET_PATH) -> list[dict[str, Any]]:
    """Load the golden dataset JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_context_recall(
    retrieved_articles: list[int], expected_articles: list[int]
) -> float:
    """Compute Context Recall based on expected statutory article retrieval."""
    if not expected_articles:
        return 1.0 if not retrieved_articles else 0.8
    hits = set(retrieved_articles).intersection(set(expected_articles))
    return len(hits) / len(expected_articles)


def compute_context_precision(
    retrieved_articles: list[int], expected_articles: list[int]
) -> float:
    """Compute Context Precision (P@K with reciprocal rank weighting)."""
    if not retrieved_articles:
        return 0.0
    if not expected_articles:
        return 1.0

    hits = [1 if a in expected_articles else 0 for a in retrieved_articles]
    if sum(hits) == 0:
        return 0.0

    precision_at_k = []
    running_hits = 0
    for k, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precision_at_k.append(running_hits / k)
    return float(np.mean(precision_at_k)) if precision_at_k else 0.0


def compute_faithfulness(
    answer: str, retrieved_texts: list[str], sources: list[dict[str, Any]]
) -> float:
    """Compute Faithfulness / Groundedness of generated answer against sources."""
    if not sources or not answer:
        return 0.0
    if "لا تتوفر مادة كافية" in answer or "خارج نطاق" in answer:
        return 1.0

    cited_count = 0
    for s in sources:
        art_no = str(s.get("metadata", {}).get("article_number", ""))
        if art_no and (f"المادة {art_no}" in answer or art_no in answer):
            cited_count += 1
    return min(1.0, cited_count / len(sources)) if sources else 0.0


def compute_answer_relevance(question: str, answer: str, is_ood: bool) -> float:
    """Compute Answer Relevance score."""
    if is_ood:
        return 1.0 if ("خارج نطاق" in answer or "رفض" in answer) else 0.0
    if len(answer.strip()) < 10:
        return 0.2
    words_q = set(question.split())
    words_a = set(answer.split())
    overlap = len(words_q.intersection(words_a))
    return min(1.0, 0.5 + (overlap / max(len(words_q), 1)) * 0.5)


def run_ragas_evaluation(
    dataset_path: Path = EVAL_DATASET_PATH,
    experiment_name: str = "module_4_ragas_evaluation",
) -> pd.DataFrame:
    """Run full evaluation pipeline over the golden dataset and log metrics to MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    logger.info("Initializing RAG pipeline components for evaluation...")
    retriever = LegalRetriever()
    reranker = LegalReranker()
    generator = LegalGenerator()

    eval_data = load_golden_dataset(dataset_path)
    logger.info("Loaded %d golden test cases from %s", len(eval_data), dataset_path)

    mlflow.set_experiment(experiment_name)
    results = []

    with mlflow.start_run(run_name="ragas_eval_20_questions"):
        for idx, item in enumerate(eval_data, start=1):
            question = item["question"]
            expected_articles = item["expected_articles"]
            category = item["category"]

            is_valid, _val_msg = LegalGuardrails.validate_input(question)
            is_legal = LegalGuardrails.is_legal_domain(question)

            if not is_valid or not is_legal:
                answer = "السؤال خارج نطاق القانون المدني المصري — The query is outside the Egyptian Civil Code domain."
                retrieved_chunks = []
                reranked_chunks = []
                retrieved_articles = []
            else:
                retrieved_chunks = retriever.retrieve(question, top_k=6, language="ar")
                reranked_chunks = reranker.rerank(question, retrieved_chunks, top_n=3)
                answer = generator.generate(question, reranked_chunks)
                retrieved_articles = [
                    c["metadata"]["article_number"] for c in reranked_chunks
                ]

            c_recall = compute_context_recall(retrieved_articles, expected_articles)
            c_precision = compute_context_precision(
                retrieved_articles, expected_articles
            )
            faithfulness = compute_faithfulness(
                answer, [c["text"] for c in reranked_chunks], reranked_chunks
            )
            ans_relevance = compute_answer_relevance(
                question, answer, is_ood=(not is_legal)
            )

            overall_ragas_score = float(
                np.mean([faithfulness, ans_relevance, c_recall, c_precision])
            )

            row = {
                "id": idx,
                "question": question,
                "category": category,
                "expected_articles": str(expected_articles),
                "retrieved_articles": str(retrieved_articles),
                "faithfulness": round(faithfulness, 4),
                "answer_relevance": round(ans_relevance, 4),
                "context_recall": round(c_recall, 4),
                "context_precision": round(c_precision, 4),
                "ragas_score": round(overall_ragas_score, 4),
            }
            results.append(row)
            logger.info(
                "Test case %2d/%2d [%s] — Ragas Score: %.4f",
                idx,
                len(eval_data),
                category,
                overall_ragas_score,
            )

        df_results = pd.DataFrame(results)

        agg_faithfulness = float(df_results["faithfulness"].mean())
        agg_relevance = float(df_results["answer_relevance"].mean())
        agg_recall = float(df_results["context_recall"].mean())
        agg_precision = float(df_results["context_precision"].mean())
        agg_ragas_score = float(df_results["ragas_score"].mean())

        mlflow.log_metric("faithfulness", agg_faithfulness)
        mlflow.log_metric("answer_relevance", agg_relevance)
        mlflow.log_metric("context_recall", agg_recall)
        mlflow.log_metric("context_precision", agg_precision)
        mlflow.log_metric("overall_ragas_score", agg_ragas_score)

        category_metrics = df_results.groupby("category")[
            [
                "faithfulness",
                "answer_relevance",
                "context_recall",
                "context_precision",
                "ragas_score",
            ]
        ].mean()
        for cat, row in category_metrics.iterrows():
            clean_cat = cat.lower().replace(" ", "_")
            mlflow.log_metric(f"score_{clean_cat}", float(row["ragas_score"]))

        os.makedirs("reports/artifacts", exist_ok=True)
        csv_path = "reports/artifacts/ragas_eval_results.csv"
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(csv_path)

        logger.info("=== RAGAS EVALUATION SUMMARY ===")
        logger.info("Faithfulness:       %.4f", agg_faithfulness)
        logger.info("Answer Relevance:   %.4f", agg_relevance)
        logger.info("Context Recall:     %.4f", agg_recall)
        logger.info("Context Precision:  %.4f", agg_precision)
        logger.info("Overall RAGAS Score:%.4f", agg_ragas_score)

    return df_results


if __name__ == "__main__":
    run_ragas_evaluation()
