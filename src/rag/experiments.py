from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
import mlflow

from src.rag.retrieval import LegalRetriever
from src.rag.reranking import LegalReranker
from src.rag.guardrails import LegalGuardrails
from src.rag.generation import LegalGenerator

EVAL_GROUND_TRUTH = [
    {"question": "ما هو مبدأ العقد شريعة المتعاقدين؟", "expected_article": 147},
    {"question": "متى يكون استعمال الحق غير مشروع والتعسف فيه؟", "expected_article": 5},
    {"question": "ما هو سن الرشد القانوني؟", "expected_article": 44},
    {"question": "كيف يتم إثبات الوفاة والولادة رسميا؟", "expected_article": 30},
    {"question": "ما حكم تصرفات عديم التمييز؟", "expected_article": 45},
]


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def run_experiment(
    run_name: str,
    top_k: int,
    use_reranker: bool,
    embedding_model: str,
    prompt_version: str = "v1.0",
):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("legal_rag_retrieval_and_eval")

    retriever = LegalRetriever()
    reranker = LegalReranker() if use_reranker else None
    generator = LegalGenerator()

    start_time = time.time()
    hits = 0
    total_latency = 0.0
    detailed_results = []

    for item in EVAL_GROUND_TRUTH:
        q = item["question"]
        expected = item["expected_article"]

        # 1. فحص حواجز الأمان
        is_valid, _ = LegalGuardrails.validate_input(q)
        if not is_valid:
            continue

        # 2. الاسترجاع وإعادة الترتيب
        t0 = time.time()
        candidates = retriever.retrieve(query=q, top_k=top_k, language="ar")

        if reranker and candidates:
            final_sources = reranker.rerank(
                query=q, candidate_chunks=candidates, top_n=min(3, top_k)
            )
        else:
            final_sources = candidates[:3]

        query_latency = time.time() - t0
        total_latency += query_latency

        # 3. بناء البرومبت والتوليد
        prompt_used = generator.build_prompt(q, final_sources)
        generated_answer = generator.generate(q, final_sources)

        # 4. حساب دقة استرجاع المادة المستهدفة
        retrieved_articles = [c["metadata"]["article_number"] for c in final_sources]
        is_hit = expected in retrieved_articles
        if is_hit:
            hits += 1

        detailed_results.append(
            {
                "question": q,
                "expected_article": expected,
                "retrieved_articles": retrieved_articles,
                "hit": is_hit,
                "latency_sec": round(query_latency, 4),
                "generated_answer": generated_answer,
                "prompt_sample": prompt_used,
            }
        )

    duration = time.time() - start_time
    hit_rate = hits / len(EVAL_GROUND_TRUTH)
    avg_latency = total_latency / len(EVAL_GROUND_TRUTH)

    # بدء تسجيل التجربة في MLflow
    with mlflow.start_run(run_name=run_name):
        # 1. المعاملات الأساسية (Parameters)
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("use_reranker", use_reranker)
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("eval_samples_count", len(EVAL_GROUND_TRUTH))

        # 2. القياسات (Metrics)
        mlflow.log_metric("hit_rate_context_recall", hit_rate)
        mlflow.log_metric("avg_latency_sec", avg_latency)
        mlflow.log_metric("total_time_sec", duration)

        # 3. الوسوم والربط بالـ Git (Tags & Lineage)
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("track", "Track_B_Legal_RAG")
        mlflow.set_tag("pipeline_stage", "retrieval_and_rerank")

        # 4. حفظ الـ Prompt والنتائج كـ Artifacts
        artifacts_dir = Path("reports/temp_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # حفظ قالب البرومبت النصي
        prompt_file = artifacts_dir / "system_prompt_template.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(generator.build_prompt("{query}", []))
        mlflow.log_artifact(str(prompt_file), artifact_path="prompts")

        # حفظ جدول الأسئلة والإجابات التفصيلي
        results_file = artifacts_dir / "eval_predictions.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(results_file), artifact_path="evaluation")

        print(
            f"Logged [{run_name}]: Hit Rate = {hit_rate:.2f}, Latency = {avg_latency:.4f}s"
        )


def main():
    configs = [
        {
            "run_name": "run_1_top2_no_rerank",
            "top_k": 2,
            "use_reranker": False,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "run_name": "run_2_top5_no_rerank",
            "top_k": 5,
            "use_reranker": False,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "run_name": "run_3_top5_with_reranker",
            "top_k": 5,
            "use_reranker": True,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "run_name": "run_4_top10_no_rerank",
            "top_k": 10,
            "use_reranker": False,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        {
            "run_name": "run_5_top10_with_reranker",
            "top_k": 10,
            "use_reranker": True,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
    ]

    for cfg in configs:
        run_experiment(**cfg)


if __name__ == "__main__":
    main()
