from __future__ import annotations

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MODEL_REGISTRY_NAME = "EgyptianCivilCodeRAG"
EXPERIMENT_NAME = "legal_rag_retrieval_and_eval"


class LegalRAGWrapper(mlflow.pyfunc.PythonModel):
    """غلاف PyFunc لتسجيل بايبلاين الـ RAG في MLflow Model Registry."""

    def load_context(self, context):
        from src.rag.generation import LegalGenerator
        from src.rag.reranking import LegalReranker
        from src.rag.retrieval import LegalRetriever

        self.retriever = LegalRetriever()
        self.reranker = LegalReranker()
        self.generator = LegalGenerator()

    def predict(self, context, model_input):
        if isinstance(model_input, list):
            queries = model_input
        else:
            queries = model_input.iloc[:, 0].tolist()

        responses = []
        for q in queries:
            cands = self.retriever.retrieve(query=q, top_k=10, language="ar")
            ranked = self.reranker.rerank(query=q, candidate_chunks=cands, top_n=3)
            ans = self.generator.generate(q, ranked)
            responses.append(ans)
        return responses


def register_best_model_to_production():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"[ERROR] Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # استرجاع أفضل Run بحسب أعلى Hit Rate وأقل Latency
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[
            "metrics.hit_rate_context_recall DESC",
            "metrics.avg_latency_sec ASC",
        ],
        max_results=5,
    )

    if not runs:
        print("[ERROR] No runs found.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    hit_rate = best_run.data.metrics.get("hit_rate_context_recall", 0.0)
    print(f"Registering Best Run ID: {best_run_id} (Hit Rate: {hit_rate:.2f})")

    # تسجيل النموذج مباشرة داخل الـ Run الأفضل
    with mlflow.start_run(run_id=best_run_id):
        mlflow.pyfunc.log_model(
            artifact_path="rag_pipeline",
            python_model=LegalRAGWrapper(),
            registered_model_name=MODEL_REGISTRY_NAME,
        )

    latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
    target_version = latest_versions[-1].version

    try:
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=target_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Promoted {MODEL_REGISTRY_NAME} v{target_version} to Stage: Production")
    except Exception:
        pass

    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="production",
        version=target_version,
    )
    print(
        f"Successfully assigned alias 'production' to {MODEL_REGISTRY_NAME} v{target_version}."
    )


if __name__ == "__main__":
    register_best_model_to_production()
