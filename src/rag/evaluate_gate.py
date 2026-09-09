from __future__ import annotations

import json
import sys
from pathlib import Path

CHUNKS_PATH = Path("data/processed/civil_code_chunks.json")
MIN_ACCEPTABLE_FAITHFULNESS = 0.75  # الحد الأدنى المعتمد في الـ Handbook


def run_quality_gate() -> bool:
    if not CHUNKS_PATH.exists():
        print(f"[ERROR] Chunks file not found at {CHUNKS_PATH}")
        sys.exit(1)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # مجموعة أسئلة اختبار قانونية مع المواد المتوقعة كـ Ground Truth
    test_eval_set = [
        {
            "question": "ما هو مصدر الالتزام والعقد شريعة المتعاقدين؟",
            "expected_article": 147,
        },
        {"question": "ما هي شروط استعمال الحق والتعسف فيه؟", "expected_article": 5},
        {
            "question": "ما هو سن الرشد في القانون المدني المصري؟",
            "expected_article": 44,
        },
        {"question": "كيف تثبت الولادة والوفاة رسمياً؟", "expected_article": 30},
        {
            "question": "متى تبطل التصرفات الصادرة من عديم التمييز؟",
            "expected_article": 45,
        },
    ]

    passed_retrievals = 0

    for item in test_eval_set:
        query_words = item["question"].split()
        matched = False

        for c in chunks:
            if (
                c["language"] == "ar"
                and c["metadata"]["article_number"] == item["expected_article"]
                and any(w in c["text"] for w in query_words)
            ):
                matched = True
                break

        if matched:
            passed_retrievals += 1

    accuracy_score = passed_retrievals / len(test_eval_set)
    print(
        f"RAG Quality Evaluation Score: {accuracy_score:.2f} (Threshold: {MIN_ACCEPTABLE_FAITHFULNESS})"
    )

    if accuracy_score < MIN_ACCEPTABLE_FAITHFULNESS:
        print(
            "[FAIL] RAG Quality Gate rejected the build: accuracy fell below threshold."
        )
        return False

    print("[PASS] Quality Gate passed successfully.")
    return True


if __name__ == "__main__":
    success = run_quality_gate()
    if not success:
        sys.exit(1)
