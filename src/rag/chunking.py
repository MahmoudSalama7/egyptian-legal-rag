from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_legal_chunks(
    input_path: str = "data/processed/civil_code_ready.json",
    output_path: str = "data/processed/civil_code_chunks.json",
    include_english: bool = True,
) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    chunks = []

    for art in articles:
        art_no = art["article_number"]
        text_ar = art["text_ar"].strip()
        text_en = (art.get("text_en") or "").strip()
        is_repealed = art.get("is_repealed", False)

        # Chunk العربي للمادة
        chunk_ar = {
            "chunk_id": f"art_{art_no}_ar",
            "article_number": art_no,
            "language": "ar",
            "text": f"المادة {art_no}:\n{text_ar}",
            "metadata": {
                "article_number": art_no,
                "is_repealed": is_repealed,
                "citation": art.get(
                    "citation", f"Egyptian Civil Code, Article {art_no}"
                ),
                "source_page": art.get("source_page"),
                "has_translation": bool(text_en),
            },
        }
        chunks.append(chunk_ar)

        # Chunk الإنجليزي المقابل (إذا كان مطلوباً)
        if include_english and text_en:
            chunk_en = {
                "chunk_id": f"art_{art_no}_en",
                "article_number": art_no,
                "language": "en",
                "text": f"Article {art_no}:\n{text_en}",
                "metadata": {
                    "article_number": art_no,
                    "is_repealed": is_repealed,
                    "citation": art.get(
                        "citation", f"Egyptian Civil Code, Article {art_no}"
                    ),
                    "source_page": art.get("source_page"),
                    "has_translation": True,
                },
            }
            chunks.append(chunk_en)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(chunks)} chunks from {len(articles)} articles.")
    print(f"Saved to: {output_path}")
    return chunks


if __name__ == "__main__":
    create_legal_chunks()
