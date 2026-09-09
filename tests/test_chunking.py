import json

from src.rag.chunking import create_legal_chunks


def test_create_legal_chunks(tmp_path):
    input_file = tmp_path / "test_articles.json"
    output_file = tmp_path / "test_chunks.json"

    sample_articles = [
        {
            "article_number": 1,
            "text_ar": "تسرى النصوص التشريعية على جميع المسائل.",
            "text_en": "Provisions of laws govern all matters.",
            "is_repealed": False,
            "citation": "Egyptian Civil Code, Article 1",
            "source_page": 1,
        }
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_articles, f, ensure_ascii=False)

    chunks = create_legal_chunks(
        input_path=str(input_file),
        output_path=str(output_file),
        include_english=True,
    )

    assert len(chunks) == 2
    assert chunks[0]["language"] == "ar"
    assert chunks[1]["language"] == "en"
    assert chunks[0]["article_number"] == 1
    assert output_file.exists()
