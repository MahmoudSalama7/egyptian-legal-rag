import json

from src.rag.validate import validate_articles


def test_validate_articles_valid(tmp_path):
    input_file = tmp_path / "valid_articles.json"
    sample_articles = [
        {
            "article_number": 1,
            "text_ar": "تسرى النصوص التشريعية على جميع المسائل في اللفظ الفحوى.",
            "text_en": "Provisions of laws govern all matters in letter or spirit.",
            "is_repealed": False,
            "citation": "Egyptian Civil Code, Article 1",
            "source_page": 1,
        }
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_articles, f, ensure_ascii=False)

    assert validate_articles(input_file) is True


def test_validate_articles_invalid(tmp_path):
    input_file = tmp_path / "invalid_articles.json"
    sample_articles = [
        {
            "article_number": -1,
            "text_ar": "",
            "text_en": "",
            "is_repealed": False,
            "citation": "Wrong Citation",
            "source_page": 0,
        }
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_articles, f, ensure_ascii=False)

    assert validate_articles(input_file) is False
