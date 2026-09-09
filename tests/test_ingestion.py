from src.rag.ingestion import (
    MAX_LEGAL_ARTICLE,
    REPEALED_ARTICLES,
    clean_arabic_article_text,
    clean_text,
    match_english_article_header,
    normalize_digits,
)


def test_normalize_digits():
    assert normalize_digits("المادة ١٤٧") == "المادة 147"
    assert normalize_digits("١٢٣٤٥٦٧٨٩٠") == "1234567890"


def test_clean_text():
    assert (
        clean_text("  العقد   شريعة \u00a0 المتعاقدين \r\n") == "العقد شريعة المتعاقدين"
    )
    assert clean_text("سطر أول\nا\nسطر ثان") == "سطر أول\nسطر ثان"


def test_match_english_headers():
    assert match_english_article_header("Article 1") == 1
    assert match_english_article_header("Article 10") == 10
    assert match_english_article_header("Article 134") == 134
    assert match_english_article_header("Article 147") == 147
    assert match_english_article_header("Articles 54-80") is None
    assert match_english_article_header("Random text") is None
    assert match_english_article_header(f"Article {MAX_LEGAL_ARTICLE + 10}") is None


def test_clean_arabic_article_text():
    raw_sample = "مادة ( )١٠\nالقانون المصري هو المرجع في تكييف العلاقات..."
    cleaned = clean_arabic_article_text(raw_sample, 10)
    assert "مادة" not in cleaned
    assert "القانون المصري هو المرجع" in cleaned

    header_with_garbage = "نصوص القانون المدنى\nمادة 5 نص العقد"
    cleaned2 = clean_arabic_article_text(header_with_garbage, 5)
    assert "نص العقد" in cleaned2


def test_repealed_articles():
    assert 54 in REPEALED_ARTICLES
    assert 80 in REPEALED_ARTICLES
    assert 389 in REPEALED_ARTICLES
    assert 1 not in REPEALED_ARTICLES
