from src.rag.ingestion import (
    normalize_digits,
    clean_text,
    match_english_article_header,
    clean_arabic_article_text,
)


def test_normalize_digits():
    assert normalize_digits("المادة ١٤٧") == "المادة 147"


def test_clean_text():
    assert clean_text("  العقد   شريعة \u00a0 المتعاقدين \r\n") == "العقد شريعة المتعاقدين"


def test_match_english_headers():
    assert match_english_article_header("Article 1") == 1
    assert match_english_article_header("Article 10") == 10
    assert match_english_article_header("Article 134") == 134
    assert match_english_article_header("Article 147") == 147
    assert match_english_article_header("Articles 54-80") is None


def test_clean_arabic_article_text():
    raw_sample = "مادة ( )١٠\nالقانون المصري هو المرجع في تكييف العلاقات..."
    cleaned = clean_arabic_article_text(raw_sample, 10)
    assert "مادة" not in cleaned
    assert "القانون المصري هو المرجع" in cleaned