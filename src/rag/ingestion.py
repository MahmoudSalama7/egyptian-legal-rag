from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# pyrefly: ignore [missing-import]
import pymupdf

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

MAX_LEGAL_ARTICLE = 1149

REPEALED_RANGES = [
    range(54, 81),  # 54 to 80
    range(389, 418),  # 389 to 417
]
REPEALED_ARTICLES = {art for r in REPEALED_RANGES for art in r}


def normalize_digits(text: str) -> str:
    return text.translate(ARABIC_DIGITS)


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # تنظيف حرف 'ا' المنفرد الشائع في نهاية المقاطع
    filtered_lines = [line for line in lines if line and line != "ا"]
    return "\n".join(filtered_lines)


def match_english_article_header(line: str) -> int | None:
    # يجب أن يبدأ السطر حصراً بكلمة Article مع رقم المادة
    if re.search(r"Articles\s+[0-9]+\s*-\s*[0-9]+", line, re.IGNORECASE):
        return None
    match = re.search(r"^\s*Article\s+([0-9]{1,4})\s*$", line, re.IGNORECASE)
    if not match:
        match = re.search(r"^\s*Article\s+([0-9]{1,4})\b", line, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= MAX_LEGAL_ARTICLE:
            return num
    return None


def clean_arabic_article_text(raw_ar: str, art_num: int) -> str:
    lines = raw_ar.splitlines()
    cleaned_lines = []
    skipped_header = False

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        if line_str in {
            "القانون المدني المصري",
            "نصوص القانون المدنى",
            "قانون الإصدار",
        }:
            continue

        norm = normalize_digits(line_str)
        # إزالة أشكال رؤوس المواد المقلوبة أو ذات الأقواس في البداية: مثل "٠١(" أو "مادة ( )١"
        if not skipped_header:
            header_pattern = re.search(
                r"^(?:\([ \t]*)?(?:مادة|ماده)?[\(\)\s]*([0-9]{1,4})[\(\)\s]*(.*)$", norm
            )
            if header_pattern:
                found_num = (
                    int(header_pattern.group(1)) if header_pattern.group(1) else None
                )
                # إذا كان الرقم في بداية السطر يطابق رقم المادة الحالية
                if found_num == art_num or (
                    found_num and str(found_num) in {str(art_num), str(art_num)[::-1]}
                ):
                    rem = header_pattern.group(2).strip()
                    if rem:
                        cleaned_lines.append(rem)
                    skipped_header = True
                    continue

        cleaned_lines.append(line_str)

    return clean_text("\n".join(cleaned_lines))


def get_precise_english_markers(
    page: pymupdf.Page, mid_x: float, min_y: float, max_y: float
) -> list[tuple[float, int]]:
    """
    يبحث عن الإحداثي الرأسي Y الدقيق لكل سطر يبدأ بـ Article N
    لتفادي اشتراك مادتين في نفس y0 الخاص بالبلوك.
    """
    markers: list[tuple[float, int]] = []
    seen_articles = set()

    # فحص الكلمات واستخراج أسطر النصف الأيسر
    words = page.get_text("words")
    # تجميع الكلمات حسب رقم السطر وبلوك PyMuPDF: w = (x0, y0, x1, y1, word, block_no, line_no, word_no)
    lines_dict: dict[tuple[int, int], list] = {}
    for w in words:
        x0, y0, _x1, y1, _word, block_no, line_no, _word_no = w
        if y1 < min_y or y0 > max_y or x0 > mid_x:
            continue
        lines_dict.setdefault((block_no, line_no), []).append(w)

    for line_words in lines_dict.values():
        line_words.sort(key=lambda x: x[0])  # فرز الكلمات أفقياً
        line_text = " ".join(w[4] for w in line_words).strip()
        art_num = match_english_article_header(line_text)
        if art_num is not None and art_num not in seen_articles:
            line_y0 = min(w[1] for w in line_words)
            markers.append((line_y0, art_num))
            seen_articles.add(art_num)

    markers.sort(key=lambda m: m[0])
    return markers


def extract_articles(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)

    articles_ar: dict[int, list[str]] = {}
    articles_en: dict[int, list[str]] = {}
    source_pages: dict[int, int] = {}

    current_open_article = None

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]
        rect = page.rect
        mid_x = rect.width / 2.0

        min_y = 25.0
        max_y = rect.height - 25.0

        raw_text = page.get_text("text")
        if (
            page_num == 1
            and "قانون الإصدار" in raw_text
            and "نصوص القانون المدنى" not in raw_text
        ):
            continue

        # استخراج مواضع Article N الدقيقة رأسياً
        article_markers = get_precise_english_markers(page, mid_x, min_y, max_y)

        # استمرار نص المادة من الصفحة السابقة في أعلى الصفحة الحالية
        first_y = article_markers[0][0] if article_markers else max_y
        if current_open_article is not None and first_y > min_y + 8.0:
            top_left = pymupdf.Rect(0, min_y, mid_x + 5.0, first_y)
            top_right = pymupdf.Rect(mid_x - 5.0, min_y, rect.width, first_y)

            cont_en = clean_text(page.get_text("text", clip=top_left))
            cont_ar = clean_text(page.get_text("text", clip=top_right))

            if cont_en:
                articles_en.setdefault(current_open_article, []).append(cont_en)
            if cont_ar:
                articles_ar.setdefault(current_open_article, []).append(cont_ar)

        # قص نصوص كل مادة على حدة بدقة
        for i, (y_pos, art_num) in enumerate(article_markers):
            current_open_article = art_num
            if current_open_article not in source_pages:
                source_pages[current_open_article] = page_num

            # نهاية المستطيل هي بداية المادة التالية أو أسفل الصفحة
            next_y = (
                article_markers[i + 1][0] if i + 1 < len(article_markers) else max_y
            )

            # تجنب أي خطأ إذا كانت المسافة الرأسية ضيقة جداً
            if next_y <= y_pos:
                next_y = max_y

            clip_left = pymupdf.Rect(0, y_pos, mid_x + 5.0, next_y)
            clip_right = pymupdf.Rect(mid_x - 5.0, y_pos, rect.width, next_y)

            raw_en_clip = page.get_text("text", clip=clip_left)
            raw_ar_clip = page.get_text("text", clip=clip_right)

            # إزالة سطر الترويسة Article N من الإنجليزي
            clean_en_lines = []
            for line in raw_en_clip.splitlines():
                if match_english_article_header(line.strip()) == art_num:
                    continue
                if line.strip():
                    clean_en_lines.append(line.strip())
            en_body = clean_text("\n".join(clean_en_lines))

            ar_body = clean_arabic_article_text(raw_ar_clip, art_num)

            if en_body:
                articles_en.setdefault(art_num, []).append(en_body)
            if ar_body:
                articles_ar.setdefault(art_num, []).append(ar_body)

    doc.close()

    # تجميع المواد وحساب الملغاة
    all_article_numbers = sorted(
        set(articles_en.keys()).union(articles_ar.keys()).union(REPEALED_ARTICLES)
    )

    extracted_records = []

    for num in all_article_numbers:
        if num > MAX_LEGAL_ARTICLE:
            continue

        is_repealed = num in REPEALED_ARTICLES

        text_ar = clean_text("\n".join(articles_ar.get(num, [])))
        text_en = clean_text("\n".join(articles_en.get(num, [])))

        if is_repealed:
            if not text_ar:
                text_ar = "ملغاة بموجب قرار تشريعي لاحق."
            if not text_en:
                text_en = "Repealed by subsequent decree."

        extracted_records.append(
            {
                "article_number": num,
                "book": None,
                "chapter": None,
                "section": None,
                "topic": None,
                "text_ar": text_ar,
                "text_en": text_en,
                "is_repealed": is_repealed,
                "source_page": source_pages.get(num, 1),
                "citation": f"Egyptian Civil Code, Article {num}",
            }
        )

    return extracted_records


def save_articles(articles: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Civil Code articles from PDF."
    )
    parser.add_argument("--input", required=True, help="Path to input Law.pdf")
    parser.add_argument(
        "--output", required=True, help="Path to output civil_code.json"
    )
    args = parser.parse_args()

    articles = extract_articles(args.input)
    print(f"Total articles extracted: {len(articles)}")
    save_articles(articles, args.output)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
