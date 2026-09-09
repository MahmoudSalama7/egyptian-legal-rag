from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_LEGAL_ARTICLE = 1149


def validate_articles(input_path: str | Path) -> bool:
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return False

    with input_path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    errors: list[str] = []

    if not isinstance(articles, list) or not articles:
        print("ERROR: Root JSON must be a non-empty list.")
        return False

    seen_numbers = []

    for index, article in enumerate(articles):
        prefix = f"Article index {index}"
        if not isinstance(article, dict):
            errors.append(f"{prefix}: not an object.")
            continue

        num = article.get("article_number")
        is_repealed = article.get("is_repealed", False)
        text_ar = article.get("text_ar", "")
        text_en = article.get("text_en", "")
        citation = article.get("citation", "")
        source_page = article.get("source_page")

        # 1. Number validation
        if not isinstance(num, int) or num < 1 or num > MAX_LEGAL_ARTICLE:
            errors.append(f"{prefix}: invalid article_number {num!r}.")
        else:
            seen_numbers.append(num)

        # 2. Citation check
        if citation != f"Egyptian Civil Code, Article {num}":
            errors.append(f"{prefix}: citation mismatch '{citation}'.")

        # 3. Source page check
        if not isinstance(source_page, int) or source_page < 1:
            errors.append(f"{prefix}: invalid source_page {source_page!r}.")

        # 4. Text validation
        if not is_repealed:
            if not isinstance(text_ar, str) or len(text_ar.strip()) < 5:
                errors.append(
                    f"{prefix} (Article {num}): Arabic text is empty or too short."
                )
            if not isinstance(text_en, str) or len(text_en.strip()) < 5:
                errors.append(
                    f"{prefix} (Article {num}): English text is empty or too short."
                )

            # Sanity: Arabic text must contain Arabic characters
            ar_chars = sum("\u0600" <= c <= "\u06ff" for c in text_ar)
            if ar_chars < 5:
                errors.append(
                    f"{prefix} (Article {num}): text_ar lacks Arabic characters."
                )

            # Sanity: English text must contain Latin characters
            en_chars = sum("a" <= c.lower() <= "z" for c in text_en)
            if en_chars < 5:
                errors.append(
                    f"{prefix} (Article {num}): text_en lacks English characters."
                )

    # Check duplicates
    duplicates = [n for n in set(seen_numbers) if seen_numbers.count(n) > 1]
    if duplicates:
        errors.append(f"Duplicate article numbers found: {sorted(duplicates)}")

    # Check sorting
    if seen_numbers != sorted(seen_numbers):
        errors.append("Articles are not strictly sorted by article_number.")

    print(f"Total articles scanned: {len(articles)}")
    print(f"Unique article numbers: {len(set(seen_numbers))}")

    if errors:
        print("\nVALIDATION FAILED:")
        print("=" * 60)
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more errors.")
        return False

    print("\nALL CHECKS PASSED: Corpus is clean, bilingual, and validated.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Civil Code JSON corpus.")
    parser.add_argument("--input", required=True, help="Path to articles JSON")
    args = parser.parse_args()

    success = validate_articles(args.input)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
