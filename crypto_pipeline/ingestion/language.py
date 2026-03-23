"""English-only filter (Phase 0.1)."""

from __future__ import annotations

from langdetect import LangDetectException, detect

from crypto_pipeline.schemas.contracts import RawArticle


def is_english_text(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def filter_english_articles(articles: list[RawArticle]) -> list[RawArticle]:
    out: list[RawArticle] = []
    for a in articles:
        blob = f"{a.title}\n{a.body or ''}"
        if is_english_text(blob):
            out.append(a)
    return out
