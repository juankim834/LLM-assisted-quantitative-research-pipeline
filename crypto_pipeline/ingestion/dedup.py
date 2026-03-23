"""Deduplication: hash on (title, published_at) for cross-source reposts."""

from __future__ import annotations

import hashlib
from datetime import datetime

from crypto_pipeline.schemas.contracts import RawArticle


def article_dedup_key(title: str, published_at: datetime) -> str:
    normalized_title = " ".join(title.strip().lower().split())
    payload = f"{normalized_title}|{published_at.astimezone().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_articles(articles: list[RawArticle]) -> list[RawArticle]:
    seen: set[str] = set()
    out: list[RawArticle] = []
    for a in articles:
        key = article_dedup_key(a.title, a.published_at)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out
