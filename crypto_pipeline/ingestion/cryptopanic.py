"""CryptoPanic API ingestion (Phase 0.1). Requires CRYPTOPANIC_API_KEY in environment."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx

from crypto_pipeline.schemas.contracts import RawArticle

# Developer API v2 base — see https://cryptopanic.com/developers/api/
DEFAULT_BASE = "https://cryptopanic.com/api/developer/v2"


def _parse_dt(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    raw = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _currency_codes(item: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for c in item.get("currencies") or []:
        if isinstance(c, dict) and c.get("code"):
            codes.append(str(c["code"]).upper())
        elif isinstance(c, str):
            codes.append(c.upper())
    return sorted(set(codes))


def _domain(item: dict[str, Any]) -> str:
    src = item.get("source") or {}
    if isinstance(src, dict):
        return str(src.get("domain") or src.get("title") or "unknown")
    return "unknown"


def _post_to_raw(item: dict[str, Any]) -> RawArticle:
    pid = item.get("id")
    post_id = str(pid) if pid is not None else str(hash(str(item)))
    title = (item.get("title") or "").strip() or "(no title)"
    url = str(item.get("url") or item.get("original_url") or "")
    published = _parse_dt(item.get("published_at") or item.get("created_at"))
    votes = item.get("votes") if isinstance(item.get("votes"), dict) else {}
    return RawArticle(
        id=post_id,
        title=title,
        body=None,
        url=url,
        published_at=published,
        source_domain=_domain(item),
        asset_mentions=_currency_codes(item),
        raw_votes=dict(votes) if votes else {},
    )


class CryptoPanicClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("CRYPTOPANIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def fetch_posts(
        self,
        *,
        currencies: str | None = None,
        filter_: str | None = None,
        public: bool = True,
        max_pages: int = 10,
    ) -> list[RawArticle]:
        """Paginate `/posts/` until empty or max_pages. Set CRYPTOPANIC_API_KEY."""
        if not self.api_key:
            raise ValueError("CRYPTOPANIC_API_KEY is not set")
        out: list[RawArticle] = []
        next_url: str | None = f"{self.base_url}/posts/"
        params: dict[str, str] = {"auth_token": self.api_key}
        if currencies:
            params["currencies"] = currencies
        if filter_:
            params["filter"] = filter_
        if public:
            params["public"] = "true"

        with httpx.Client(timeout=self.timeout_s) as client:
            for _ in range(max_pages):
                if not next_url:
                    break
                r = client.get(next_url, params=params if next_url.endswith("/posts/") else None)
                r.raise_for_status()
                data = r.json()
                for item in data.get("results") or []:
                    if isinstance(item, dict):
                        out.append(_post_to_raw(item))
                next_url = data.get("next") or None
                params = {}

        return out


def fetch_cryptopanic_posts(
    *,
    currencies: str | None = None,
    api_key: str | None = None,
    max_pages: int = 10,
) -> list[RawArticle]:
    return CryptoPanicClient(api_key=api_key).fetch_posts(
        currencies=currencies,
        max_pages=max_pages,
    )
