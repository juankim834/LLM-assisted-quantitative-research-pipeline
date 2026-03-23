"""Fixed hourly windows (Phase 1.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from crypto_pipeline.schemas.contracts import RawArticle

MIN_CLUSTER_ARTICLES = 5


def floor_to_hour(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.replace(minute=0, second=0, microsecond=0)


def hourly_windows(
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime]]:
    """Inclusive start, exclusive end per window."""
    a = floor_to_hour(start)
    b = floor_to_hour(end)
    if b <= a:
        b = a + timedelta(hours=1)
    out: list[tuple[datetime, datetime]] = []
    cur = a
    while cur < b:
        nxt = cur + timedelta(hours=1)
        out.append((cur, nxt))
        cur = nxt
    return out


def articles_in_window(
    articles: list[RawArticle],
    window_start: datetime,
    window_end: datetime,
) -> list[RawArticle]:
    ws = window_start if window_start.tzinfo else window_start.replace(tzinfo=UTC)
    we = window_end if window_end.tzinfo else window_end.replace(tzinfo=UTC)
    ws = ws.astimezone(UTC)
    we = we.astimezone(UTC)
    return [a for a in articles if ws <= a.published_at < we]
