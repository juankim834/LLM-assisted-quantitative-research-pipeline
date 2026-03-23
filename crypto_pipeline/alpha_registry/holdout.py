"""90-day rolling hold-out anti-overfitting (Phase 4.4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from crypto_pipeline.schemas.contracts import Alpha


def default_holdout_window(now: datetime | None = None, *, days: int = 90) -> tuple[datetime, datetime]:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)
    end = now
    start = end - timedelta(days=days)
    return (start, end)


def intervals_overlap(
    a: tuple[datetime, datetime],
    b: tuple[datetime, datetime],
) -> bool:
    a0, a1 = a
    b0, b1 = b
    return a0 < b1 and b0 < a1


def reject_if_holdout_overlap(alpha: Alpha, holdout: tuple[datetime, datetime] | None = None) -> bool:
    """Returns True if alpha should be rejected (overlaps hold-out)."""
    h = holdout or default_holdout_window()
    return intervals_overlap(alpha.backtest_period, h)


def assert_alpha_passes_holdout(alpha: Alpha, holdout: tuple[datetime, datetime] | None = None) -> None:
    if reject_if_holdout_overlap(alpha, holdout):
        raise ValueError(
            f"Alpha {alpha.alpha_id} backtest period overlaps the {holdout or default_holdout_window()} hold-out window"
        )
