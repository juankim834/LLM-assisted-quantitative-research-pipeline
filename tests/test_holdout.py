from datetime import UTC, datetime, timedelta

from crypto_pipeline.alpha_registry.holdout import (
    default_holdout_window,
    intervals_overlap,
    reject_if_holdout_overlap,
)
from crypto_pipeline.schemas.contracts import Alpha


def test_intervals_overlap():
    a = (datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 6, 1, tzinfo=UTC))
    b = (datetime(2020, 5, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    assert intervals_overlap(a, b) is True
    c = (datetime(2021, 1, 2, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC))
    assert intervals_overlap(a, c) is False


def test_reject_holdout_overlap():
    now = datetime(2025, 1, 15, tzinfo=UTC)
    h = (now - timedelta(days=90), now)
    alpha = Alpha(
        alpha_id="x",
        event_type="hack",
        feature_name="f",
        backtest_sharpe=1.0,
        backtest_period=(h[0] + timedelta(days=1), h[1] - timedelta(days=1)),
        status="paper",
    )
    assert reject_if_holdout_overlap(alpha, h) is True

    old = Alpha(
        alpha_id="y",
        event_type="hack",
        feature_name="f",
        backtest_sharpe=1.0,
        backtest_period=(
            datetime(2019, 1, 1, tzinfo=UTC),
            datetime(2019, 12, 31, tzinfo=UTC),
        ),
        status="paper",
    )
    assert reject_if_holdout_overlap(old, h) is False


def test_default_holdout_window():
    s, e = default_holdout_window(datetime(2025, 3, 1, tzinfo=UTC), days=90)
    assert (e - s).days == 90
