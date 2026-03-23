from datetime import UTC, datetime, timedelta

import pandas as pd

from crypto_pipeline.qr import tools


def _sample_df():
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    rows = []
    for i in range(200):
        ts = t0 + timedelta(hours=i)
        for sym in ["BTC/USDT", "ETH/USDT"]:
            price = 40000.0 + i * 10 if "BTC" in sym else 2000.0 + i
            rows.append(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000.0 + i,
                    "symbol": sym,
                }
            )
    return pd.DataFrame(rows)


def test_abnormal_return():
    df = _sample_df()
    t0 = datetime(2024, 1, 5, 12, 0, tzinfo=UTC)
    out = tools.abnormal_return("BTC/USDT", t0, df, baseline_symbol="ETH/USDT")
    assert "abnormal_vs_baseline" in out
    assert isinstance(out["abnormal_vs_baseline"], dict)


def test_volume_spike():
    df = _sample_df()
    t0 = datetime(2024, 1, 5, 12, 0, tzinfo=UTC)
    z = tools.volume_spike("BTC/USDT", t0, df)
    assert isinstance(z, float)
