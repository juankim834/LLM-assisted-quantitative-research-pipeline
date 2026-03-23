"""OHLCV at 1h via CCXT (Phase 0.2). Default exchange: Binance spot — document in config."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import ccxt
import pandas as pd


@dataclass(frozen=True)
class OHLCVConfig:
    exchange_id: str = "binance"
    market_type: str = "spot"
    timeframe: str = "1h"
    """All timestamps UTC; prices from a single exchange (see exchange_id)."""


def fetch_ohlcv_1h_ccxt(
    symbols: list[str],
    *,
    since: datetime,
    until: datetime | None = None,
    config: OHLCVConfig | None = None,
) -> pd.DataFrame:
    """
    Returns long dataframe: columns timestamp, open, high, low, close, volume, symbol.
    `since`/`until` are interpreted as UTC.
    """
    cfg = config or OHLCVConfig()
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    else:
        since = since.astimezone(UTC)
    if until is None:
        until = datetime.now(UTC)
    elif until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    else:
        until = until.astimezone(UTC)

    ex_class = getattr(ccxt, cfg.exchange_id)
    ex: Any = ex_class({"enableRateLimit": True, "options": {"defaultType": cfg.market_type}})
    ms_since = int(since.timestamp() * 1000)
    ms_until = int(until.timestamp() * 1000)

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        cursor = ms_since
        while cursor < ms_until:
            batch = ex.fetch_ohlcv(sym, timeframe=cfg.timeframe, since=cursor, limit=1000)
            if not batch:
                break
            for o, h, l, c, v, ts in batch:
                if ts > ms_until:
                    continue
                rows.append(
                    {
                        "timestamp": pd.Timestamp(ts, unit="ms", tz=UTC),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v),
                        "symbol": sym,
                    }
                )
            cursor = int(batch[-1][0]) + 1
            if len(batch) < 1000:
                break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def ohlcv_last_n_days(
    symbols: list[str],
    *,
    days: int = 7,
    config: OHLCVConfig | None = None,
) -> pd.DataFrame:
    until = datetime.now(UTC)
    since = until - timedelta(days=days)
    return fetch_ohlcv_1h_ccxt(symbols, since=since, until=until, config=config)
