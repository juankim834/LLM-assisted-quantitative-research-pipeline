"""Alpha registry: Postgres durable + Redis cache + public API (Phase 4.3)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import redis
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from crypto_pipeline.alpha_registry.holdout import assert_alpha_passes_holdout, default_holdout_window
from crypto_pipeline.alpha_registry.models import AlphaBase, AlphaPnLDailyRow, AlphaRow
from crypto_pipeline.ingestion.storage import StorageService
from crypto_pipeline.schemas.contracts import Alpha, TradeRecord


def _alpha_to_row(a: Alpha, retire_reason: str | None = None) -> AlphaRow:
    bs, be = a.backtest_period
    payload = a.model_dump(mode="json")
    return AlphaRow(
        alpha_id=a.alpha_id,
        version=a.version,
        event_type=a.event_type,
        asset=a.asset,
        feature_name=a.feature_name,
        weight=a.weight,
        confidence=a.confidence,
        backtest_sharpe=a.backtest_sharpe,
        backtest_start=bs,
        backtest_end=be,
        live_since=a.live_since,
        expires_at=a.expires_at,
        status=a.status,
        retire_reason=retire_reason,
        payload=payload,
    )


def _row_to_alpha(row: AlphaRow) -> Alpha:
    return Alpha(
        alpha_id=row.alpha_id,
        version=row.version,
        event_type=row.event_type,
        asset=row.asset,
        feature_name=row.feature_name,
        weight=row.weight,
        confidence=row.confidence,
        backtest_sharpe=row.backtest_sharpe,
        backtest_period=(row.backtest_start, row.backtest_end),
        live_since=row.live_since,
        expires_at=row.expires_at,
        status=row.status,
    )


@dataclass
class AlphaRegistryConfig:
    database_url: str = "sqlite:///./data/pipeline.db"
    redis_url: str | None = None
    redis_key_prefix: str = "alpha:"
    enforce_holdout: bool = True


class AlphaRegistry:
    """
    Single entry point for QR writes and trading reads. Pipelines should use this class
    rather than ad-hoc SQL (Phase 4 exit criteria).
    """

    def __init__(self, config: AlphaRegistryConfig | None = None) -> None:
        self.config = config or AlphaRegistryConfig()
        self.engine = create_engine(self.config.database_url, future=True)
        AlphaBase.metadata.create_all(self.engine)
        self._redis: redis.Redis | None = None
        if self.config.redis_url or os.environ.get("REDIS_URL"):
            url = self.config.redis_url or os.environ.get("REDIS_URL", "")
            self._redis = redis.from_url(url, decode_responses=True)

    def _redis_set(self, a: Alpha) -> None:
        if not self._redis:
            return
        key = f"{self.config.redis_key_prefix}{a.alpha_id}"
        self._redis.set(key, json.dumps(a.model_dump(mode="json")))

    def _redis_delete(self, alpha_id: str) -> None:
        if self._redis:
            self._redis.delete(f"{self.config.redis_key_prefix}{alpha_id}")

    def upsert_alpha(self, alpha: Alpha) -> None:
        if self.config.enforce_holdout:
            assert_alpha_passes_holdout(alpha, default_holdout_window())
        with Session(self.engine) as session:
            row = session.get(AlphaRow, alpha.alpha_id)
            new = _alpha_to_row(alpha, retire_reason=None)
            if row is None:
                session.add(new)
            else:
                for c in [
                    "version",
                    "event_type",
                    "asset",
                    "feature_name",
                    "weight",
                    "confidence",
                    "backtest_sharpe",
                    "backtest_start",
                    "backtest_end",
                    "live_since",
                    "expires_at",
                    "status",
                    "payload",
                ]:
                    setattr(row, c, getattr(new, c))
                if alpha.status != "retired":
                    row.retire_reason = None
            session.commit()
        self._redis_set(alpha)

    def retire_alpha(self, alpha_id: str, reason: str) -> None:
        with Session(self.engine) as session:
            row = session.get(AlphaRow, alpha_id)
            if row is None:
                return
            row.status = "retired"
            row.retire_reason = reason
            if row.payload:
                row.payload = dict(row.payload)
                row.payload["retire_reason"] = reason
            session.commit()
        self._redis_delete(alpha_id)

    def get_live_alphas(self, asset: str | None = None) -> list[Alpha]:
        """Trading pipeline read — Redis cache when populated; else Postgres."""
        if self._redis:
            out: list[Alpha] = []
            for k in self._redis.scan_iter(f"{self.config.redis_key_prefix}*"):
                raw = self._redis.get(k)
                if not raw:
                    continue
                data = json.loads(raw)
                a = Alpha.model_validate(data)
                if a.status != "live":
                    continue
                if asset and (a.asset or "").upper() != asset.upper():
                    continue
                out.append(a)
            if out:
                return sorted(out, key=lambda x: x.alpha_id)

        with Session(self.engine) as session:
            q = select(AlphaRow).where(AlphaRow.status == "live")
            rows = list(session.scalars(q).all())
        result = [_row_to_alpha(r) for r in rows]
        if asset:
            result = [a for a in result if (a.asset or "").upper() == asset.upper()]
        return sorted(result, key=lambda x: x.alpha_id)

    def get_alpha_pnl(
        self,
        alpha_id: str,
        start: datetime,
        end: datetime,
        *,
        storage: StorageService | None = None,
    ) -> pd.Series:
        """
        Daily PnL series from closed trades (realized_pnl) attributed to alpha_id.
        Falls back to `alpha_pnl_daily` table if populated.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        else:
            start = start.astimezone(UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        else:
            end = end.astimezone(UTC)

        with Session(self.engine) as session:
            q = (
                select(AlphaPnLDailyRow)
                .where(AlphaPnLDailyRow.alpha_id == alpha_id)
                .where(AlphaPnLDailyRow.day >= start)
                .where(AlphaPnLDailyRow.day <= end)
                .order_by(AlphaPnLDailyRow.day.asc())
            )
            daily = list(session.scalars(q).all())
        if daily:
            idx = pd.DatetimeIndex([d.day for d in daily])
            return pd.Series([d.pnl_usd for d in daily], index=idx, name="pnl_usd")

        if storage is None:
            storage = StorageService()
        trades = storage.query_trades_by_alpha(alpha_id, start=start, end=end)
        return _pnl_series_from_trades(trades, start, end)


def _pnl_series_from_trades(
    trades: list[TradeRecord],
    start: datetime,
    end: datetime,
) -> pd.Series:
    buckets: dict[datetime, float] = {}
    for t in trades:
        if t.realized_pnl is None:
            continue
        day = t.exit_at or t.entry_at
        if day.tzinfo is None:
            day = day.replace(tzinfo=UTC)
        day = day.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        if day < start or day > end:
            continue
        buckets[day] = buckets.get(day, 0.0) + float(t.realized_pnl)
    if not buckets:
        return pd.Series(dtype=float, name="pnl_usd")
    idx = pd.DatetimeIndex(sorted(buckets.keys()))
    return pd.Series([buckets[d] for d in idx], index=idx, name="pnl_usd")
