"""PostgreSQL (JSONB) + SQLite dev store for raw news and OHLCV (Phase 0.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Sequence

from sqlalchemy import JSON, DateTime, Float, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from crypto_pipeline.schemas.contracts import RawArticle, TradeRecord


class Base(DeclarativeBase):
    pass


class RawArticleRow(Base):
    __tablename__ = "raw_articles"

    dedup_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source_domain: Mapped[str] = mapped_column(String(512))
    asset_mentions: Mapped[list[Any]] = mapped_column(JSON)
    raw_votes: Mapped[dict[str, Any]] = mapped_column(JSON)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class OHLCVRow(Base):
    __tablename__ = "ohlcv_1h"
    __table_args__ = (UniqueConstraint("symbol", "ts", name="uq_ohlcv_symbol_ts"),)

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class TradeRecordRow(Base):
    __tablename__ = "trade_records"

    trade_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    alpha_id: Mapped[str] = mapped_column(String(256), index=True)
    asset: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    size_usd: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    exit_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)


class AnalysisResultRow(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[str] = mapped_column(String(128), index=True)
    cluster_id: Mapped[str] = mapped_column(String(128), index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


@dataclass
class StorageConfig:
    database_url: str = "sqlite:///./data/pipeline.db"


class StorageService:
    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig()
        self.engine = create_engine(self.config.database_url, echo=False, future=True)
        Base.metadata.create_all(self.engine)

    def upsert_raw_articles(self, articles: Sequence[RawArticle], dedup_keys: Sequence[str]) -> int:
        if len(dedup_keys) != len(articles):
            raise ValueError("dedup_keys must align with articles")
        n = 0
        with Session(self.engine) as session:
            for a, dk in zip(articles, dedup_keys, strict=True):
                row = session.get(RawArticleRow, dk)
                payload = a.model_dump(mode="json")
                if row is None:
                    session.add(
                        RawArticleRow(
                            dedup_key=dk,
                            article_id=a.id,
                            title=a.title,
                            body=a.body,
                            url=a.url,
                            published_at=a.published_at,
                            source_domain=a.source_domain,
                            asset_mentions=list(a.asset_mentions),
                            raw_votes=dict(a.raw_votes),
                            payload=payload,
                        )
                    )
                    n += 1
                else:
                    row.title = a.title
                    row.body = a.body
                    row.url = a.url
                    row.published_at = a.published_at
                    row.source_domain = a.source_domain
                    row.asset_mentions = list(a.asset_mentions)
                    row.raw_votes = dict(a.raw_votes)
                    row.payload = payload
            session.commit()
        return n

    def query_articles(
        self,
        *,
        asset: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 10_000,
    ) -> list[RawArticle]:
        with Session(self.engine) as session:
            q = select(RawArticleRow)
            if start is not None:
                q = q.where(RawArticleRow.published_at >= start)
            if end is not None:
                q = q.where(RawArticleRow.published_at <= end)
            q = q.order_by(RawArticleRow.published_at.desc()).limit(limit)
            rows = list(session.scalars(q).all())
        out: list[RawArticle] = []
        for row in rows:
            if asset and asset.upper() not in {x.upper() for x in (row.asset_mentions or [])}:
                continue
            out.append(
                RawArticle(
                    id=row.article_id,
                    title=row.title,
                    body=row.body,
                    url=row.url,
                    published_at=row.published_at,
                    source_domain=row.source_domain,
                    asset_mentions=list(row.asset_mentions or []),
                    raw_votes=dict(row.raw_votes or {}),
                )
            )
        return out

    def insert_ohlcv_dataframe_rows(self, records: list[dict[str, Any]]) -> int:
        """Each dict: timestamp (datetime), open, high, low, close, volume, symbol."""
        n = 0
        with Session(self.engine) as session:
            for r in records:
                ts = r["timestamp"]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.replace(tzinfo=UTC)
                session.merge(
                    OHLCVRow(
                        symbol=str(r["symbol"]),
                        ts=ts,
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                    )
                )
                n += 1
            session.commit()
        return n

    def query_ohlcv(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVRow]:
        with Session(self.engine) as session:
            q = select(OHLCVRow).where(OHLCVRow.symbol == symbol)
            if start is not None:
                q = q.where(OHLCVRow.ts >= start)
            if end is not None:
                q = q.where(OHLCVRow.ts <= end)
            q = q.order_by(OHLCVRow.ts.asc())
            return list(session.scalars(q).all())

    def save_analysis_result_payload(self, payload: dict[str, Any]) -> None:
        from datetime import datetime as dt

        computed = payload.get("computed_at")
        if isinstance(computed, str):
            computed = dt.fromisoformat(computed.replace("Z", "+00:00"))
        et = payload.get("event_tag") or {}
        cluster_id = str(et.get("cluster_id", ""))
        hyp = str(payload.get("hypothesis_id", ""))
        with Session(self.engine) as session:
            session.add(
                AnalysisResultRow(
                    hypothesis_id=hyp,
                    cluster_id=cluster_id,
                    computed_at=computed if isinstance(computed, datetime) else datetime.now(UTC),
                    payload=payload,
                )
            )
            session.commit()

    def upsert_trade_record(self, tr: TradeRecord) -> None:
        with Session(self.engine) as session:
            row = session.get(TradeRecordRow, tr.trade_id)
            snap = dict(tr.signal_snapshot)
            if row is None:
                session.add(
                    TradeRecordRow(
                        trade_id=tr.trade_id,
                        alpha_id=tr.alpha_id,
                        asset=tr.asset,
                        side=tr.side,
                        size_usd=tr.size_usd,
                        entry_price=tr.entry_price,
                        exit_price=tr.exit_price,
                        entry_at=tr.entry_at,
                        exit_at=tr.exit_at,
                        realized_pnl=tr.realized_pnl,
                        signal_snapshot=snap,
                    )
                )
            else:
                row.exit_price = tr.exit_price
                row.exit_at = tr.exit_at
                row.realized_pnl = tr.realized_pnl
                row.signal_snapshot = snap
            session.commit()

    def query_trades_by_alpha(
        self,
        alpha_id: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[TradeRecord]:
        with Session(self.engine) as session:
            q = (
                select(TradeRecordRow)
                .where(TradeRecordRow.alpha_id == alpha_id)
                .where(TradeRecordRow.entry_at >= start)
                .where(TradeRecordRow.entry_at <= end)
                .order_by(TradeRecordRow.entry_at.asc())
            )
            rows = list(session.scalars(q).all())
        return [
            TradeRecord(
                trade_id=r.trade_id,
                alpha_id=r.alpha_id,
                asset=r.asset,
                side=r.side,
                size_usd=r.size_usd,
                entry_price=r.entry_price,
                exit_price=r.exit_price,
                entry_at=r.entry_at,
                exit_at=r.exit_at,
                realized_pnl=r.realized_pnl,
                signal_snapshot=dict(r.signal_snapshot or {}),
            )
            for r in rows
        ]


def row_to_ohlcv_dict(row: OHLCVRow) -> dict[str, Any]:
    return {
        "timestamp": row.ts,
        "open": row.open,
        "high": row.high,
        "low": row.low,
        "close": row.close,
        "volume": row.volume,
        "symbol": row.symbol,
    }
