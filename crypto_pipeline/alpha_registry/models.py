"""SQLAlchemy rows for alpha registry (Phase 4.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class AlphaBase(DeclarativeBase):
    pass


class AlphaRow(AlphaBase):
    __tablename__ = "alphas"

    alpha_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    event_type: Mapped[str] = mapped_column(String(64))
    asset: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feature_name: Mapped[str] = mapped_column(String(256))
    weight: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    backtest_sharpe: Mapped[float] = mapped_column(Float)
    backtest_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    backtest_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    live_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    retire_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class AlphaPnLDailyRow(AlphaBase):
    """Optional daily PnL series for attribution (Phase 4 API surface)."""

    __tablename__ = "alpha_pnl_daily"
    __table_args__ = (UniqueConstraint("alpha_id", "day", name="uq_alpha_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alpha_id: Mapped[str] = mapped_column(String(256), index=True)
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pnl_usd: Mapped[float] = mapped_column(Float)
