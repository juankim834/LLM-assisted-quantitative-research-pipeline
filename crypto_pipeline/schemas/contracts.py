"""Frozen schema contracts (Phase 0) — Pydantic models shared across stages."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RawArticle(BaseModel):
    id: str
    title: str
    body: str | None = None
    url: str
    published_at: datetime
    source_domain: str
    asset_mentions: list[str] = Field(default_factory=list)
    raw_votes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class ClusterObject(BaseModel):
    cluster_id: str
    window_start: datetime
    window_end: datetime
    size: int
    representative_texts: list[str]
    representative_ids: list[str]
    centroid_embedding: list[float]
    asset_mentions: list[str] = Field(default_factory=list)

    @field_validator("window_start", "window_end")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class EventTag(BaseModel):
    cluster_id: str
    event_type: str
    asset: str | None = None
    secondary_tag: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tagged_at: datetime = Field(default_factory=_utc_now)

    @field_validator("tagged_at")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class AnalysisResult(BaseModel):
    event_tag: EventTag
    hypothesis_id: str
    tool_outputs: dict[str, Any]
    computed_at: datetime
    summary: str | None = None

    @field_validator("computed_at")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class Alpha(BaseModel):
    alpha_id: str
    version: int = 1
    event_type: str
    asset: str | None = None
    feature_name: str
    weight: float = 0.0
    confidence: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_period: tuple[datetime, datetime]
    live_since: datetime | None = None
    expires_at: datetime | None = None
    status: str = "paper"

    @field_validator("backtest_period")
    @classmethod
    def validate_period(cls, v: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
        a, b = v
        if a.tzinfo is None:
            a = a.replace(tzinfo=UTC)
        else:
            a = a.astimezone(UTC)
        if b.tzinfo is None:
            b = b.replace(tzinfo=UTC)
        else:
            b = b.astimezone(UTC)
        if a >= b:
            raise ValueError("backtest_period start must be before end")
        return (a, b)

    @field_validator("live_since", "expires_at")
    @classmethod
    def ensure_utc_optional(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class TradeRecord(BaseModel):
    trade_id: str
    alpha_id: str
    asset: str
    side: str
    size_usd: float
    entry_price: float
    exit_price: float | None = None
    entry_at: datetime
    exit_at: datetime | None = None
    realized_pnl: float | None = None
    signal_snapshot: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entry_at", "exit_at")
    @classmethod
    def ensure_utc_optional(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)
