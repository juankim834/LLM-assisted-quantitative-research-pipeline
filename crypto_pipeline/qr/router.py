"""Hypothesis router + AnalysisResult assembly (Phase 3.1, 3.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from crypto_pipeline.qr import tools
from crypto_pipeline.schemas.contracts import AnalysisResult, EventTag

HYPOTHESIS_MAP: dict[str, tuple[str, list[str]]] = {
    "listing": ("H1_listing_spike_revert", ["abnormal_return", "volume_spike"]),
    "hack": ("H2_hack_contagion", ["abnormal_return", "contagion_spread"]),
    "regulatory": ("H3_regulatory_relative", ["relative_return", "regime_shift"]),
    "macro": ("H4_macro_btc_dom", ["btc_dominance_change", "altcoin_beta"]),
    "partnership": ("H5_partnership_drift", ["abnormal_return", "volume_spike"]),
    "other": ("H0_other", ["abnormal_return"]),
}


def route_and_analyze(
    tag: EventTag,
    ohlcv_df: pd.DataFrame,
    *,
    t0: datetime | None = None,
    baseline: str = "BTC/USDT",
    peer_group: list[str] | None = None,
) -> AnalysisResult:
    """
    Runs tools for the event's hypothesis. `ohlcv_df` must be long-form from ingestion
    (timestamp, open, high, low, close, volume, symbol).
    """
    t0 = t0 or tag.tagged_at
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=UTC)
    else:
        t0 = t0.astimezone(UTC)

    hyp_id, tool_names = HYPOTHESIS_MAP.get(tag.event_type, ("H0_other", ["abnormal_return"]))
    asset = tag.asset
    sym = f"{asset}/USDT" if asset else baseline
    peers = peer_group or ["ETH/USDT", "SOL/USDT", "BNB/USDT"]

    outputs: dict[str, Any] = {}
    if "abnormal_return" in tool_names:
        outputs["abnormal_return"] = tools.abnormal_return(
            sym, t0, ohlcv_df, baseline_symbol=baseline
        )
    if "volume_spike" in tool_names:
        outputs["volume_spike"] = tools.volume_spike(sym, t0, ohlcv_df)
    if "contagion_spread" in tool_names:
        outputs["contagion_spread"] = tools.contagion_spread(sym, peers, t0, ohlcv_df)
    if "relative_return" in tool_names and asset:
        outputs["relative_return"] = tools.relative_return(sym, peers, t0, ohlcv_df)
    elif "relative_return" in tool_names:
        outputs["relative_return"] = {"skipped": "no asset on tag"}
    if "regime_shift" in tool_names:
        outputs["regime_shift"] = tools.regime_shift(
            t0, ohlcv_df, symbols=[baseline, sym] + peers[:2]
        )
    if "btc_dominance_change" in tool_names:
        outputs["btc_dominance_change"] = tools.btc_dominance_change(t0, ohlcv_df)
    if "altcoin_beta" in tool_names:
        outputs["altcoin_beta"] = tools.altcoin_beta(t0, ohlcv_df)

    return AnalysisResult(
        event_tag=tag,
        hypothesis_id=hyp_id,
        tool_outputs=outputs,
        computed_at=datetime.now(UTC),
        summary=None,
    )
