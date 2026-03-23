"""Deterministic quant tools (Phase 3.2) — pure Python/pandas, no LLM."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _slice(
    df: pd.DataFrame,
    symbol: str,
    t0: datetime,
    lookback: timedelta,
    forward: timedelta,
) -> pd.DataFrame:
    """Rows for symbol in [t0 - lookback, t0 + forward]."""
    t0 = _utc(t0)
    sym = df[df["symbol"] == symbol].copy()
    if sym.empty:
        return sym
    sym = sym.sort_values("timestamp")
    start = t0 - lookback
    end = t0 + forward
    ts = sym["timestamp"]
    if hasattr(ts.iloc[0], "tz") and ts.iloc[0].tz is None:
        ts = pd.to_datetime(ts, utc=True)
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    return sym.loc[mask]


def _close_at_or_before(df_sym: pd.DataFrame, t: datetime) -> float | None:
    if df_sym.empty:
        return None
    t = _utc(t)
    ts = df_sym["timestamp"]
    sub = df_sym[pd.to_datetime(ts, utc=True) <= pd.Timestamp(t)]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["close"])


def abnormal_return(
    asset: str,
    t0: datetime,
    df: pd.DataFrame,
    *,
    baseline_symbol: str,
    windows: list[timedelta] | None = None,
) -> dict[str, Any]:
    """
    Price residuals vs. market baseline: (R_asset - R_baseline) over each window
    ending at t0 + window (forward window from event).
    """
    windows = windows or [
        timedelta(hours=1),
        timedelta(hours=4),
        timedelta(hours=24),
        timedelta(hours=72),
    ]
    out: dict[str, Any] = {}
    t0 = _utc(t0)
    for w in windows:
        pre = _slice(df, asset, t0, w, timedelta(0))
        post = _slice(df, asset, t0, timedelta(0), w)
        base_pre = _slice(df, baseline_symbol, t0, w, timedelta(0))
        base_post = _slice(df, baseline_symbol, t0, timedelta(0), w)
        c0 = _close_at_or_before(pre, t0)
        c1 = _close_at_or_before(post, t0 + w)
        b0 = _close_at_or_before(base_pre, t0)
        b1 = _close_at_or_before(base_post, t0 + w)
        label = f"{int(w.total_seconds() // 3600)}h"
        if None in (c0, c1, b0, b1) or b0 == 0 or c0 == 0:
            out[label] = None
            continue
        r_a = (c1 / c0) - 1.0
        r_b = (b1 / b0) - 1.0
        out[label] = float(r_a - r_b)
    return {"asset": asset, "baseline": baseline_symbol, "abnormal_vs_baseline": out}


def volume_spike(
    asset: str,
    t0: datetime,
    df: pd.DataFrame,
    *,
    window: timedelta = timedelta(hours=24),
    lookback_days: int = 30,
) -> float:
    """Z-score of volume at t0 vs 30d rolling mean (hourly bars)."""
    t0 = _utc(t0)
    hist = _slice(df, asset, t0, timedelta(days=lookback_days), timedelta(0))
    if hist.empty or len(hist) < 2:
        return 0.0
    vols = hist["volume"].astype(float)
    mu = float(vols.mean())
    sigma = float(vols.std(ddof=0)) or 1e-12
    win = _slice(df, asset, t0, window, timedelta(0))
    v_now = float(win.iloc[-1]["volume"]) if not win.empty else float(vols.iloc[-1])
    return float((v_now - mu) / sigma)


def relative_return(
    target_asset: str,
    peer_group: list[str],
    t0: datetime,
    df: pd.DataFrame,
    *,
    windows: list[timedelta] | None = None,
) -> dict[str, Any]:
    windows = windows or [timedelta(hours=24), timedelta(days=7)]
    out: dict[str, float | None] = {}
    t0 = _utc(t0)
    peers = [p for p in peer_group if p != target_asset]
    for w in windows:
        label = f"{int(w.total_seconds() // 3600)}h" if w < timedelta(days=2) else f"{int(w.total_seconds() // 86400)}d"
        t_tar = _slice(df, target_asset, t0, timedelta(0), w)
        c0 = _close_at_or_before(_slice(df, target_asset, t0, w, timedelta(0)), t0)
        c1 = _close_at_or_before(t_tar, t0 + w)
        if c0 is None or c1 is None or c0 == 0:
            out[label] = None
            continue
        r_t = (c1 / c0) - 1.0
        peer_rs: list[float] = []
        for p in peers:
            p_pre = _slice(df, p, t0, w, timedelta(0))
            p_post = _slice(df, p, t0, timedelta(0), w)
            a = _close_at_or_before(p_pre, t0)
            b = _close_at_or_before(p_post, t0 + w)
            if a and b and a != 0:
                peer_rs.append((b / a) - 1.0)
        mean_peer = float(np.mean(peer_rs)) if peer_rs else 0.0
        out[label] = float(r_t - mean_peer)
    return {"target": target_asset, "peers": peers, "relative": out}


def contagion_spread(
    primary_asset: str,
    peer_group: list[str],
    t0: datetime,
    df: pd.DataFrame,
    *,
    window: timedelta = timedelta(hours=48),
) -> dict[str, Any]:
    """Drawdown from t0 over window for primary and peers."""
    t0 = _utc(t0)
    dd: dict[str, float | None] = {}
    for sym in [primary_asset, *peer_group]:
        pre = _slice(df, sym, t0, timedelta(hours=1), timedelta(0))
        post = _slice(df, sym, t0, timedelta(0), window)
        p0 = _close_at_or_before(pre, t0)
        path = pd.concat([pre, post]).drop_duplicates(subset=["timestamp"])
        if p0 is None or path.empty:
            dd[sym] = None
            continue
        closes = path["close"].astype(float)
        min_c = float(closes.min())
        dd[sym] = float((min_c / p0) - 1.0) if p0 else None
    return {"primary": primary_asset, "max_drawdown_fraction": dd}


def regime_shift(
    t0: datetime,
    df: pd.DataFrame,
    *,
    symbols: list[str],
    pre_window: timedelta = timedelta(days=30),
    post_window: timedelta = timedelta(days=7),
) -> dict[str, Any]:
    """Correlation structure change: mean abs diff of pairwise correlation matrices."""
    t0 = _utc(t0)
    pre = {s: _slice(df, s, t0, pre_window, timedelta(0)) for s in symbols}
    post = {s: _slice(df, s, t0, timedelta(0), post_window) for s in symbols}

    def corr_mat(dfs: dict[str, pd.DataFrame]) -> np.ndarray:
        wide = None
        for s, d in dfs.items():
            if d.empty:
                continue
            col = d.set_index("timestamp")["close"].rename(s)
            wide = col if wide is None else wide.to_frame().join(col.to_frame(), how="outer")
        if wide is None or wide.shape[1] < 2:
            return np.zeros((0, 0))
        r = wide.pct_change().dropna().corr().values
        return r

    c1 = corr_mat(pre)
    c2 = corr_mat(post)
    if c1.size == 0 or c2.size == 0 or c1.shape != c2.shape:
        return {"mean_abs_corr_diff": None, "pre_shape": getattr(c1, "shape", None), "post_shape": getattr(c2, "shape", None)}
    diff = np.abs(c1 - c2)
    return {"mean_abs_corr_diff": float(np.mean(diff))}


def btc_dominance_change(
    t0: datetime,
    df: pd.DataFrame,
    *,
    btc: str = "BTC/USDT",
    universe: list[str] | None = None,
    window: timedelta = timedelta(hours=72),
) -> float:
    """Proxy: BTC share of sum of close * volume across universe at t0 vs t0+window."""
    t0 = _utc(t0)
    universe = universe or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    def score(sym: str, t: datetime) -> float:
        row = _slice(df, sym, t, timedelta(hours=2), timedelta(0))
        if row.empty:
            return 0.0
        last = row.iloc[-1]
        return float(last["close"]) * float(last["volume"])

    s0 = sum(score(s, t0) for s in universe)
    s1 = sum(score(s, t0 + window) for s in universe)
    b0 = score(btc, t0)
    b1 = score(btc, t0 + window)
    if s0 <= 0 or s1 <= 0:
        return 0.0
    dom0 = b0 / s0
    dom1 = b1 / s1
    return float(dom1 - dom0)


def altcoin_beta(
    t0: datetime,
    df: pd.DataFrame,
    *,
    alts: list[str] | None = None,
    btc: str = "BTC/USDT",
    window: timedelta = timedelta(hours=48),
) -> float:
    """Mean rolling beta of alt vs BTC returns over window before t0."""
    t0 = _utc(t0)
    alts = alts or ["ETH/USDT", "SOL/USDT"]
    betas: list[float] = []
    btc_hist = _slice(df, btc, t0, window, timedelta(0))
    if btc_hist.empty or len(btc_hist) < 3:
        return 0.0
    r_b = btc_hist["close"].pct_change().dropna().values
    for a in alts:
        ah = _slice(df, a, t0, window, timedelta(0))
        if ah.empty or len(ah) < 3:
            continue
        r_a = ah["close"].pct_change().dropna().values
        n = min(len(r_a), len(r_b))
        if n < 3:
            continue
        x = r_b[-n:]
        y = r_a[-n:]
        var = float(np.var(x)) or 1e-12
        beta = float(np.cov(x, y, bias=True)[0, 1] / var)
        betas.append(beta)
    return float(np.mean(betas)) if betas else 0.0
