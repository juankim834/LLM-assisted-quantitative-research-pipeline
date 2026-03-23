# Crypto Dual-Pipeline Roadmap v3
## QR Research Pipeline + Online Trading Pipeline

> **What this system does:** Runs two parallel pipelines sharing the same data sources. The **QR pipeline** (batch, hourly) continuously discovers, validates, and stores news-driven alphas. The **Trading pipeline** (online, real-time) consumes those alphas through a live alpha registry, executes trades via simple rules and ML models, and feeds performance data back to the QR pipeline for continuous refinement. An **agentic LLM loop** actively searches for new alpha opportunities and triggers new hypothesis cycles.
>
> **What this system is not:** A black-box trading bot. Every alpha in the trading pipeline must pass backtested validation before going live. The risk gate is hardcoded rules — it cannot be overridden by any ML model or LLM output.

---

## Architecture Overview

```
                    Shared data sources
        ┌───────────────────────────────────────┐
        │  News stream │ OHLCV data │ Order book │
        └───────┬───────────────┬───────────────┘
                │               │
    ────────────┘               └────────────────────
    │  QR pipeline (batch, hourly)                   │  Trading pipeline (online, real-time)
    │                                                │
    │  Embedding + clustering                        │  Feature engine
    │         ↓                                      │         ↓
    │  LLM event tagger                              │  ML signal model
    │         ↓                                      │         ↓
    │  Alpha hypothesis engine                       │  Position sizer
    │         ↓                                      │         ↓
    │  Alpha validation (backtest)                   │  Risk gate  ← hardcoded rules
    │         ↓                                      │         ↓
    │  Alpha store  ←──── Alpha update bus ─────────→│  Execution engine
    │         ↓             (Redis / Postgres)        │         ↓
    │  LLM summarizer                                │  Trade + PnL logger
    │         ↓                                      │         ↓
    │  Research dashboard                            │  Monitoring dashboard
    └──────────────────┬─────────────────────────────┘
                       │
             Shared feedback loop
             ┌─────────┴──────────┐
             │  Alpha decay monitor│  Signal attribution
             └─────────┬──────────┘
                        │
              LLM alpha search (agentic)
              Actively queries gaps + new events
                        │
              ↑ triggers new QR hypothesis cycle
```

**Cost-control insight (inherited from v2):** Clustering reduces LLM calls from ~500/day (per article) to ~30/day (per cluster). Cluster size itself is a signal — pass it everywhere.

**Core new insight (v3):** The alpha store is the contract between the two pipelines. Neither pipeline knows about the other's internals. They communicate only through the alpha registry schema.

---

## Phases at a Glance

| Phase | Name | Pipeline | Focus | Est. Effort |
|-------|------|----------|-------|-------------|
| 0 | Foundation | Shared | Data infra, schema contracts | 1–2 weeks |
| 1 | Embedding & clustering | QR | Pre-filter before LLM | 2–3 weeks |
| 2 | LLM event tagger | QR | Constrained classification | 2–3 weeks |
| 3 | Router & tool library | QR | Hypothesis branches + quant tools | 3–4 weeks |
| 4 | Alpha store & registry | Shared | The bridge between pipelines | 1–2 weeks |
| 5 | Trading pipeline v1 | Trading | Feature engine + ML model + risk gate | 3–4 weeks |
| 6 | Execution engine | Trading | Order routing + PnL logging | 2–3 weeks |
| 7 | QR agentic loop | QR | LLM-driven alpha search | 2–3 weeks |
| 8 | Feedback loop | Shared | Decay monitoring + signal attribution | 2–3 weeks |
| 9 | Dashboard & summarizer | Shared | Research + trading output layers | 2–3 weeks |
| 10 | Validation & iteration | Both | Does anything actually work? | Ongoing |

**Total estimate: 22–32 weeks**

---

## Phase 0 — Foundation (Shared)

**Goal:** Establish shared data infrastructure and freeze all schema contracts before writing any ML or trading code. Both pipelines depend on this foundation — getting it wrong creates expensive refactoring debt everywhere.

### 0.1 News ingestion
- Primary source: CryptoPanic API (free tier covers research volume)
- Pull fields: `id`, `title`, `url`, `published_at`, `currencies[]`, `source.domain`, `votes`
- Supplement with full article body where accessible; document which sources block scraping
- Deduplication: hash on `(title, published_at)` to catch cross-source reposts
- Language filter: English only for now

### 0.2 Market data feed
- OHLCV at 1h resolution minimum for BTC, ETH, SOL, and target assets
- Tick/order book feed for the trading pipeline: separate ingestion process, stored independently
- All timestamps UTC; all price data from a single exchange (pick one, document it)

### 0.3 Storage
- Time-series store: TimescaleDB or ClickHouse for OHLCV and tick data
- Document store: PostgreSQL with JSONB for raw and processed news
- Embedding store: pgvector on PostgreSQL or lightweight Chroma instance
- Alpha registry: Redis (fast reads for the trading pipeline) with Postgres as the durable backing store

### 0.4 Schema contracts
Freeze these as Pydantic models before Phase 1. They are the interfaces between every stage — changes require updating all consumers.

```python
class RawArticle:
    id: str
    title: str
    body: str | None
    url: str
    published_at: datetime
    source_domain: str
    asset_mentions: list[str]   # canonical tickers, e.g. ["BTC", "ETH"]
    raw_votes: dict

class ClusterObject:
    cluster_id: str
    window_start: datetime
    window_end: datetime
    size: int
    representative_texts: list[str]   # top-k closest to centroid
    representative_ids: list[str]
    centroid_embedding: list[float]
    asset_mentions: list[str]

class EventTag:
    cluster_id: str
    event_type: str             # "listing" | "hack" | "regulatory" | "macro" | "other"
    asset: str | None
    secondary_tag: str | None
    confidence: float           # calibrated, not raw LLM output
    metadata: dict
    tagged_at: datetime

class AnalysisResult:
    event_tag: EventTag
    hypothesis_id: str
    tool_outputs: dict
    computed_at: datetime
    summary: str | None

class Alpha:
    alpha_id: str               # stable identifier, e.g. "hack_eth_4h_abnormal_return"
    version: int
    event_type: str
    asset: str | None
    feature_name: str           # the column name the trading feature engine looks up
    weight: float               # current weight in the ML model
    confidence: float           # from backtest validation
    backtest_sharpe: float
    backtest_period: tuple[datetime, datetime]
    live_since: datetime | None # None = paper trading only
    expires_at: datetime | None # set by alpha decay monitor
    status: str                 # "paper" | "live" | "retired"

class TradeRecord:
    trade_id: str
    alpha_id: str               # which alpha triggered this
    asset: str
    side: str                   # "buy" | "sell"
    size_usd: float
    entry_price: float
    exit_price: float | None
    entry_at: datetime
    exit_at: datetime | None
    realized_pnl: float | None
    signal_snapshot: dict       # feature values at time of trade entry
```

**Exit criteria:** Can ingest 7 days of CryptoPanic data and 7 days of OHLCV data, store them, and query by asset and time range.

---

## Phase 1 — Embedding & Clustering (QR)

**Goal:** Compress a batch of raw articles into a small set of semantic clusters. This is the pre-filter that makes LLM tagging affordable. Unchanged from v2.

### 1.1 Embedding
- Model: `E5-base` or `BGE-base-en` — both run on CPU for batch workloads
- Embed on: `title + body[:500]`
- Normalize to unit length before clustering (required for cosine-based HDBSCAN)
- Cache embeddings — re-embedding the same article is wasteful

### 1.2 Windowing strategy
- Fixed hourly windows for v1
- Minimum window size: fewer than 5 articles → skip clustering, pass directly to tagger

### 1.3 Clustering
- Algorithm: HDBSCAN with `min_cluster_size=3`, cosine metric
- Noise bin (label = -1): if article mentions a high-priority asset, pass as singleton; otherwise discard
- Tune `min_cluster_size` on a sample week before moving to Phase 2

### 1.4 Representative selection
- Top-k=3 articles closest to centroid
- Extract `cluster_size`, `window_start/end`, union of `asset_mentions`
- Pass `cluster_size` to the tagger — it's a signal, not just metadata

### 1.5 Validation (do this before Phase 2)
Manually inspect 2–3 days of clusters. Key questions: Do same-event articles cluster together? Do different events separate cleanly? Are major events appearing in the noise bin?

**Exit criteria:** Given one hour of raw articles, produces a list of `ClusterObject` instances. Manually verified on a 3-day sample.

---

## Phase 2 — LLM Event Tagger (QR)

**Goal:** For each cluster, make one LLM call that reads the representative articles and outputs a structured `EventTag`. The LLM classifies — it does not analyze.

### 2.1 Prompt design
The prompt receives the k=3 representative texts, `cluster_size`, and the union of `asset_mentions`. It must return a JSON-structured response matching the `EventTag` schema. Constrain the output to a fixed taxonomy — do not let the model free-form the `event_type` field.

### 2.2 Event taxonomy (v1)
- `listing` — new exchange listing or token launch
- `hack` — exploit, bridge attack, rug pull
- `regulatory` — SEC action, government ban, legal ruling
- `macro` — Fed decision, CPI print, broader market event
- `partnership` — protocol integration, institutional deal
- `other` — catch-all; review weekly for new taxonomy candidates

### 2.3 Confidence calibration
Do not use the LLM's own confidence estimate — it is poorly calibrated. Instead, train a small logistic regression on cluster features (cluster size, source diversity, asset specificity) against a labeled ground-truth set to produce a calibrated confidence score.

### 2.4 Cost control
With clustering: ~30 LLM calls/day at ~$0.002/call ≈ $0.06/day. Without clustering: ~500 calls/day ≈ $1/day. The clustering step pays for itself in the first month.

**Exit criteria:** Given a `ClusterObject`, produces an `EventTag` with correct event type on >85% of a manually labeled 50-cluster test set.

---

## Phase 3 — Router & Tool Library (QR)

**Goal:** Route each `EventTag` to the appropriate set of quant analysis tools and store the results. Unchanged from v2 in structure; extended with new event types introduced in v3.

### 3.1 Hypothesis map

| Event type | Hypotheses | Key tools |
|------------|-----------|-----------|
| Listing | H1: Price spike then mean-revert | `abnormal_return`, `volume_spike` |
| Hack | H2: Protocol drawdown + contagion | `abnormal_return`, `contagion_spread` |
| Regulatory | H3: Targeted asset underperforms peers | `relative_return`, `regime_shift` |
| Macro | H4: BTC dominance shift | `btc_dominance_change`, `altcoin_beta` |
| Partnership | H5: Positive drift, sustained | `abnormal_return`, `volume_spike` |

### 3.2 Core quant tools
All tools are pure deterministic Python functions. No LLM involvement here.

```python
# Returns price residuals vs. market baseline
abnormal_return(asset, t0, windows=[1h, 4h, 24h, 72h]) -> dict

# Returns z-score of volume relative to 30d rolling mean
volume_spike(asset, t0, window=24h) -> float

# Returns drawdown and recovery for assets correlated with the event asset
contagion_spread(primary_asset, peer_group, t0, window=48h) -> dict

# Returns target asset return minus peer group mean return
relative_return(target_asset, peer_group, t0, windows=[24h, 7d]) -> dict

# Returns correlation structure change before/after event
regime_shift(t0, pre_window=30d, post_window=7d) -> dict

# Returns BTC dominance delta around event
btc_dominance_change(t0, window=72h) -> float

# Returns average alt/BTC beta around event
altcoin_beta(t0, window=48h) -> float
```

### 3.3 Result storage
Each tool returns a JSON dict. The router collects all outputs for a given event into a single `AnalysisResult` stored in the results database.

**Exit criteria:** Each tool runs correctly in isolation on 5 historical events of the appropriate type. Router correctly assigns hypotheses for a manually constructed set of 20 `EventTag` inputs.

---

## Phase 4 — Alpha Store & Registry (Shared)

**Goal:** Build the bridge between the QR pipeline and the trading pipeline. The alpha registry is the only interface between them — neither pipeline knows about the other's internals.

### 4.1 Alpha registry design
- Durable store: Postgres table with the `Alpha` schema defined in Phase 0
- Fast read cache: Redis hash keyed by `alpha_id`, polled by the trading feature engine at startup and on each QR cycle completion
- Write path: QR pipeline writes to Postgres on validation; a sync process populates Redis

### 4.2 Alpha lifecycle
```
Discovered (QR hypothesis engine)
    ↓
Backtested (validation: Sharpe > 0.5 on held-out data required)
    ↓
Paper trading (status = "paper"; trading pipeline uses it with zero real weight)
    ↓
Live (status = "live"; requires 2 weeks paper with positive paper PnL)
    ↓
Retired (status = "retired"; triggered by decay monitor or manual review)
```

### 4.3 Alpha registry API

```python
# Trading pipeline reads this — must be fast
get_live_alphas(asset: str | None = None) -> list[Alpha]

# QR pipeline writes this after validation
upsert_alpha(alpha: Alpha) -> None

# Decay monitor writes this
retire_alpha(alpha_id: str, reason: str) -> None

# Signal attribution reads this
get_alpha_pnl(alpha_id: str, start: datetime, end: datetime) -> pd.Series
```

### 4.4 Anti-overfitting safeguard
Backtest validation must always use a held-out period that the QR pipeline's hypothesis engine has not seen. Maintain a rolling 90-day hold-out window. Any alpha whose backtest period overlaps with the hold-out window is rejected automatically.

**Exit criteria:** QR pipeline can write a validated alpha, trading pipeline can read it within 5 seconds, and the lifecycle transitions are enforced by the API (no direct DB writes from either pipeline).

---

## Phase 5 — Trading Pipeline v1 (Trading)

**Goal:** Build the online trading pipeline: a feature engine that consumes live alphas, an ML model that scores them, a position sizer, and a hardcoded risk gate.

### 5.1 Feature engine
Runs on each new tick or at a configurable polling interval (1-minute minimum).

**Real-time features (always on):**
- Price momentum: returns over [5m, 15m, 1h, 4h]
- Volume z-score: current volume vs. 24h rolling mean
- Bid-ask spread (as a fraction of mid-price)
- Funding rate (for perpetual futures)

**Alpha-driven features (from registry):**
- For each live `Alpha` in the registry: compute the corresponding feature and attach the alpha's `weight` and `confidence` as metadata
- Feature name must match the `feature_name` field in the `Alpha` schema exactly — this is the contract

### 5.2 ML signal model
- Model: LightGBM (fast inference, interpretable feature importance, good on tabular financial data)
- Target: sign of next-N-minute return (binary classification)
- Retrained nightly on the last 90 days of labeled data
- Features: all real-time features + alpha-driven features from the registry
- Output: a directional signal score in [-1, 1]; 0 = no position

Alpha weights in the registry are not directly used as model weights — they are features. The model learns how to combine them. This separation means a new alpha entering the registry does not immediately affect the model; it enters on the next nightly retrain. This is intentional.

### 5.3 Position sizer
- Baseline: fractional Kelly sizing on the signal score
- Scale down by current portfolio volatility (target 10% annualized vol)
- Hard cap: no single position > 5% of total capital
- Output: a signed dollar size for the proposed trade

### 5.4 Risk gate (hardcoded — never overridden by ML)
The risk gate is the last line of defense before any order is sent. It is deterministic rules only.

```python
def risk_gate(proposed_trade: ProposedTrade, portfolio: Portfolio) -> bool:
    if portfolio.daily_drawdown > MAX_DAILY_DRAWDOWN:      # e.g. 3%
        return False
    if proposed_trade.size_usd > MAX_POSITION_SIZE_USD:
        return False
    if market_spread_bps > MAX_SPREAD_BPS:                 # don't trade illiquid markets
        return False
    if portfolio.gross_exposure > MAX_GROSS_EXPOSURE:
        return False
    if current_hour in BLACKOUT_HOURS:                     # e.g. low-liquidity windows
        return False
    return True
```

These thresholds are configuration, not code — store them in a config file that requires a human to edit. No ML model, LLM, or automated process can modify them.

**Exit criteria:** Feature engine produces a feature vector within 500ms of each tick. ML model scores it. Position sizer outputs a dollar size. Risk gate passes or blocks the trade. All steps logged.

---

## Phase 6 — Execution Engine (Trading)

**Goal:** Send orders to the exchange and log all fills and PnL.

### 6.1 Order routing
- Default strategy: TWAP over a 5-minute window for entries, market order for emergency exits
- Slippage model: estimate expected slippage from current order book depth before sending
- If estimated slippage > signal edge, do not send the order (log as "slippage abort")

### 6.2 Position management
- Maintain a local in-memory position ledger reconciled against exchange state every 30 seconds
- Stop-loss: fixed at 1.5× the expected move from the signal model (not a trailing stop in v1)
- Take-profit: optional; default is time-based exit (close position after N hours regardless of PnL)

### 6.3 Trade logging
Every trade produces a `TradeRecord` (see Phase 0 schema). The `signal_snapshot` field stores all feature values at entry — this is what the signal attribution system reads in Phase 8.

### 6.4 Error handling
- Exchange API timeout: retry with exponential backoff (max 3 attempts), then cancel the order and log
- Partial fill: track remaining open size, attempt to complete on next tick
- Position mismatch (local vs. exchange): halt new trades, alert, reconcile manually

**Exit criteria:** Pipeline executes a simulated order on testnet, logs a `TradeRecord`, and reconciles position correctly. Runs for 48 hours on paper trading without crashing.

---

## Phase 7 — QR Agentic Alpha Search (QR)

**Goal:** Replace manual hypothesis discovery with an LLM agent that proactively identifies alpha gaps and queues new hypothesis cycles.

### 7.1 What the agent does
The agentic loop runs once per day (after the human researcher has had a chance to review the prior day's output). It is not autonomous — it queues proposals for human approval before any new alpha is written to the registry.

**Inputs the agent receives:**
- Signal attribution report (which event types are generating PnL, which are flat)
- Alpha registry state (which alphas are live, paper, or recently retired)
- Recent `EventTag` outputs from the tagger (what events are being detected)
- A list of event types that exist in the taxonomy but have no live alphas

**What the agent produces:**
- A ranked list of alpha gap proposals: "Event type X has been detected N times in the last 30 days, but we have no live alpha for it — here is a proposed hypothesis and the historical clusters to backtest it on"
- A set of new `secondary_tag` candidates for event types that are being lumped into "other"
- A summary of which existing alphas are showing signs of decay based on recent live performance

### 7.2 Agent design
The agent uses a fixed set of tools — it does not have free-form internet access. Its tools are:

```python
# Retrieves signal attribution data for a date range
query_alpha_pnl(alpha_id: str, start: datetime, end: datetime) -> dict

# Retrieves recent EventTag outputs filtered by event type
query_event_history(event_type: str, days: int) -> list[EventTag]

# Retrieves all clusters tagged as "other" in the last N days
query_other_clusters(days: int) -> list[ClusterObject]

# Submits a new hypothesis proposal for human review
submit_hypothesis_proposal(proposal: HypothesisProposal) -> str
```

The agent cannot write to the alpha registry directly. It can only submit proposals. A human approves or rejects each proposal before the QR pipeline backtests it.

### 7.3 Prompt constraints
The agent prompt includes explicit constraints:
- Do not propose alphas based on fewer than 20 historical events
- Do not propose alphas for event types with no clear price-impact mechanism
- Do not propose overlapping alphas (if a similar alpha already exists, explain why this one is different)
- All proposals must cite the specific cluster IDs they are based on

**Exit criteria:** Agent runs for 2 weeks, produces at least 3 non-overlapping hypothesis proposals, and at least 1 passes human review and backtesting.

---

## Phase 8 — Feedback Loop (Shared)

**Goal:** Close the loop between live trading performance and the QR pipeline's alpha discovery process.

### 8.1 Signal attribution
For each closed `TradeRecord`, attribute PnL to the alpha that triggered it. Store daily PnL-per-alpha in the alpha registry.

Key questions to answer from attribution:
- Which alphas are consistently generating positive PnL?
- Which alphas have a high signal rate but are being blocked by the risk gate? (Are the risk limits too tight?)
- Which alphas generate positive gross PnL but negative net PnL after fees? (The edge is real but too small to capture.)

### 8.2 Alpha decay monitor
Runs nightly. For each live alpha:
- Compute a rolling 30-day Sharpe on live PnL
- If Sharpe drops below 0.2 for 14 consecutive days: flag for review
- If Sharpe drops below 0 for 7 consecutive days: auto-retire (status → "retired"), alert researcher

Retired alphas are not deleted — they stay in the registry as historical records and can be manually re-activated if conditions change.

### 8.3 Feedback into the QR pipeline
The agentic alpha search (Phase 7) reads the signal attribution output as one of its primary inputs. This creates the feedback loop:

```
Live trade performance
    → signal attribution
        → alpha decay monitor
            → agentic alpha search
                → new hypothesis proposal
                    → backtest
                        → new alpha in registry
                            → live trade performance
```

The key discipline: **the QR pipeline is allowed to know which alphas are working, but the backtest validation must always use held-out data**. The agent cannot tune a hypothesis specifically to fit recent live performance — it must show the alpha worked in a prior period it has not seen.

**Exit criteria:** Attribution correctly attributes PnL to alphas for a 2-week paper trading period. Decay monitor correctly flags a manually injected weak signal within 3 days.

---

## Phase 9 — Dashboard & Summarizer (Shared)

**Goal:** Make both pipelines' outputs human-readable and browsable in a unified interface.

### 9.1 Research dashboard (from QR pipeline)
- Event log: chronological list of tagged events with type, asset, confidence, cluster size
- Event detail view: representative articles, tool output tables, LLM summary
- Alpha pipeline view: all alphas by lifecycle status, backtest metrics, live PnL if applicable
- Hypothesis proposal queue: pending agent proposals awaiting human review

### 9.2 Trading dashboard (from Trading pipeline)
- Live P&L: real-time position value, daily PnL, drawdown gauge
- Active positions: asset, size, entry price, current P&L, triggering alpha
- Signal log: last N signal scores with feature breakdowns (which features drove the score)
- Risk gate log: trades blocked by the gate with reason codes
- Alpha attribution heatmap: which alphas are generating PnL this week

### 9.3 LLM summarizer (QR side only)
- Input: `EventTag` + all `tool_outputs`
- Output: 3–5 sentence plain-English briefing
- Constraint: "Describe what happened, what the tools measured, and what the numbers suggest. Do not make predictions. Do not recommend trades."

Example output:
```
[HACK · ETH · 2024-03-13 06:00 UTC]
Euler Finance was exploited for approximately $197M via a flash loan attack.
The cluster of 84 articles represents high-volume discourse (confidence: 0.89).
Price reaction shows ETH down 4.2% in the 4h window (abnormal return: -3.1% vs BTC).
Contagion analysis identifies AAVE and COMP as correlated drawdowns (-6.3%, -4.8%).
No stablecoin depeg detected. Alpha H2 (hack contagion) was active — live trade entry logged.
```

### 9.4 Stack recommendation
- Backend: FastAPI serving both the results database and the alpha registry
- Frontend: Streamlit for v1 (fast to build); migrate if needed
- Charts: Plotly (works in both Streamlit and standalone)
- Alerts: simple email or Slack webhook for decay monitor triggers and risk gate blocks

**Exit criteria:** A researcher can open the dashboard, review the last 24h of tagged events, inspect the alpha pipeline state, and check live trading performance — all in one interface.

---

## Phase 10 — Validation & Iteration (Both)

**Goal:** Determine whether the system produces accurate alphas and whether the trading pipeline captures them profitably. This phase is ongoing.

### 10.1 QR pipeline validation
- Collect 4 weeks of tagger outputs → manually review a random sample of 50 per week
- Track: Is `event_type` correct? Is `asset` correct? Is `secondary_tag` correct?
- Use disagreements to improve the prompt and retrain the confidence model

### 10.2 Trading pipeline validation
- Paper trade for a minimum of 4 weeks before enabling any live capital
- During paper trading: compare paper PnL against what live PnL would have been (accounting for simulated slippage)
- Required before going live: positive Sharpe > 0.5 on paper trading, maximum drawdown < daily limit for 30 consecutive days

### 10.3 Iteration loop
The natural cycle:
1. QR tagger makes errors on a class of events → improve prompt or add `secondary_tag`
2. A hypothesis shows no consistent pattern in tools → revise the hypothesis or lookback window
3. A new event type appears frequently in "other" clusters → add to taxonomy, build tools, propose as new alpha
4. An alpha's live PnL diverges from paper PnL → investigate slippage model or market impact assumptions
5. Risk gate blocks a disproportionate fraction of good signals → recalibrate thresholds

### 10.4 Reddit as a second source (future)
Once the news pipeline is stable, Reddit (r/CryptoCurrency, r/ethtrader) can be added as a sentiment-reinforcement source. Reddit feeds the sentiment-related tools and the trading feature engine's volume z-score, not the event tagger. Reddit is better for measuring crowd reaction to events than for detecting events themselves.

---

## Key Design Principles

**The alpha store is the only interface between pipelines.**
The QR pipeline and the trading pipeline share no code, no direct database connections, and no function calls. They communicate exclusively through the alpha registry. This means either pipeline can be rewritten, paused, or replaced without affecting the other.

**LLM roles are narrow and explicit.**
The event tagger classifies. The summarizer explains. The agentic search proposes. None of them trade, size positions, or set risk limits. All numerical analysis lives in deterministic Python functions.

**The risk gate is sacred.**
It is hardcoded rules. It cannot be modified by any automated process. It can only be changed by a human editing a config file. No exception.

**Backtest validation uses held-out data, always.**
The QR pipeline cannot tune hypotheses to fit live trading performance. The 90-day hold-out window is enforced by the alpha registry API — not by discipline alone.

**Alpha decay is expected, not a failure.**
Crypto market regimes shift. An alpha that worked for 6 months and then stopped working is a success, not a failure. The decay monitor exists to retire them cleanly, not to prevent decay from happening.

**Everything is logged.**
Every LLM call, every routing decision, every tool run, every trade, every risk gate decision. The log is your training data, your audit trail, and your debugging tool.

---

## Interface Contracts

### What the QR pipeline exposes to the trading pipeline

```python
# Get all live alphas for a given asset
get_live_alphas(asset: str | None = None) -> list[Alpha]

# Get factor time-series for quant model input
get_factor_timeseries(asset: str, factor: str, start: datetime, end: datetime) -> pd.Series
```

### What the trading pipeline exposes to the QR pipeline

```python
# Get trade records attributed to a specific alpha
get_trades_by_alpha(alpha_id: str, start: datetime, end: datetime) -> list[TradeRecord]

# Get daily PnL attributed to a specific alpha
get_alpha_pnl(alpha_id: str, start: datetime, end: datetime) -> pd.Series
```

These four functions are the complete API surface between the two pipelines. Any additional coupling is a design violation.