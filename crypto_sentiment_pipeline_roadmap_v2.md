# Crypto Sentiment QR Pipeline — Project Roadmap v2

> **What this system does:** Ingests crypto news continuously, compresses it into semantic clusters, uses an LLM to tag each cluster as a financial event, routes each event to a hypothesis-specific analysis toolkit, and surfaces structured results on a dashboard with an LLM-written briefing.
>
> **What this system is not:** A live trading signal or execution system. The output is research intelligence for a quant research workflow.

---

## Architecture Overview

```
Raw news stream
      ↓
  Ingestion & normalization
      ↓
  Embedding + clustering          ← compress 500 articles → 20–30 clusters
      ↓
  Representative selection        ← top-k closest to centroid per cluster
      ↓
  LLM event tagger                ← one LLM call per cluster, not per article
      ↓
  Event router                    ← structured tag → hypothesis branch
      ↓
  Hardcoded analysis tools        ← deterministic quant functions
      ↓
  Structured results store
      ↓
  Dashboard  +  LLM summarizer
```

The key cost-control insight: clustering means the LLM sees ~20–30 representative samples
per batch window, not hundreds of raw articles. Cluster size itself becomes a signal
(a cluster of 80 articles about the same hack is more significant than a cluster of 3).

---

## Phases at a Glance

| Phase | Name | Focus | Est. Effort |
|-------|------|--------|-------------|
| 0 | Foundation | Data infra, schema contracts | 1–2 weeks |
| 1 | Embedding & clustering | Pre-filter before LLM | 2–3 weeks |
| 2 | LLM event tagger | Constrained classification | 2–3 weeks |
| 3 | Router & tool library | Hypothesis branches + quant tools | 3–4 weeks |
| 4 | Agentic loop | Wire pipeline end-to-end | 1–2 weeks |
| 5 | Dashboard & summarizer | Output layer | 2–3 weeks |
| 6 | Validation & iteration | Does anything actually work? | Ongoing |

**Total estimate: 11–17 weeks**

---

## Phase 0 — Foundation

**Goal:** Establish data infrastructure and freeze schema contracts before writing any ML code.
Getting this wrong early creates expensive refactoring debt in every subsequent phase.

### 0.1 News ingestion
- Primary source: CryptoPanic API (free tier covers research volume)
- Pull fields: `id`, `title`, `url`, `published_at`, `currencies[]` (asset tags), `source.domain`, `votes` (upvotes/downvotes as crude quality signal)
- Supplement with full article body where accessible (some sources block scraping — document which ones)
- Deduplication: hash on `(title, published_at)` to catch cross-source reposts of identical headlines
- Language filter: English only for now

### 0.2 Storage
- Time-series store for market data: TimescaleDB or ClickHouse (align on whatever your teammate uses)
- Document store for raw and processed news: PostgreSQL with JSONB columns is fine at research scale
- Embedding store: pgvector extension on PostgreSQL, or a lightweight Chroma instance
- All timestamps in UTC, all price data from a single exchange (pick one, document it)

### 0.3 Market data feed
- OHLCV at 1h resolution minimum for BTC, ETH, SOL, and any other target assets
- Store alongside news with aligned UTC timestamps
- This is what the analysis tools query — it needs to be reliable before Phase 3

### 0.4 Schema contracts
Freeze these as Pydantic models before Phase 1. They are the interfaces between every stage.

```python
class RawArticle:
    id: str
    title: str
    body: str | None
    url: str
    published_at: datetime
    source_domain: str
    asset_mentions: list[str]   # canonical tickers, e.g. ["BTC", "ETH"]
    raw_votes: dict             # from CryptoPanic

class ClusterObject:
    cluster_id: str
    window_start: datetime
    window_end: datetime
    size: int                   # number of articles in cluster
    representative_texts: list[str]   # top-k closest to centroid
    representative_ids: list[str]
    centroid_embedding: list[float]
    asset_mentions: list[str]   # union of mentions across cluster

class EventTag:
    cluster_id: str
    event_type: str             # "listing" | "hack" | "regulatory" | "macro" | "other"
    asset: str | None           # primary asset affected
    secondary_tag: str | None   # e.g. "lending_protocol", "stablecoin", "sec_action"
    confidence: float           # calibrated, not LLM-generated
    metadata: dict              # event-specific fields (amount_usd, protocol, etc.)
    tagged_at: datetime

class AnalysisResult:
    event_tag: EventTag
    hypothesis_id: str          # e.g. "H1", "H3"
    tool_outputs: dict          # structured JSON from each tool that ran
    computed_at: datetime
    summary: str | None         # populated by LLM summarizer in Phase 5
```

**Exit criteria:** Can ingest 7 days of CryptoPanic data, store it, and query by asset and time range.

---

## Phase 1 — Embedding & Clustering

**Goal:** Compress a batch of raw articles into a small set of semantic clusters, each
representing a distinct event or narrative. This is the pre-filter that makes LLM tagging
affordable and avoids the duplicate-article problem.

### 1.1 Embedding
- Model: `E5-base` or `BGE-base-en` — both run fine on CPU for batch workloads
- Embed on: `title + body[:500]` (titles alone lose too much context; full body adds noise)
- Normalize embeddings to unit length before clustering (required for cosine-based HDBSCAN)
- Cache embeddings — re-embedding the same article is wasteful

### 1.2 Windowing strategy
- **Fixed hourly windows** for v1: every hour, cluster the articles published in the last 60 minutes
- Why not sliding windows: more complex, more duplicate cluster handling, not necessary for a research system
- Minimum window size: if fewer than 5 articles arrive in a window, skip clustering and pass articles directly to the tagger (rare but handle it)

### 1.3 Clustering
- Algorithm: HDBSCAN with `min_cluster_size=3`, cosine metric
- HDBSCAN naturally produces an "unclustered" bin (label = -1) for articles that don't fit any cluster — these are noise or genuinely unique events
- Handle noise bin: if a noise article mentions a high-priority asset (BTC, ETH), pass it directly to the tagger as a singleton cluster; otherwise discard
- Tune `min_cluster_size` on a sample week of data — crypto news can be bursty and the default may be too aggressive

### 1.4 Representative selection
- For each cluster, select top-k articles closest to the centroid by cosine distance
- k=3 is a good default: enough context for the LLM, not so much that the prompt bloats
- Also extract: `cluster_size`, `window_start/end`, union of `asset_mentions` across all articles in cluster
- `cluster_size` is itself a feature — pass it to the tagger prompt so the LLM knows whether this is a 3-article cluster or an 80-article cluster

### 1.5 Validation (important — do this before Phase 2)
Manually inspect 2–3 days of clusters. Key questions:
- Do articles about the same event land in the same cluster?
- Do articles about different events (a hack and a listing on the same day) separate cleanly?
- Are major events appearing in the noise bin (bad — lower `min_cluster_size`)?
- Are unrelated articles being merged into one cluster (bad — raise `min_cluster_size`)?

Do not move to Phase 2 until clustering quality is visually acceptable on a sample.

**Exit criteria:** Given one hour of raw articles, produces a list of `ClusterObject` instances
with representative texts selected. Manually verified on a 3-day sample.

---

## Phase 2 — LLM Event Tagger

**Goal:** For each cluster, make one LLM call that reads the representative articles and
outputs a structured `EventTag`. The LLM is doing classification, not analysis.

### 2.1 Prompt design
The prompt receives:
- The k=3 representative article texts
- `cluster_size` (so the model knows the scale of discourse)
- The list of asset mentions across the cluster
- A strict output schema enforced via function calling / JSON mode

The LLM must choose from a closed taxonomy — no free-text event types:

```
event_type:    "listing" | "delisting" | "hack" | "exploit" | "regulatory" |
               "macro" | "partnership" | "other"

secondary_tag: "cex_listing" | "dex_listing" | "lending_protocol" | "bridge" |
               "stablecoin" | "sec_action" | "country_ban" | "fed_decision" |
               "cpi_release" | "other"
               (nullable — only populated when event_type warrants it)

asset:         canonical ticker or null

metadata:      free JSON dict for event-specific fields
               e.g. { "amount_usd": 197000000, "protocol": "Euler Finance" }
```

Crucially: **do not ask the LLM to output a confidence score.** LLM self-reported
confidence is uncalibrated and unreliable.

### 2.2 Confidence model (separate from LLM)
Train a small classifier to predict `P(event_type_label_is_correct)`:

- Features: cluster_size, embedding tightness (mean pairwise cosine distance within cluster),
  asset_mention_count, source_diversity (number of distinct domains), title_similarity_std
- Labels: collect by manually verifying 200–300 historical tagger outputs
- Model: logistic regression or gradient boosted tree — keep it interpretable
- Evaluate with Brier score and a reliability diagram; target Brier score < 0.15

This produces the `confidence` field in `EventTag` — a real calibrated probability.

### 2.3 The "other" category handling
Not every cluster is a financial event. Some clusters will be:
- Opinion pieces / analysis with no new event
- Price commentary ("BTC is going to 100k")
- Unrelated crypto industry news

The tagger should label these `event_type: "other"` with low confidence.
The router (Phase 3) will drop these — they do not trigger any hypothesis tools.
Tune the confidence threshold at which "other" events are suppressed; start at 0.5.

### 2.4 Evaluation
Build a labeled test set of 150–200 historical clusters (manually tagged).
Metrics to track:
- Per-class precision and recall for `event_type`
- Confusion matrix — which event types get confused with each other?
- False positive rate on "other" (how often does the tagger fire on noise?)

Common failure modes to watch for:
- Hack clusters being tagged as "regulatory" (both are negative sentiment events)
- Macro clusters being tagged as "other" (macro news is often indirect — the LLM may not connect CPI to crypto)
- Delist clusters tagged as "listing" (headline framing varies)

**Exit criteria:** On the 150–200 item test set, per-class F1 > 0.75 for all four primary
event types. Confidence model Brier score < 0.15.

---

## Phase 3 — Router & Tool Library

**Goal:** Given a tagged `EventTag`, route to the correct hypothesis branch and run
the appropriate hardcoded analysis tools. All tools are deterministic Python functions —
no LLM involvement in this phase.

### 3.1 Router logic

```python
ROUTING_TABLE = {
    ("listing",    None):                 ["H1", "H2"],
    ("delisting",  None):                 ["H1", "H2"],
    ("hack",       "lending_protocol"):   ["H3", "H4"],
    ("hack",       "bridge"):             ["H3"],
    ("hack",       "stablecoin"):         ["H4"],
    ("hack",       None):                 ["H3"],
    ("regulatory", "sec_action"):         ["H5", "H6"],
    ("regulatory", "country_ban"):        ["H5", "H6"],
    ("macro",      "fed_decision"):       ["H7", "H8"],
    ("macro",      "cpi_release"):        ["H7", "H8"],
    ("macro",      None):                 ["H7"],
}
```

The router uses `(event_type, secondary_tag)` as its key. If a combination is not in the
table, it falls through to a default single-tool "generic reaction" analysis.
Confidence gate: if `EventTag.confidence < 0.5`, do not route — log and skip.

### 3.2 Tool library

All tools share the same interface:
```python
def tool_name(asset: str, t0: datetime, **params) -> dict:
    # fetches market data, computes metric, returns structured JSON result
    ...
```

**Listing / delist tools (H1, H2):**
- `price_reaction(asset, t0, windows=[1,4,24,72])` → abnormal return at each window vs BTC benchmark
- `volume_spike(asset, t0, window=24h)` → volume vs 30d rolling average, expressed as z-score
- `spread_widening(asset, t0)` → bid-ask spread change on primary CEX (proxy: high-low range)
- `dex_vs_cex_volume(asset, t0)` → ratio of DEX to CEX volume before and after event

**Hack / exploit tools (H3, H4):**
- `contagion(primary_asset, related_assets, t0, window=24h)` → correlation of drawdowns across asset group
- `drawdown_ranking(asset_universe, t0, window=24h)` → rank assets by drawdown, identify outliers
- `depeg_check(stablecoin, t0, window=48h)` → max deviation from peg, recovery time
- `protocol_tvl_change(protocol, t0)` → TVL delta from DeFiLlama API (if available)

**Regulatory tools (H5, H6):**
- `regime_shift(t0, pre_window=30d, post_window=7d)` → did market correlation structure change?
- `btc_sp500_corr(t0, window=14d)` → rolling correlation before and after event
- `relative_return(target_asset, peer_group, t0, windows=[24h, 7d])` → target vs peers
- `volume_flight(target_asset, t0, window=48h)` → volume shift away from targeted asset

**Macro tools (H7, H8):**
- `macro_corr(crypto_asset, macro_proxy, t0, window=48h)` → crypto vs SPY/GLD correlation around event
- `btc_dominance_change(t0, window=72h)` → BTC.D change (risk-off tends to increase BTC dominance)
- `altcoin_beta(t0, window=48h)` → average alt/BTC beta around event
- `btc_vs_gold(t0, window=7d)` → relative performance (tests inflation-hedge narrative)

### 3.3 Result schema
Each tool returns a JSON dict. The router collects all tool outputs for a given event
into a single `AnalysisResult` object stored in the results database.

**Exit criteria:** Each tool runs correctly in isolation on 5 historical events of the
appropriate type. Router correctly assigns hypotheses for a manually constructed set of
20 `EventTag` inputs covering all branches.

---

## Phase 4 — Agentic Loop

**Goal:** Wire all stages into a continuous pipeline that runs automatically every hour.

### 4.1 Orchestration
- Scheduler: simple cron job or APScheduler — one run per hour
- Each run:
  1. Pull new articles from CryptoPanic since last run timestamp
  2. Embed new articles
  3. Cluster within current window
  4. For each cluster: call LLM tagger, score confidence, build `EventTag`
  5. Route each `EventTag` to hypothesis tools
  6. Store `AnalysisResult` objects
  7. Trigger dashboard refresh and summarizer (Phase 5)
- Idempotent runs: if a run fails midway, re-running should not duplicate results
  (use the `cluster_id` as an idempotency key)

### 4.2 Error handling
- LLM call failures: retry with exponential backoff (max 3 attempts), then log and skip cluster
- Market data unavailable for a tool: tool returns `{ "error": "data_unavailable", "asset": "...", "t0": "..." }` — do not crash the pipeline
- Clustering produces zero clusters (slow news hour): log and exit gracefully

### 4.3 Logging
- Every LLM call: log prompt, response, latency, token count, cost estimate
- Every routing decision: log `(cluster_id, event_type, confidence, hypotheses_triggered)`
- Every tool run: log inputs, outputs, runtime
- This log is your audit trail and your dataset for improving the tagger over time

**Exit criteria:** Pipeline runs unattended for 48 hours, processes all incoming news,
produces `AnalysisResult` objects, and logs everything without crashing.

---

## Phase 5 — Dashboard & Summarizer

**Goal:** Make the analysis output human-readable and browsable.

### 5.1 LLM summarizer
- Input: `EventTag` + all `tool_outputs` from the corresponding `AnalysisResult`
- Output: a structured briefing in plain English — 3–5 sentences
- Prompt constraint: "Describe what happened, what the tools measured, and what the
  numbers suggest. Do not make predictions. Do not recommend trades."
- The summarizer reads numbers and explains them — it does not do the analysis itself
- Store the summary string back into the `AnalysisResult` object

Example output format:
```
[HACK · ETH · 2024-03-13 06:00 UTC]
Euler Finance was exploited for approximately $197M via a flash loan attack.
The cluster of 84 articles represents high-volume discourse. Price reaction
analysis shows ETH down 4.2% in the 4h window (abnormal return: -3.1% vs BTC).
Contagion analysis identifies AAVE and COMP as correlated drawdowns (-6.3%,
-4.8%). No stablecoin depeg detected. Cluster confidence: 0.89.
```

### 5.2 Dashboard components
- **Event log:** Chronological list of tagged events with type, asset, confidence, cluster size
- **Event detail view:** Click any event → see representative articles, tool output tables, LLM summary
- **Asset timeline:** For a selected asset, show all events affecting it over a date range
- **Tool output charts:** Price reaction chart (CAR curve), volume spike bar, correlation heatmap (for contagion events)
- **Tagger performance panel:** Running precision/recall on labeled subset, confidence distribution

### 5.3 Stack recommendation
- Backend: FastAPI serving the results database
- Frontend: Streamlit for v1 (fast to build, good enough for research); migrate later if needed
- Charts: Plotly (works in both Streamlit and standalone)

**Exit criteria:** A researcher can open the dashboard, see the last 24h of tagged events,
click into any event, read the LLM summary, and inspect the raw tool outputs.

---

## Phase 6 — Validation & Iteration

**Goal:** Determine whether the system is producing accurate tags and meaningful analysis.
This phase is ongoing — it does not end.

### 6.1 Tagger validation
- Collect 4 weeks of tagger outputs → manually review a random sample of 50 per week
- Track: Is the event_type correct? Is the asset correct? Is the secondary_tag correct?
- Use disagreements to improve the prompt and retrain the confidence model

### 6.2 Tool output validation
- For each event type, accumulate 20–30 historical examples
- Run the hypothesis tools on all of them in batch
- Ask: do the tool outputs show a consistent pattern? (e.g. do listing events consistently
  show positive abnormal returns in the 1h window, with mean reversion by 72h?)
- If yes: the hypothesis is supported. Document it.
- If no: either the hypothesis is wrong, or the tagger is mislabeling. Diagnose which.

### 6.3 Iteration loop
The natural cycle is:
1. Find a class of events where the tagger is making errors → improve prompt or add secondary_tag
2. Find a hypothesis where tools show no consistent pattern → revise the hypothesis or the tool's lookback window
3. Find a new event type appearing in "other" clusters → add it to the taxonomy and build tools for it

### 6.4 Reddit as a second source (future)
Once the news pipeline is stable, Reddit (r/CryptoCurrency, r/ethtrader) can be added
as a second ingestion source. Reddit content feeds the sentiment-related tools
(`volume_spike`, `contagion` sentiment variant) rather than the event tagger —
Reddit is better for measuring crowd reaction to events than for detecting events themselves.
This is a Phase 6+ extension, not a Phase 0 requirement.

---

## Key Design Principles

**LLM roles are narrow and explicit.**
The tagger classifies. The summarizer explains. Neither does quant analysis.
All numerical analysis lives in deterministic Python functions.

**Clustering is a cost control mechanism, not just a quality improvement.**
Without it, tagging 500 articles/day at ~$0.002/call = $1/day = $365/year.
With clustering reducing to ~30 calls/day = ~$0.06/day. At scale this matters.

**Cluster size is a signal.**
A cluster of 80 articles about a regulatory action carries more weight than a cluster
of 4. Pass `cluster_size` to the tagger prompt and store it in `EventTag`.
Downstream, weight analysis results by cluster size when aggregating over time.

**Schema contracts are enforced from Day 1.**
Every stage communicates through the frozen Pydantic models defined in Phase 0.
If a stage needs a new field, add it to the schema and update all consumers — do not
pass ad-hoc dicts between stages.

**Everything is logged.**
Every LLM call, every routing decision, every tool run. The log is your training data
for the confidence model and your audit trail for debugging.

---

## What Your Teammate Needs From You

Your deliverable to the shared quant pipeline is a queryable results store with this interface:

```python
# Get all events for an asset in a date range
get_events(asset="ETH", start=..., end=...) -> list[EventTag]

# Get full analysis result for an event
get_analysis(cluster_id=...) -> AnalysisResult

# Get aggregated factor values for quant model
get_factor_timeseries(asset="BTC", factor="abnormal_return_24h", start=..., end=...)
-> pd.Series
```

The factor time-series query is the most important one — it's what lets your teammate
plug your event-driven features into their factor model without understanding your pipeline internals.
