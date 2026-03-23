"""CLI entry: `python -m crypto_pipeline ingest-news | ingest-ohlcv` (Phase 0 helpers)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from crypto_pipeline.ingestion.dedup import article_dedup_key, deduplicate_articles
from crypto_pipeline.ingestion.language import filter_english_articles
from crypto_pipeline.ingestion.ohlcv import OHLCVConfig, ohlcv_last_n_days
from crypto_pipeline.ingestion.storage import StorageConfig, StorageService


def _cmd_ingest_news(_: argparse.Namespace) -> None:
    from crypto_pipeline.ingestion.cryptopanic import CryptoPanicClient

    client = CryptoPanicClient()
    raw = client.fetch_posts(max_pages=int(os.environ.get("CRYPTOPANIC_MAX_PAGES", "5")))
    raw = filter_english_articles(raw)
    raw = deduplicate_articles(raw)
    storage = StorageService(StorageConfig(database_url=os.environ.get("DATABASE_URL", "sqlite:///./data/pipeline.db")))
    Path("./data").mkdir(parents=True, exist_ok=True)
    keys = [article_dedup_key(a.title, a.published_at) for a in raw]
    n = storage.upsert_raw_articles(raw, keys)
    print(f"upserted {n} articles (batch size {len(raw)})")


def _cmd_ingest_ohlcv(ns: argparse.Namespace) -> None:
    syms = [s.strip() for s in ns.symbols.split(",") if s.strip()]
    df = ohlcv_last_n_days(syms, days=ns.days, config=OHLCVConfig())
    storage = StorageService(StorageConfig(database_url=os.environ.get("DATABASE_URL", "sqlite:///./data/pipeline.db")))
    Path("./data").mkdir(parents=True, exist_ok=True)
    n = storage.insert_ohlcv_dataframe_rows(df.to_dict("records"))
    print(f"inserted {n} OHLCV rows for {syms} ({ns.days} days)")


def main() -> None:
    p = argparse.ArgumentParser(prog="crypto_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_news = sub.add_parser("ingest-news", help="CryptoPanic → SQLite/Postgres (needs CRYPTOPANIC_API_KEY)")
    p_news.set_defaults(func=_cmd_ingest_news)

    p_ohlcv = sub.add_parser("ingest-ohlcv", help="CCXT Binance 1h OHLCV")
    p_ohlcv.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated CCXT symbols")
    p_ohlcv.add_argument("--days", type=int, default=7)
    p_ohlcv.set_defaults(func=_cmd_ingest_ohlcv)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
