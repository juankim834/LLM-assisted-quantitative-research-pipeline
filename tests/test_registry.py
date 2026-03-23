import tempfile
from datetime import UTC, datetime
from pathlib import Path

from crypto_pipeline.alpha_registry.registry import AlphaRegistry, AlphaRegistryConfig
from crypto_pipeline.schemas.contracts import Alpha


def test_upsert_and_get_live():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        reg = AlphaRegistry(
            AlphaRegistryConfig(
                database_url=f"sqlite:///{db}",
                enforce_holdout=False,
            )
        )
        a = Alpha(
            alpha_id="hack_eth_test",
            event_type="hack",
            asset="ETH",
            feature_name="feat_hack_eth",
            weight=0.1,
            confidence=0.7,
            backtest_sharpe=0.8,
            backtest_period=(
                datetime(2019, 1, 1, tzinfo=UTC),
                datetime(2019, 12, 31, tzinfo=UTC),
            ),
            status="live",
        )
        reg.upsert_alpha(a)
        live = reg.get_live_alphas()
        assert len(live) == 1
        assert live[0].alpha_id == "hack_eth_test"
        reg.retire_alpha("hack_eth_test", "test")
        assert reg.get_live_alphas() == []
        reg.engine.dispose()
