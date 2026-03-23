"""LLM event tagger — constrained JSON taxonomy (Phase 2)."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from openai import OpenAI

from crypto_pipeline.qr.calibration import ConfidenceCalibrator
from crypto_pipeline.qr.taxonomy import EVENT_TYPE_SET, EVENT_TYPES
from crypto_pipeline.schemas.contracts import ClusterObject, EventTag

TAGGER_SYSTEM = """You are an event classifier for crypto news clusters.
Return a single JSON object with keys: event_type, asset, secondary_tag, metadata.
- event_type must be one of: """ + ", ".join(EVENT_TYPES) + """.
- asset: primary ticker like BTC, ETH, or null if unclear.
- secondary_tag: short string or null.
- metadata: object with any extra structured notes (no prose outside JSON).
Do not include confidence in your output."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


class LLMEventTagger:
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        calibrator: ConfidenceCalibrator | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator or ConfidenceCalibrator()
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def tag_cluster(self, cluster: ClusterObject) -> EventTag:
        user = {
            "cluster_id": cluster.cluster_id,
            "cluster_size": cluster.size,
            "asset_mentions": cluster.asset_mentions,
            "representative_texts": cluster.representative_texts,
        }
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": TAGGER_SYSTEM},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        raw = _extract_json(content)
        et = str(raw.get("event_type", "other")).lower()
        if et not in EVENT_TYPE_SET:
            et = "other"
        asset = raw.get("asset")
        secondary = raw.get("secondary_tag")
        meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        conf = self.calibrator.predict_proba(cluster)
        return EventTag(
            cluster_id=cluster.cluster_id,
            event_type=et,
            asset=str(asset).upper() if asset else None,
            secondary_tag=str(secondary) if secondary else None,
            confidence=conf,
            metadata=meta,
            tagged_at=datetime.now(UTC),
        )
