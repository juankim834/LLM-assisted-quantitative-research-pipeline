"""Logistic calibration on cluster features — train on labeled data (Phase 2.3)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from crypto_pipeline.schemas.contracts import ClusterObject


def cluster_features(cluster: ClusterObject) -> np.ndarray:
    texts = cluster.representative_texts
    domains = 0.0
    if texts:
        rough = sum(1 for t in texts if "http" in t or "www." in t)
        domains = float(rough) / len(texts)
    n_sources = float(len(set(cluster.representative_ids)))
    asset_n = float(len(cluster.asset_mentions))
    return np.array(
        [
            float(cluster.size),
            n_sources,
            asset_n,
            domains,
        ],
        dtype=np.float64,
    )


class ConfidenceCalibrator:
    """Maps cluster features to calibrated confidence in [0, 1] via Platt-style logistic."""

    def __init__(self) -> None:
        self._clf: LogisticRegression | None = None
        self._scaler = StandardScaler()

    def fit(self, clusters: list[ClusterObject], labels_correct: list[int]) -> None:
        if len(clusters) != len(labels_correct):
            raise ValueError("clusters and labels must align")
        X = np.stack([cluster_features(c) for c in clusters])
        y = np.asarray(labels_correct, dtype=np.int32)
        Xs = self._scaler.fit_transform(X)
        self._clf = LogisticRegression(max_iter=200)
        self._clf.fit(Xs, y)

    def predict_proba(self, cluster: ClusterObject) -> float:
        if self._clf is None:
            return 0.5
        X = self._scaler.transform(cluster_features(cluster).reshape(1, -1))
        p = self._clf.predict_proba(X)[0, 1]
        return float(max(0.0, min(1.0, p)))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps({"scaler": self._scaler, "clf": self._clf}))

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self._clf = None
            return
        data = pickle.loads(p.read_bytes())
        self._scaler = data["scaler"]
        self._clf = data["clf"]
