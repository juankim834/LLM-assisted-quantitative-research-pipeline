"""HDBSCAN clustering + representatives (Phase 1.3–1.4)."""

from __future__ import annotations

import uuid
from datetime import datetime

import hdbscan
import numpy as np

from crypto_pipeline.qr.embedding import embed_text_for_article, embed_texts
from crypto_pipeline.qr.windows import MIN_CLUSTER_ARTICLES
from crypto_pipeline.schemas.contracts import ClusterObject, RawArticle

HIGH_PRIORITY_ASSETS = {"BTC", "ETH", "SOL"}


def _centroid(embeddings: np.ndarray) -> np.ndarray:
    return np.mean(embeddings, axis=0)


def _cluster_object_from_labels(
    articles: list[RawArticle],
    embeddings: np.ndarray,
    labels: np.ndarray,
    window_start: datetime,
    window_end: datetime,
    *,
    top_k: int = 3,
) -> list[ClusterObject]:
    out: list[ClusterObject] = []
    for lab in sorted(set(labels.tolist())):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        sub = embeddings[idx]
        art = [articles[i] for i in idx.tolist()]
        c = _centroid(sub)
        c_norm = c / (np.linalg.norm(c) + 1e-12)
        sims = sub @ c_norm
        order = np.argsort(-sims)[:top_k]
        rep_ids = [art[int(i)].id for i in order]
        rep_texts = [embed_text_for_article(art[int(i)]) for i in order]
        assets: set[str] = set()
        for a in art:
            assets.update(a.asset_mentions)
        cid = f"w{int(window_start.timestamp())}_c{int(lab)}_{uuid.uuid4().hex[:8]}"
        out.append(
            ClusterObject(
                cluster_id=cid,
                window_start=window_start,
                window_end=window_end,
                size=len(art),
                representative_texts=rep_texts,
                representative_ids=rep_ids,
                centroid_embedding=c_norm.astype(float).tolist(),
                asset_mentions=sorted(assets),
            )
        )
    return out


def _noise_singletons(
    articles: list[RawArticle],
    embeddings: np.ndarray,
    labels: np.ndarray,
    window_start: datetime,
    window_end: datetime,
) -> list[ClusterObject]:
    out: list[ClusterObject] = []
    noise_idx = np.where(labels == -1)[0]
    for i in noise_idx.tolist():
        a = articles[int(i)]
        if not (set(a.asset_mentions) & HIGH_PRIORITY_ASSETS):
            continue
        emb = embeddings[int(i)]
        vec = emb / (np.linalg.norm(emb) + 1e-12)
        cid = f"w{int(window_start.timestamp())}_n_{a.id}_{uuid.uuid4().hex[:6]}"
        out.append(
            ClusterObject(
                cluster_id=cid,
                window_start=window_start,
                window_end=window_end,
                size=1,
                representative_texts=[embed_text_for_article(a)],
                representative_ids=[a.id],
                centroid_embedding=vec.astype(float).tolist(),
                asset_mentions=list(a.asset_mentions),
            )
        )
    return out


def cluster_hourly_articles(
    articles: list[RawArticle],
    window_start: datetime,
    window_end: datetime,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 1,
    model_name: str | None = None,
    device: str | None = None,
) -> list[ClusterObject]:
    """
    If fewer than MIN_CLUSTER_ARTICLES articles, skip clustering — return one synthetic
    cluster per article (single-pass to tagger) as single-article clusters.
    """
    from crypto_pipeline.qr.embedding import DEFAULT_MODEL

    mname = model_name or DEFAULT_MODEL
    if len(articles) < MIN_CLUSTER_ARTICLES:
        out: list[ClusterObject] = []
        for a in articles:
            emb = embed_texts([embed_text_for_article(a)], model_name=mname, device=device)
            vec = emb[0] / (np.linalg.norm(emb[0]) + 1e-12)
            cid = f"w{int(window_start.timestamp())}_direct_{a.id}"
            out.append(
                ClusterObject(
                    cluster_id=cid,
                    window_start=window_start,
                    window_end=window_end,
                    size=1,
                    representative_texts=[embed_text_for_article(a)],
                    representative_ids=[a.id],
                    centroid_embedding=vec.astype(float).tolist(),
                    asset_mentions=list(a.asset_mentions),
                )
            )
        return out

    texts = [embed_text_for_article(a) for a in articles]
    emb = embed_texts(texts, model_name=mname, device=device)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(emb)
    clusters = _cluster_object_from_labels(
        articles, emb, labels, window_start, window_end, top_k=3
    )
    clusters.extend(_noise_singletons(articles, emb, labels, window_start, window_end))
    return clusters
