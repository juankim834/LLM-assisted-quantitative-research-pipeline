"""E5/BGE embeddings on title + body[:500], L2-normalized (Phase 1.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from crypto_pipeline.schemas.contracts import RawArticle

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def embed_text_for_article(a: RawArticle, body_max: int = 500) -> str:
    body = (a.body or "")[:body_max]
    return f"{a.title.strip()}\n{body}".strip()


def embed_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return np.asarray(emb, dtype=np.float32)


class EmbeddingCache:
    """Disk cache keyed by (model, text hash) to avoid re-embedding."""

    def __init__(self, path: Path | str, model_name: str = DEFAULT_MODEL) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self.model_name}|{text}".encode()).hexdigest()
        return h

    def get(self, text: str) -> list[float] | None:
        return self._data.get(self._key(text))

    def set(self, text: str, vector: Sequence[float]) -> None:
        self._data[self._key(text)] = list(map(float, vector))
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def embed_missing(
        self,
        texts: list[str],
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        missing_idx: list[int] = []
        vecs: list[list[float] | None] = [None] * len(texts)
        for i, t in enumerate(texts):
            v = self.get(t)
            if v is not None:
                vecs[i] = v
            else:
                missing_idx.append(i)
        if missing_idx:
            to_compute = [texts[i] for i in missing_idx]
            new_emb = embed_texts(to_compute, model_name=self.model_name, device=device, batch_size=batch_size)
            for j, row_idx in enumerate(missing_idx):
                vec = new_emb[j].tolist()
                self.set(texts[row_idx], vec)
                vecs[row_idx] = vec
        dim = len(vecs[0] or [])
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v is None:
                raise RuntimeError("embedding cache inconsistency")
            out[i] = np.asarray(v, dtype=np.float32)
        return out
