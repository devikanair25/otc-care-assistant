"""Embedding backends.

`FastEmbedEmbedder` is used in production: a small ONNX model (bge-small) with no PyTorch,
so the Cloud Run image stays small and runs on CPU.

`HashingEmbedder` is deterministic and needs no model download. Tests and offline runs use it.
It captures keyword overlap only, so search quality is lower than the real model.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype("float32")


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams, L2-normalized."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        v = np.zeros(self.dim, dtype="float32")
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 == 0 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text)[None, :])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        # The Docker image pre-downloads the model into this folder at build time.
        self._model = TextEmbedding(model_name=model_name,
                                    cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        self.name = f"fastembed:{model_name}"
        self.dim = len(next(iter(self._model.query_embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.vstack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.vstack(list(self._model.query_embed([text]))))[0]


def get_embedder(kind: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    raise ValueError(f"Unknown embedder '{kind}'. Use 'fastembed' or 'hashing'.")
