"""Load the prebuilt label index (labels + chunks + FAISS vectors) and search it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss

from .chunking import chunk_label, embedding_text
from .embeddings import Embedder
from .models import DrugLabel, LabelChunk, SearchHit, dailymed_url

LABELS_FILE = "labels.jsonl"
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.faiss"
META_FILE = "meta.json"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class LabelStore:
    def __init__(
        self,
        labels: list[DrugLabel],
        chunks: list[LabelChunk],
        index: faiss.Index,
        embedder: Embedder,
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(f"Index has {index.ntotal} vectors but {len(chunks)} chunks")
        self.labels = {lbl.set_id: lbl for lbl in labels}
        self.chunks = chunks
        self.index = index
        self.embedder = embedder

    # ---------- building ----------

    @classmethod
    def build(cls, labels: list[DrugLabel], embedder: Embedder) -> LabelStore:
        chunks = [c for lbl in labels for c in chunk_label(lbl)]
        if not chunks:
            raise ValueError("No searchable chunks: labels have no 'Uses' or 'Purpose' text")
        by_id = {lbl.set_id: lbl for lbl in labels}
        vectors = embedder.embed_documents([embedding_text(by_id[c.set_id], c) for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
        index.add(vectors)
        return cls(labels, chunks, index, embedder)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _write_jsonl(directory / LABELS_FILE, [lbl.model_dump() for lbl in self.labels.values()])
        _write_jsonl(directory / CHUNKS_FILE, [c.model_dump() for c in self.chunks])
        faiss.write_index(self.index, str(directory / VECTORS_FILE))
        meta = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "labels": len(self.labels),
            "chunks": len(self.chunks),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------- loading ----------

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> LabelStore:
        meta_path = directory / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No label index in {directory}. Run `otc-ingest` then `otc-build-index`."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"Index was built with '{meta['embedder']}' but the server is using "
                f"'{embedder.name}'. Rebuild the index or set OTC_EMBEDDER to match."
            )
        labels = [DrugLabel(**r) for r in _read_jsonl(directory / LABELS_FILE)]
        chunks = [LabelChunk(**r) for r in _read_jsonl(directory / CHUNKS_FILE)]
        index = faiss.read_index(str(directory / VECTORS_FILE))
        return cls(labels, chunks, index, embedder)

    # ---------- querying ----------

    def get(self, set_id: str) -> DrugLabel | None:
        return self.labels.get(set_id)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Return the best-matching products, at most one hit per active-ingredient combo."""
        if not query.strip():
            return []
        q = self.embedder.embed_query(query).reshape(1, -1).astype("float32")
        n = min(len(self.chunks), max(top_k * 8, 20))
        scores, idxs = self.index.search(q, n)
        hits: list[SearchHit] = []
        seen: set[tuple[str, ...]] = set()
        for score, i in zip(scores[0], idxs[0], strict=True):
            if i < 0:
                continue
            chunk = self.chunks[i]
            label = self.labels[chunk.set_id]
            combo = tuple(sorted(label.active_ingredients))
            if combo in seen:
                continue
            seen.add(combo)
            hits.append(
                SearchHit(
                    set_id=label.set_id,
                    brand_name=label.brand_name,
                    generic_name=label.generic_name,
                    active_ingredients=label.active_ingredients,
                    matched_section=chunk.section,
                    matched_text=chunk.text,
                    score=round(float(score), 4),
                    source_url=dailymed_url(label.set_id),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def labels_by_ingredient(self, ingredient: str) -> list[DrugLabel]:
        ing = ingredient.lower().strip()
        return [lbl for lbl in self.labels.values() if ing in lbl.active_ingredients]
