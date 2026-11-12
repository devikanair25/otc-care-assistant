"""Build the FAISS search index from raw openFDA label records.

Usage:
    otc-build-index --raw data/raw_labels.jsonl --out data/index --embedder fastembed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import parse_openfda_label
from .config import DEFAULT_DATA_DIR
from .embeddings import get_embedder
from .label_store import LabelStore


def load_raw_labels(path: Path):
    labels = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                label = parse_openfda_label(json.loads(line))
                if label:
                    labels.append(label)
    return labels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=DEFAULT_DATA_DIR / "raw_labels.jsonl")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "index")
    p.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])
    args = p.parse_args(argv)

    labels = load_raw_labels(args.raw)
    if not labels:
        print(f"No usable labels in {args.raw}")
        return 1
    store = LabelStore.build(labels, get_embedder(args.embedder))
    store.save(args.out)
    print(f"Indexed {len(store.labels)} labels / {len(store.chunks)} chunks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
