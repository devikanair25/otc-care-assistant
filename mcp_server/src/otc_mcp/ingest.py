"""Download OTC drug labels from openFDA into data/raw_labels.jsonl.

Usage:
    otc-ingest --ingredients data/otc_ingredients.txt --per-ingredient 5 --out data/raw_labels.jsonl

Runs in CI (GitHub Actions) or on a laptop; it needs internet access to api.fda.gov.
Set OPENFDA_API_KEY for higher rate limits (free key from open.fda.gov).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .chunking import parse_openfda_label
from .clients.openfda import OpenFDAClient
from .config import DEFAULT_DATA_DIR


def read_ingredients(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ingredients", type=Path, default=DEFAULT_DATA_DIR / "otc_ingredients.txt")
    p.add_argument("--per-ingredient", type=int, default=5,
                   help="Max distinct labels kept per ingredient")
    p.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "raw_labels.jsonl")
    p.add_argument("--sleep", type=float, default=0.3, help="Pause between requests (seconds)")
    args = p.parse_args(argv)

    client = OpenFDAClient(api_key=os.getenv("OPENFDA_API_KEY"))
    seen: set[str] = set()
    kept = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for ingredient in read_ingredients(args.ingredients):
            try:
                records = client.otc_labels_for_ingredient(ingredient, limit=25)
            except httpx.HTTPError as exc:
                print(f"  ! {ingredient}: {exc}", file=sys.stderr)
                continue
            n = 0
            for rec in records:
                label = parse_openfda_label(rec)
                if not label or label.set_id in seen:
                    continue
                seen.add(label.set_id)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n >= args.per_ingredient:
                    break
            kept += n
            print(f"  {ingredient:<28} {n} labels")
            time.sleep(args.sleep)
    print(f"Saved {kept} labels to {args.out}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
