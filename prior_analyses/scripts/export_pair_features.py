#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from sciver_vector_db.fetching import DEFAULT_PROCESSED_DIR, DEFAULT_QDRANT_PATH
from sciver_vector_db.qdrant_store import make_client, scroll_all


VECTOR_NAMES = ("claim_vec", "evidence_text_vec", "pair_text_vec", "image_vec")


def named_vectors(point: Any) -> dict[str, list[float]]:
    vectors = getattr(point, "vector", None) or {}
    if isinstance(vectors, dict):
        return {name: vectors.get(name) for name in VECTOR_NAMES if vectors.get(name) is not None}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SciVer pair vectors and metadata from Qdrant.")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default="sciver_pairs")
    parser.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR / "sciver_pair_features.parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    client = make_client(qdrant_path=args.qdrant_path, qdrant_url=args.qdrant_url)
    rows = []
    for point in tqdm(
        scroll_all(client, args.collection, with_vectors=True, batch_size=args.batch_size),
        desc=f"Exporting {args.collection}",
    ):
        payload = dict(getattr(point, "payload", None) or {})
        vectors = named_vectors(point)
        row = dict(payload)
        for name in VECTOR_NAMES:
            row[name] = vectors.get(name)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No points found in collection {args.collection!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"Exported {len(rows)} pair rows to {args.out}")
    print("Vector columns: " + ", ".join(name for name in VECTOR_NAMES if any(row.get(name) is not None for row in rows)))


if __name__ == "__main__":
    main()
