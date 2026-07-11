#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import models

from sciver_vector_db.fetching import DEFAULT_QDRANT_PATH
from sciver_vector_db.qdrant_store import make_client, query_points, scroll_all


def get_vectors(point: Any) -> dict[str, list[float]]:
    vectors = getattr(point, "vector", None) or {}
    return vectors if isinstance(vectors, dict) else {}


def summarize_hits(title: str, hits: list[Any]) -> None:
    print(f"\n{title}")
    if not hits:
        print("  (no hits)")
        return
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        score = getattr(hit, "score", None)
        score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(
            "  "
            f"score={score_text} pair_id={payload.get('pair_id')} split={payload.get('split')} "
            f"label={payload.get('label')} modality={payload.get('modality')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nearest-neighbor smoke queries against SciVer Qdrant pairs.")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default="sciver_pairs")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    client = make_client(qdrant_path=args.qdrant_path, qdrant_url=args.qdrant_url)
    first_point = next(scroll_all(client, args.collection, with_vectors=True, batch_size=1), None)
    if first_point is None:
        raise RuntimeError(f"No points found in collection {args.collection!r}")
    vectors = get_vectors(first_point)
    payload = getattr(first_point, "payload", None) or {}
    print(f"Seed pair_id={payload.get('pair_id')} split={payload.get('split')} label={payload.get('label')}")

    filter_val_entailed_chart = models.Filter(
        must=[
            models.FieldCondition(key="split", match=models.MatchValue(value="val")),
            models.FieldCondition(key="label", match=models.MatchValue(value="entailed")),
            models.FieldCondition(key="modality", match=models.MatchValue(value="chart")),
        ]
    )

    for vector_name in ("pair_text_vec", "claim_vec", "image_vec"):
        vector = vectors.get(vector_name)
        if vector is None:
            print(f"\nSkipping {vector_name}: not available on seed point")
            continue
        summarize_hits(
            f"Nearest neighbors by {vector_name}",
            query_points(
                client,
                args.collection,
                vector_name=vector_name,
                vector=vector,
                limit=args.limit,
            ),
        )
        summarize_hits(
            f"Filtered neighbors by {vector_name} (split=val, label=entailed, modality=chart)",
            query_points(
                client,
                args.collection,
                vector_name=vector_name,
                vector=vector,
                limit=args.limit,
                query_filter=filter_val_entailed_chart,
            ),
        )


if __name__ == "__main__":
    main()
