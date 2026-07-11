#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from qdrant_client import models
from tqdm import tqdm

from sciver_vector_db.embeddings import ImageEmbedder, TextEmbedder
from sciver_vector_db.fetching import DEFAULT_PROCESSED_DIR, DEFAULT_QDRANT_PATH, DEFAULT_RAW_DATASET_DIR
from sciver_vector_db.parsing import (
    build_pair_records,
    counts_for_examples,
    find_split_files,
    load_split_examples,
    pair_manifest_row,
)
from sciver_vector_db.qdrant_store import (
    create_payload_indexes,
    make_client,
    point_uuid,
    recreate_collection,
    vector_list,
)


FEATURE_VERSION = "sciver_vector_db_v1"
INDEX_FIELDS = {
    "split": "keyword",
    "label": "keyword",
    "label_id": "integer",
    "paperid": "keyword",
    "claim_type": "keyword",
    "modality": "keyword",
    "pair_mode": "keyword",
    "gold_pair": "bool",
    "n_evidence_items": "integer",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def upsert_batches(client, collection_name: str, points: list[models.PointStruct], batch_size: int) -> None:
    for start in tqdm(range(0, len(points), batch_size), desc=f"Upserting {collection_name}"):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qdrant vector DB from local SciVer data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_RAW_DATASET_DIR,
        help="Local SciVer dataset root. Defaults to data/raw/SciVer.",
    )
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH, help="Qdrant local-mode path.")
    parser.add_argument("--qdrant-url", default=None, help="Optional Qdrant server URL. Overrides --qdrant-path.")
    parser.add_argument("--collection-prefix", default="sciver", help="Collection prefix.")
    parser.add_argument("--modalities", nargs="+", default=["chart", "table"], help="Allowed modalities, e.g. chart table.")
    parser.add_argument(
        "--pair-mode",
        choices=["single_visual_only", "explode_visuals", "aggregate_visuals"],
        default="single_visual_only",
    )
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--image-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--no-image-embeddings", action="store_true")
    parser.add_argument("--allow-missing-image-vec", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--upsert-batch-size", type=int, default=128)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate target collections.")
    args = parser.parse_args()

    split_files = find_split_files(args.data_root)
    examples, parser_skips = load_split_examples(args.data_root, split_files)
    pairs, pair_skips = build_pair_records(
        examples,
        modalities=set(args.modalities),
        pair_mode=args.pair_mode,
    )
    skip_counts = Counter(parser_skips)
    skip_counts.update(pair_skips)
    if not pairs:
        raise RuntimeError(
            "No usable SciVer chart/table-claim pairs were found. Run scripts/inspect_sciver.py "
            "and check modalities, image paths, and skip reasons."
        )

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.processed_dir / "embedding_cache"

    if not args.no_image_embeddings:
        image_embedder = ImageEmbedder(args.image_model, cache_root=cache_root, batch_size=args.image_batch_size)
        image_vectors, image_failures = image_embedder.encode([record.image_path for record in pairs])
        if image_failures:
            skip_counts["image_embedding_failed"] += len(image_failures)
            if not args.allow_missing_image_vec:
                failed_paths = set(image_failures)
                pairs = [record for record in pairs if record.image_path not in failed_paths]
                if not pairs:
                    raise RuntimeError("All usable pairs failed image embedding.")
        embedding_model_image = args.image_model
    else:
        image_vectors = {}
        image_failures = {}
        embedding_model_image = ""

    text_embedder = TextEmbedder(args.text_model, cache_root=cache_root, batch_size=args.batch_size)
    texts: list[str] = []
    for record in pairs:
        texts.extend([record.claim_text(), record.evidence_text(), record.pair_text()])
    text_vectors = text_embedder.encode(texts)

    claim_collection = f"{args.collection_prefix}_claims"
    evidence_collection = f"{args.collection_prefix}_evidence_items"
    pair_collection = f"{args.collection_prefix}_pairs"
    collections = {
        "claims": claim_collection,
        "evidence_items": evidence_collection,
        "pairs": pair_collection,
    }

    claim_dim = len(next(iter(text_vectors.values())))
    evidence_text_dim = claim_dim
    pair_text_dim = claim_dim
    image_dim = 0
    if not args.no_image_embeddings and image_vectors:
        image_dim = len(next(iter(image_vectors.values())))

    client = make_client(qdrant_path=args.qdrant_path, qdrant_url=args.qdrant_url)
    recreate_collection(client, claim_collection, {"claim_vec": claim_dim}, reset=args.reset)
    evidence_vectors_config = {"evidence_text_vec": evidence_text_dim}
    pair_vectors_config = {
        "claim_vec": claim_dim,
        "evidence_text_vec": evidence_text_dim,
        "pair_text_vec": pair_text_dim,
    }
    if not args.no_image_embeddings and image_dim:
        evidence_vectors_config["image_vec"] = image_dim
        pair_vectors_config["image_vec"] = image_dim
    recreate_collection(client, evidence_collection, evidence_vectors_config, reset=args.reset)
    recreate_collection(client, pair_collection, pair_vectors_config, reset=args.reset)
    for collection in (claim_collection, evidence_collection, pair_collection):
        create_payload_indexes(client, collection, INDEX_FIELDS)

    claim_rows: dict[str, dict[str, Any]] = {}
    evidence_rows: dict[str, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    claim_points: dict[str, models.PointStruct] = {}
    evidence_points: dict[str, models.PointStruct] = {}
    pair_points: list[models.PointStruct] = []

    for record in pairs:
        evidence_item_ids_json = json.dumps([record.evidence_item_id])
        claim_payload = {
            "claim_id": record.claim_id,
            "paperid": record.paperid,
            "request_id": record.request_id,
            "split": record.split,
            "claim": record.claim,
            "claim_type": record.claim_type,
            "label": record.label,
            "label_id": record.label_id,
            "evidence_item_ids_json": evidence_item_ids_json,
            "feature_version": FEATURE_VERSION,
            "embedding_model_text": args.text_model,
        }
        claim_rows.setdefault(record.claim_id, claim_payload)
        claim_points.setdefault(
            record.claim_id,
            models.PointStruct(
                id=point_uuid(record.claim_id),
                vector={"claim_vec": vector_list(text_vectors[record.claim_text()])},
                payload=claim_payload,
            ),
        )

        evidence_payload = {
            "evidence_item_id": record.evidence_item_id,
            "paperid": record.paperid,
            "split": record.split,
            "modality": record.modality,
            "item_key": record.item_key,
            "image_path": str(record.image_path),
            "caption": record.caption,
            "ocr_text": record.ocr_text,
            "context_text": record.context_text,
            "section_json": record.section_json,
            "paper_path": record.paper_path,
            "feature_version": FEATURE_VERSION,
            "embedding_model_text": args.text_model,
            "embedding_model_image": embedding_model_image,
        }
        evidence_vector = {"evidence_text_vec": vector_list(text_vectors[record.evidence_text()])}
        if not args.no_image_embeddings and record.image_path in image_vectors:
            evidence_vector["image_vec"] = vector_list(image_vectors[record.image_path])
        evidence_rows.setdefault(record.evidence_item_id, evidence_payload)
        evidence_points.setdefault(
            record.evidence_item_id,
            models.PointStruct(
                id=point_uuid(record.evidence_item_id),
                vector=evidence_vector,
                payload=evidence_payload,
            ),
        )

        pair_payload = {
            "pair_id": record.pair_id,
            "claim_id": record.claim_id,
            "evidence_item_id": record.evidence_item_id,
            "paperid": record.paperid,
            "request_id": record.request_id,
            "split": record.split,
            "claim": record.claim,
            "label": record.label,
            "label_id": record.label_id,
            "claim_type": record.claim_type,
            "modality": record.modality,
            "image_path": str(record.image_path),
            "section_json": record.section_json,
            "n_evidence_items": record.n_evidence_items,
            "gold_pair": record.gold_pair,
            "pair_mode": record.pair_mode,
            "feature_version": FEATURE_VERSION,
            "embedding_model_text": args.text_model,
            "embedding_model_image": embedding_model_image,
        }
        pair_vector = {
            "claim_vec": vector_list(text_vectors[record.claim_text()]),
            "evidence_text_vec": vector_list(text_vectors[record.evidence_text()]),
            "pair_text_vec": vector_list(text_vectors[record.pair_text()]),
        }
        if not args.no_image_embeddings and record.image_path in image_vectors:
            pair_vector["image_vec"] = vector_list(image_vectors[record.image_path])
        pair_points.append(
            models.PointStruct(id=point_uuid(record.pair_id), vector=pair_vector, payload=pair_payload)
        )
        row = pair_manifest_row(record)
        row.update(
            {
                "feature_version": FEATURE_VERSION,
                "embedding_model_text": args.text_model,
                "embedding_model_image": embedding_model_image,
            }
        )
        pair_rows.append(row)

    upsert_batches(client, claim_collection, list(claim_points.values()), args.upsert_batch_size)
    upsert_batches(client, evidence_collection, list(evidence_points.values()), args.upsert_batch_size)
    upsert_batches(client, pair_collection, pair_points, args.upsert_batch_size)

    pd.DataFrame(claim_rows.values()).to_parquet(args.processed_dir / "sciver_claims_manifest.parquet", index=False)
    pd.DataFrame(evidence_rows.values()).to_parquet(
        args.processed_dir / "sciver_evidence_items_manifest.parquet", index=False
    )
    pd.DataFrame(pair_rows).to_parquet(args.processed_dir / "sciver_pairs_manifest.parquet", index=False)

    example_counts = counts_for_examples(examples)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "files_used": {split: str(path) for split, path in split_files.items()},
        "text_model": args.text_model,
        "image_model": embedding_model_image,
        "collection_names": collections,
        "vector_dimensions": {
            "claim_vec": claim_dim,
            "evidence_text_vec": evidence_text_dim,
            "pair_text_vec": pair_text_dim,
            "image_vec": image_dim,
        },
        "total_rows_loaded": len(examples),
        "total_records_inserted": {
            "claims": len(claim_points),
            "evidence_items": len(evidence_points),
            "pairs": len(pair_points),
        },
        "counts_by_split": Counter(record.split for record in pairs),
        "counts_by_label": Counter(record.label for record in pairs),
        "counts_by_modality": Counter(record.modality for record in pairs),
        "counts_by_claim_type": Counter(record.claim_type or "unknown" for record in pairs),
        "source_counts_by_split": example_counts["split"],
        "skip_counts_and_reasons": skip_counts,
        "image_embedding_failures": {str(path): reason for path, reason in image_failures.items()},
        "cli_arguments": vars(args),
    }
    write_json(args.processed_dir / "sciver_build_report.json", report)

    print(f"Inserted {len(pair_points)} pair records into {pair_collection}")
    print(f"Label distribution: {dict(Counter(record.label for record in pairs))}")
    print(f"Modality distribution: {dict(Counter(record.modality for record in pairs))}")
    print(f"Vector dimensions: {report['vector_dimensions']}")
    print(f"Skipped records: {dict(skip_counts)}")


if __name__ == "__main__":
    main()
