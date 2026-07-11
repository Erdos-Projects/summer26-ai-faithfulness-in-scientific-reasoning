#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sciver_vector_db.fetching import DEFAULT_PROCESSED_DIR, DEFAULT_QDRANT_PATH, DEFAULT_RAW_DIR


def run_step(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch, convert, smoke-query, and export SciVer features.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--collection-prefix", default="sciver")
    parser.add_argument("--modalities", nargs="+", default=["chart", "table"])
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-image-embeddings", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    fetch_cmd = [python, "scripts/fetch_sciver.py", "--raw-dir", str(args.raw_dir)]
    if args.force_fetch:
        fetch_cmd.append("--force")
    run_step(fetch_cmd)
    run_step([python, "scripts/inspect_sciver.py", "--data-root", str(args.raw_dir / "SciVer")])

    build_cmd = [
        python,
        "scripts/build_sciver_qdrant.py",
        "--data-root",
        str(args.raw_dir / "SciVer"),
        "--processed-dir",
        str(args.processed_dir),
        "--qdrant-path",
        str(args.qdrant_path),
        "--collection-prefix",
        args.collection_prefix,
        "--modalities",
        *args.modalities,
    ]
    if args.reset:
        build_cmd.append("--reset")
    if args.no_image_embeddings:
        build_cmd.append("--no-image-embeddings")
    run_step(build_cmd)
    run_step(
        [
            python,
            "scripts/query_sciver_qdrant_smoke.py",
            "--qdrant-path",
            str(args.qdrant_path),
            "--collection",
            f"{args.collection_prefix}_pairs",
            "--limit",
            "5",
        ]
    )
    run_step(
        [
            python,
            "scripts/export_pair_features.py",
            "--qdrant-path",
            str(args.qdrant_path),
            "--collection",
            f"{args.collection_prefix}_pairs",
            "--out",
            str(args.processed_dir / "sciver_pair_features.parquet"),
        ]
    )


if __name__ == "__main__":
    main()
