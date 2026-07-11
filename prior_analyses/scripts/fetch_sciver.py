#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sciver_vector_db.fetching import DEFAULT_DATASET_ID, DEFAULT_RAW_DIR, fetch_sciver_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the SciVer dataset from Hugging Face into data/raw.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory that will contain SciVer/.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset id.")
    parser.add_argument("--revision", default="main", help="Dataset revision, branch, or commit.")
    parser.add_argument("--force", action="store_true", help="Remove an existing data/raw/SciVer before fetching.")
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, Hugging Face environment/auth cache is used.",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    result = fetch_sciver_dataset(
        raw_dir=args.raw_dir,
        dataset_id=args.dataset_id,
        revision=args.revision,
        force=args.force,
        token=token,
    )
    print(f"Fetched {result.dataset_id}@{result.revision}")
    print(f"Local path: {result.local_path}")
    print(f"Files found: {len(result.files_found)}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
