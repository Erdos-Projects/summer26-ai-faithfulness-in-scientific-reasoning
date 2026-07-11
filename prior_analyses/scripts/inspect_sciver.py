#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sciver_vector_db.fetching import DEFAULT_RAW_DATASET_DIR
from sciver_vector_db.parsing import build_pair_records, counts_for_examples, find_split_files, load_split_examples


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}")
    if not counter:
        print("  (none)")
        return
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local SciVer JSON files and visual-evidence usability.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_RAW_DATASET_DIR,
        help="Local SciVer dataset root. Defaults to data/raw/SciVer.",
    )
    parser.add_argument("--modalities", nargs="+", default=["chart", "table"], help="Modalities to count as usable.")
    args = parser.parse_args()

    split_files = find_split_files(args.data_root)
    print("Located split files:")
    for split, path in split_files.items():
        print(f"  {split}: {path}")

    examples, parser_skips = load_split_examples(args.data_root, split_files)
    counts = counts_for_examples(examples)

    print_counter("Counts by original split", counts["split"])
    print_counter("Counts by label", counts["label"])
    print_counter("Counts by claim_type / reasoning type", counts["claim_type"])
    print_counter("Counts by detected evidence modality", counts["modality"])
    if parser_skips:
        print_counter("Parser-level skipped rows", parser_skips)

    pairs, pair_skips = build_pair_records(
        examples,
        modalities=set(args.modalities),
        pair_mode="single_visual_only",
    )
    print(f"\nUsable single_visual_only examples ({', '.join(args.modalities)}): {len(pairs)}")
    print_counter("single_visual_only skip reasons", pair_skips)


if __name__ == "__main__":
    main()
