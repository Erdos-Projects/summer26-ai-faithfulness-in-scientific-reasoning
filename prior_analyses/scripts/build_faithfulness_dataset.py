#!/usr/bin/env python
"""Build a combined, one-row-per-item dataset from the annotation and prediction databases.

Inputs (SQLite):
  - annotations_prod.db : claim/evidence items + per-dimension difficulty scores
  - predictions.db      : repeated LLM entailed/refuted predictions per item

Output (one row per item):
  item metadata + 13 difficulty-dimension scores (wide) + a summary of the LLM
  predictions from the chosen target run.

The eventual modeling target is whether the LLM correctly judged the claim:
  - pred_frac_correct     (continuous, fraction of trials correct)
  - pred_majority_correct (binary, 1 if the majority of trials were correct)
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# Canonical order of the 13 difficulty dimensions (short codes used in the rubric).
DIM_ORDER = [
    "AS", "AT", "CL", "GS", "KNf", "MA", "MCr", "MCu", "QLl", "QLq", "SNs", "VL", "VO",
]

ITEM_COLUMNS = [
    "item_id", "source", "paperid", "claim_type", "vtype", "image_path", "claim", "label",
]


def read_table(db_path: Path, table: str) -> pd.DataFrame:
    """Read a whole SQLite table into a DataFrame."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def dimension_names(annotation: pd.DataFrame) -> pd.DataFrame:
    """Map each dim_code to its human-readable name and its item coverage."""
    names = (
        annotation.drop_duplicates("dim_code")
        .set_index("dim_code")["dim_name"]
    )
    coverage = annotation.groupby("dim_code")["item_id"].nunique()
    out = pd.DataFrame({"dim_name": names, "items_scored": coverage})
    out = out.reindex(DIM_ORDER)
    out.index.name = "dim_code"
    return out.reset_index()


def build_annotation_wide(annotation: pd.DataFrame) -> pd.DataFrame:
    """Pivot annotations to one row per item with one column per dimension score.

    A few items were scored twice for some dimensions (a small test pass). We keep
    a single score per (item, dimension) by preferring the latest pass, then the
    latest row id. Dimensions never scored for an item stay blank (NaN).
    """
    deduped = (
        annotation.sort_values(["item_id", "dim_code", "pass", "id"])
        .drop_duplicates(["item_id", "dim_code"], keep="last")
    )
    wide = deduped.pivot(index="item_id", columns="dim_code", values="score")
    # Stable, complete column order (missing dimensions become all-NaN columns).
    wide = wide.reindex(columns=DIM_ORDER)
    wide.columns = [f"dim_{code}" for code in wide.columns]
    return wide.reset_index()


def build_prediction_summary(prediction: pd.DataFrame, target_run: int) -> pd.DataFrame:
    """Summarize the repeated LLM predictions for one run, one row per item."""
    run_df = prediction[prediction["run"] == target_run]
    if run_df.empty:
        raise ValueError(f"No predictions found for run={target_run!r}.")

    grouped = run_df.groupby("item_id")
    summary = pd.DataFrame({
        "pred_n_trials": grouped.size(),
        "pred_n_correct": grouped["correct"].sum(),
        "pred_n_entailed": grouped["predicted"].sum(),
    })
    summary["pred_frac_correct"] = summary["pred_n_correct"] / summary["pred_n_trials"]
    summary["pred_frac_entailed"] = summary["pred_n_entailed"] / summary["pred_n_trials"]
    # Majority votes (ties round up to the positive class).
    summary["pred_majority_correct"] = (summary["pred_frac_correct"] >= 0.5).astype(int)
    summary["pred_majority_label"] = (summary["pred_frac_entailed"] >= 0.5).astype(int)
    summary["pred_unanimous"] = (
        (summary["pred_n_correct"] == 0) | (summary["pred_n_correct"] == summary["pred_n_trials"])
    ).astype(int)
    return summary.reset_index()


def build_combined(
    item: pd.DataFrame,
    annotation_wide: pd.DataFrame,
    prediction_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge item metadata, dimension scores, and prediction summary on item_id."""
    combined = item[ITEM_COLUMNS].merge(annotation_wide, on="item_id", how="left")
    combined = combined.merge(prediction_summary, on="item_id", how="left")
    return combined.sort_values("item_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-db", type=Path, default=Path("data/annotations_prod.db"))
    parser.add_argument("--predictions-db", type=Path, default=Path("data/predictions.db"))
    parser.add_argument("--target-run", type=int, default=1, help="Prediction run that defines the LLM target.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--basename", default="faithfulness_dataset")
    args = parser.parse_args()

    item = read_table(args.annotations_db, "item")
    annotation = read_table(args.annotations_db, "annotation")
    prediction = read_table(args.predictions_db, "prediction")

    annotation_wide = build_annotation_wide(annotation)
    prediction_summary = build_prediction_summary(prediction, args.target_run)
    combined = build_combined(item, annotation_wide, prediction_summary)
    dims = dimension_names(annotation)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / f"{args.basename}.parquet"
    csv_path = args.out_dir / f"{args.basename}.csv"
    dims_path = args.out_dir / f"{args.basename}_dimensions.csv"
    combined.to_parquet(parquet_path, index=False)
    combined.to_csv(csv_path, index=False)
    dims.to_csv(dims_path, index=False)

    print(f"Combined dataset: {combined.shape[0]} rows x {combined.shape[1]} cols")
    print(f"  parquet -> {parquet_path}")
    print(f"  csv     -> {csv_path}")
    print(f"  dims    -> {dims_path}")
    missing = combined[[c for c in combined.columns if c.startswith("dim_")]].isna().sum()
    print("\nMissing dimension scores per column:")
    print(missing[missing > 0].to_string() if (missing > 0).any() else "  none")


if __name__ == "__main__":
    main()
