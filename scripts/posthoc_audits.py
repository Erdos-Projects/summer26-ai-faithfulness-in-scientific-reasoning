"""Post-hoc audits for the faithfulness probe findings.

Closes the verification gaps identified in review of
`docs/FAITHFULNESS_PROBE_FINDINGS.md`:

  1. Duplication audit — how often the same image appears in multiple pairs,
     with which label patterns, and whether images/papers cross the val/test
     boundary (distinguishes "lucky split" from "contaminated split").
  2. Image-only ceiling — the maximum accuracy any deterministic
     image-feature-only classifier could reach given images that carry both
     labels.
  3. Grouped cross-validation — StratifiedGroupKFold by paperid, so no paper
     (and therefore no image) spans a train/test fold boundary, compared
     against plain StratifiedKFold run the same way.
  4. Family-wise (max-statistic) permutation test — was the best single-split
     accuracy (claim+image, 0.576) surprising given that six feature blocks
     were tried? Each permutation refits all six blocks on the same shuffled
     training labels and records the best test accuracy.
  5. Caption/context recoverability — what fraction of pairs have a real
     caption and related-sentence context recoverable from the `paper_path`
     JSONs that the current parser never reads.
  6. Analytic power — minimum detectable accuracy for the sample sizes used.

Usage:
    python scripts/posthoc_audits.py

Reads the exported features (`data/processed/sciver_pair_features.parquet`)
and the raw SciVer snapshot (`data/raw/SciVer`); writes
`docs/posthoc_audit_results.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from scripts.faithfulness_probe import assemble_blocks, lr_pipeline

SEED = 0
N_PERM_FAMILY = 300
FIGURE_RE = re.compile(r"_figure_(\d+)\.")
TABLE_RE = re.compile(r"-Table(\d+)-")


def load_features(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    for col in ("claim_vec", "evidence_text_vec", "pair_text_vec", "image_vec"):
        df = df[df[col].notna()]
    return df.reset_index(drop=True)


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=np.float32) for v in df[col]])


def duplication_audit(df: pd.DataFrame) -> dict:
    """Image reuse across pairs, label co-occurrence, and split overlap."""

    def audit(sub: pd.DataFrame) -> dict:
        by_img = defaultdict(list)
        for img, label, split in zip(sub["image_path"], sub["label_id"], sub["split"]):
            by_img[img].append((split, int(label)))
        multi = {k: v for k, v in by_img.items() if len(v) > 1}
        both_labels = [k for k, v in multi.items() if len({l for _, l in v}) > 1]
        cross_split = [k for k, v in multi.items() if len({s for s, _ in v}) > 1]
        papers = {
            s: set(sub.loc[sub["split"] == s, "paperid"]) for s in ("val", "test")
        }
        return {
            "n_pairs": len(sub),
            "n_unique_images": len(by_img),
            "images_in_multiple_pairs": len(multi),
            "images_with_both_labels": len(both_labels),
            "images_spanning_val_and_test": len(cross_split),
            "papers_in_both_splits": len(papers["val"] & papers["test"]),
        }

    direct_chart = df[(df["modality"] == "chart") & (df["claim_type"] == "direct")]
    return {"all_pairs": audit(df), "direct_chart": audit(direct_chart)}


def image_only_ceiling(df: pd.DataFrame) -> dict:
    """Best accuracy a deterministic image-only classifier could achieve.

    The same image must get the same prediction, so for an image with both an
    entailed and a refuted pair, at most the majority of its pairs can be
    correct.
    """

    def ceiling(sub: pd.DataFrame) -> float:
        by_img = defaultdict(list)
        for img, label in zip(sub["image_path"], sub["label_id"]):
            by_img[img].append(int(label))
        best = sum(max(labels.count(0), labels.count(1)) for labels in by_img.values())
        return round(best / len(sub), 4)

    direct_chart = df[(df["modality"] == "chart") & (df["claim_type"] == "direct")]
    return {
        "direct_chart_test": ceiling(direct_chart[direct_chart["split"] == "test"]),
        "direct_chart_all": ceiling(direct_chart),
        "all_pairs": ceiling(df),
    }


def grouped_vs_plain_cv(blocks: dict, y: np.ndarray, groups: np.ndarray) -> dict:
    """StratifiedGroupKFold by paper vs plain StratifiedKFold, 5 folds x 10 seeds."""
    out = {}
    for name, X in blocks.items():
        scores = {"grouped_by_paper": [], "plain": []}
        for seed in range(10):
            for kind, cv in (
                ("grouped_by_paper", StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)),
                ("plain", StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)),
            ):
                split_groups = groups if kind == "grouped_by_paper" else None
                for tr, te in cv.split(X, y, groups=split_groups):
                    pred = lr_pipeline().fit(X[tr], y[tr]).predict(X[te])
                    scores[kind].append(balanced_accuracy_score(y[te], pred))
        out[name] = {
            kind: {
                "cv_balanced_accuracy_mean": round(float(np.mean(vals)), 3),
                "cv_balanced_accuracy_std": round(float(np.std(vals)), 3),
            }
            for kind, vals in scores.items()
        }
    return out


def family_wise_permutation(blocks: dict, y: np.ndarray, split: np.ndarray, rng: np.random.Generator) -> dict:
    """Max-statistic permutation test across all feature blocks at once.

    Answers: given that six feature blocks were evaluated on the same split,
    how surprising is the best observed accuracy? Each permutation applies one
    shuffled training-label vector to every block and keeps the maximum test
    accuracy, building the null of "best of six zero-skill models".
    """
    tr, te = split == "val", split == "test"
    observed = {}
    for name, X in blocks.items():
        pred = lr_pipeline().fit(X[tr], y[tr]).predict(X[te])
        observed[name] = accuracy_score(y[te], pred)
    best_name = max(observed, key=observed.get)
    best_acc = observed[best_name]

    null_max = []
    for _ in range(N_PERM_FAMILY):
        perm = rng.permutation(y[tr])
        null_max.append(
            max(
                accuracy_score(y[te], lr_pipeline().fit(X[tr], perm).predict(X[te]))
                for X in blocks.values()
            )
        )
    null_max = np.array(null_max)
    return {
        "observed_best_block": best_name,
        "observed_best_accuracy": round(best_acc, 3),
        "observed_per_block": {k: round(v, 3) for k, v in observed.items()},
        "null_max_mean": round(float(null_max.mean()), 3),
        "null_max_p95": round(float(np.percentile(null_max, 95)), 3),
        "null_max_max": round(float(null_max.max()), 3),
        "family_wise_perm_p": round(float((np.sum(null_max >= best_acc) + 1) / (len(null_max) + 1)), 3),
    }


def caption_recoverability(df: pd.DataFrame, data_root: Path) -> dict:
    """How many pairs have a caption/context recoverable from paper_path JSONs."""
    papers_dir = data_root / "papers"
    counts = {
        "pairs_checked": 0,
        "paper_json_found": 0,
        "caption_found": 0,
        "caption_nonempty": 0,
        "related_context_found": 0,
    }
    caption_lengths = []
    paper_cache: dict[str, dict | None] = {}
    for paperid, image_path in zip(df["paperid"], df["image_path"]):
        counts["pairs_checked"] += 1
        if paperid not in paper_cache:
            path = papers_dir / f"{paperid}.json"
            try:
                paper_cache[paperid] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                paper_cache[paperid] = None
        paper = paper_cache[paperid]
        if paper is None:
            continue
        counts["paper_json_found"] += 1

        name = Path(str(image_path)).name
        caption = None
        context_key = None
        if m := FIGURE_RE.search(name):
            entry = (paper.get("image_paths") or {}).get(m.group(1))
            caption = (entry or {}).get("caption")
            context_key = ("image_result", m.group(1))
        elif m := TABLE_RE.search(name):
            entry = (paper.get("tables") or {}).get(m.group(1))
            caption = (entry or {}).get("capture")
            context_key = ("table_result", m.group(1))
        if caption is not None:
            counts["caption_found"] += 1
            if str(caption).strip():
                counts["caption_nonempty"] += 1
                caption_lengths.append(len(str(caption)))
        if context_key is not None:
            section_map = (paper.get(context_key[0]) or {}).get(context_key[1]) or {}
            if any((v or {}).get("related_sentences_ids") for v in section_map.values()):
                counts["related_context_found"] += 1
    counts["mean_caption_chars"] = round(float(np.mean(caption_lengths)), 1) if caption_lengths else 0.0
    return counts


def analytic_power() -> dict:
    """Minimum detectable accuracy vs 0.5 at 80% power (binomial, normal approx)."""
    z_power = norm.ppf(0.80)

    def mde(n: int, alpha: float, sided: int) -> float:
        z_alpha = norm.ppf(1 - alpha / sided)
        return round(0.5 + (z_alpha + z_power) * float(np.sqrt(0.25 / n)), 3)

    return {
        f"n_{n}": {
            "one_sided_alpha_05": mde(n, 0.05, 1),
            "two_sided_alpha_05": mde(n, 0.05, 2),
        }
        for n in (83, 140, 276, 416, 996)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/SciVer", type=Path)
    parser.add_argument("--features-parquet", default="data/processed/sciver_pair_features.parquet", type=Path)
    parser.add_argument("--out", default="docs/posthoc_audit_results.json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    df = load_features(args.features_parquet)
    direct_chart = df[(df["modality"] == "chart") & (df["claim_type"] == "direct")].reset_index(drop=True)
    y = direct_chart["label_id"].astype(int).to_numpy()
    split = direct_chart["split"].astype(str).to_numpy()
    groups = direct_chart["paperid"].astype(str).to_numpy()
    blocks = assemble_blocks(
        stack(direct_chart, "claim_vec"),
        stack(direct_chart, "evidence_text_vec"),
        stack(direct_chart, "pair_text_vec"),
        stack(direct_chart, "image_vec"),
    )
    cv_blocks = {k: blocks[k] for k in ("claim_vec", "image_vec", "claim+image")}

    results = {
        "duplication_audit": duplication_audit(df),
        "image_only_ceiling": image_only_ceiling(df),
        "grouped_vs_plain_cv": grouped_vs_plain_cv(cv_blocks, y, groups),
        "family_wise_permutation": family_wise_permutation(blocks, y, split, rng),
        "caption_recoverability": caption_recoverability(df, args.data_root),
        "analytic_power_min_detectable_accuracy": analytic_power(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
