"""Faithfulness probe for the SciVer chart/table baseline.

Answers one question: does the lightweight baseline carry real, leakage-free
signal for entailed-vs-refuted classification, or is the reported ~0.55 accuracy
consistent with random chance? Produces the numbers reported in
`docs/FAITHFULNESS_PROBE_FINDINGS.md`.

It is read-only with respect to the pipeline: it reuses the project's own parser
(`sciver_vector_db.parsing`) and, when no exported features exist, its embedders
(`sciver_vector_db.embeddings`), so the features are exactly what the Qdrant build
produces. It then runs:

  1. Claim-text-only TF-IDF probe + label-permutation test
     (tests whether the claim string alone leaks the answer -> a perturbation
     artifact, since the claim cannot honestly reveal entailment).
  2. Embedding ablation on the original val->test split, replicating the
     notebook's StandardScaler + balanced LogisticRegression, with a
     label-permutation test and a bootstrap CI per feature block.
  3. Repeated 5-fold x 10 cross-validation + CV permutation test, which exposes
     the split-selection variance a single split hides.

Usage:
    python scripts/faithfulness_probe.py --data-root data/raw/SciVer

By default it reads the canonical exported features
(`data/processed/sciver_pair_features.parquet`) if present, so it does NOT
re-embed; otherwise it builds them once via the project's disk-cached embedders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sciver_vector_db.embeddings import ImageEmbedder, TextEmbedder
from sciver_vector_db.parsing import build_pair_records, load_split_examples

SEED = 0
N_PERM_SINGLE = 300        # label permutations for the single-split tests
N_PERM_CV = 200            # label permutations for the CV permutation test
N_BOOT = 2000              # bootstrap resamples for the test-accuracy CI


def lr_pipeline() -> Pipeline:
    """The notebook's exact classifier: standardized features + balanced LR."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def tfidf_text_probe(recs, rng: np.random.Generator) -> dict:
    """Claim-text-only TF-IDF classifier with a label-permutation null."""
    tr = [r for r in recs if r.split == "val"]
    te = [r for r in recs if r.split == "test"]
    Xtr, ytr = [r.claim for r in tr], np.array([r.label_id for r in tr])
    Xte, yte = [r.claim for r in te], np.array([r.label_id for r in te])

    def fit_score(labels: np.ndarray) -> float:
        pipe = Pipeline(
            [
                ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipe.fit(Xtr, labels)
        return accuracy_score(yte, pipe.predict(Xte))

    acc = fit_score(ytr)
    null = np.array([fit_score(rng.permutation(ytr)) for _ in range(N_PERM_SINGLE)])
    majority = max(np.bincount(yte)) / len(yte)
    return {
        "n_train": len(tr),
        "n_test": len(te),
        "accuracy": round(acc, 3),
        "majority_baseline": round(majority, 3),
        "null_mean": round(float(null.mean()), 3),
        "null_std": round(float(null.std()), 3),
        "null_p95": round(float(np.percentile(null, 95)), 3),
        "null_max": round(float(null.max()), 3),
        "perm_p": round(float((np.sum(null >= acc) + 1) / (len(null) + 1)), 3),
    }


def single_split_ablation(blocks: dict, y: np.ndarray, split: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-feature-set evaluation on the original val->test split."""
    tr, te = split == "val", split == "test"
    out = {}
    for name, X in blocks.items():
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        pipe = lr_pipeline().fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)

        null = []
        for _ in range(N_PERM_SINGLE):
            null.append(accuracy_score(yte, lr_pipeline().fit(Xtr, rng.permutation(ytr)).predict(Xte)))
        null = np.array(null)

        correct = (pred == yte).astype(int)
        boots = [correct[rng.integers(0, len(yte), len(yte))].mean() for _ in range(N_BOOT)]
        out[name] = {
            "n_features": X.shape[1],
            "accuracy": round(acc, 3),
            "balanced_accuracy": round(balanced_accuracy_score(yte, pred), 3),
            "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "null_max": round(float(null.max()), 3),
            "perm_p": round(float((np.sum(null >= acc) + 1) / (len(null) + 1)), 3),
        }
    return out


def repeated_cv(blocks: dict, y: np.ndarray) -> dict:
    """Repeated 5-fold x 10 CV with a CV-level permutation test."""
    out = {}
    for name, X in blocks.items():
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
        scores = cross_val_score(lr_pipeline(), X, y, cv=cv, scoring="balanced_accuracy")
        _, _, p = permutation_test_score(
            lr_pipeline(),
            X,
            y,
            scoring="balanced_accuracy",
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1),
            n_permutations=N_PERM_CV,
            random_state=1,
        )
        out[name] = {
            "cv_balanced_accuracy_mean": round(float(scores.mean()), 3),
            "cv_balanced_accuracy_std": round(float(scores.std()), 3),
            "perm_p": round(float(p), 3),
        }
    return out


def assemble_blocks(claim_vec, evidence_vec, pair_vec, image_vec) -> dict:
    """Build the notebook's feature stack (cell 8 / export_pair_features) from vectors."""
    absdiff = np.abs(claim_vec - evidence_vec)
    product = claim_vec * evidence_vec
    full = np.hstack([claim_vec, evidence_vec, pair_vec, image_vec, absdiff, product])
    return {
        "claim_vec": claim_vec,
        "evidence_text_vec": evidence_vec,
        "pair_text_vec": pair_vec,
        "image_vec": image_vec,
        "claim+image": np.hstack([claim_vec, image_vec]),
        "full_notebook_stack": full,
    }


def features_from_parquet(parquet_path: Path):
    """Load the canonical embeddings already produced by the pipeline (preferred path).

    Reuses the single exported feature table (`scripts/export_pair_features.py`)
    rather than re-embedding, mirroring notebook 02's fast path.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    df = df[(df["modality"] == "chart") & (df["claim_type"] == "direct")].reset_index(drop=True)
    for col in ("claim_vec", "evidence_text_vec", "pair_text_vec", "image_vec"):
        df = df[df[col].notna()]
    stack = lambda col: np.vstack([np.asarray(v, dtype=np.float32) for v in df[col]])
    blocks = assemble_blocks(
        stack("claim_vec"), stack("evidence_text_vec"), stack("pair_text_vec"), stack("image_vec")
    )
    return blocks, df["label_id"].astype(int).to_numpy(), df["split"].astype(str).to_numpy()


def features_from_embedders(recs):
    """Fallback: build features with the project embedders (disk-cached; computed once)."""
    text = TextEmbedder()
    claims = [r.claim for r in recs]
    evidence = [r.evidence_text() for r in recs]   # empty for every SciVer pair
    pairs = [r.pair_text() for r in recs]
    tmap = text.encode(claims + evidence + pairs)
    image = ImageEmbedder()
    imap, failures = image.encode([r.image_path for r in recs])
    if failures:
        raise RuntimeError(f"{len(failures)} image embeddings failed; cannot run a clean probe")
    blocks = assemble_blocks(
        np.vstack([tmap[t] for t in claims]),
        np.vstack([tmap[t] for t in evidence]),
        np.vstack([tmap[t] for t in pairs]),
        np.vstack([imap[Path(r.image_path)] for r in recs]),
    )
    return blocks, np.array([r.label_id for r in recs]), np.array([r.split for r in recs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/SciVer", type=Path)
    parser.add_argument(
        "--features-parquet",
        default="data/processed/sciver_pair_features.parquet",
        type=Path,
        help="Canonical exported features; used if present so nothing is re-embedded.",
    )
    parser.add_argument("--out", default="docs/faithfulness_probe_results.json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    examples, _ = load_split_examples(args.data_root)
    recs, _ = build_pair_records(examples, modalities={"chart", "table"})
    direct_chart = [r for r in recs if r.modality == "chart" and r.claim_type == "direct"]

    # Confirm the empty-evidence finding directly.
    empty_evidence = sum(1 for r in recs if not r.evidence_text().strip())

    # Prefer the single canonical embedding the pipeline already produced.
    if args.features_parquet.exists():
        feats, y, split = features_from_parquet(args.features_parquet)
        feature_source = str(args.features_parquet)
    else:
        feats, y, split = features_from_embedders(direct_chart)
        feature_source = "project embedders (disk-cached)"

    results = {
        "feature_source": feature_source,
        "dataset": {
            "pairs_total": len(recs),
            "pairs_by_split": {s: int((np.array([r.split for r in recs]) == s).sum()) for s in ("val", "test")},
            "direct_chart_pairs": len(direct_chart),
            "direct_chart_by_split": {s: int((split == s).sum()) for s in ("val", "test")},
            "pairs_with_empty_evidence_text": empty_evidence,
        },
        "tfidf_text_probe": {
            "direct_chart": tfidf_text_probe(direct_chart, rng),
            "all_chart_table": tfidf_text_probe(recs, rng),
        },
        "single_split_ablation": single_split_ablation(feats, y, split, rng),
        "repeated_cv": repeated_cv(
            {k: feats[k] for k in ("claim_vec", "image_vec", "claim+image")}, y
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
