"""Run the future directions from docs/CHECKPOINT_0605_AND_ROADMAP_FOR_EDA.md.

Executes the roadmap items that the faithfulness probe and post-hoc audits did
not already cover, on the rederived feature store:

Baseline-improvement roadmap:
  1. Expanded supervised population — all chart+table pairs, all claim types
     (train = val 504, test = 996), per-block accuracies with permutation
     tests; metadata one-hots (claim_type, modality) alone and added to the
     full stack.
  2. Regularization grid — C in {0.01, 0.1, 1, 10} chosen by 5-fold CV on the
     training split only, then scored once on test.
  3. Calibration — Brier score, 10-bin expected calibration error, and the
     spread of predicted probabilities for the strongest feature block.
  4. Error analysis — test accuracy broken down by modality and claim type.

Embedding-EDA roadmap:
  5. Similarity-distribution analysis — mean cosine similarity for same-label
     vs different-label and same-paper vs different-paper pairs, per vector
     type; 1-nearest-neighbor label/paper agreement.
  6. Near-duplicate claims — pairs of distinct examples whose claim embeddings
     exceed cosine 0.9/0.95, their label-disagreement rate, and how many cross
     the val/test boundary.
  7. Per-paper label balance — whether labels cluster within papers (which
     would let paper-identity proxies masquerade as signal).

Usage:
    python scripts/roadmap_followup.py

Writes docs/roadmap_followup_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.faithfulness_probe import assemble_blocks, lr_pipeline

SEED = 0
N_PERM = 200


def stack(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=np.float32) for v in df[col]])


def build_blocks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    blocks = assemble_blocks(
        stack(df, "claim_vec"),
        stack(df, "evidence_text_vec"),
        stack(df, "pair_text_vec"),
        stack(df, "image_vec"),
    )
    onehots = pd.get_dummies(df[["claim_type", "modality"]], dtype=np.float32).to_numpy()
    blocks["metadata_onehots"] = onehots
    blocks["full+onehots"] = np.hstack([blocks["full_notebook_stack"], onehots])
    return blocks


def expanded_baseline(blocks: dict, y: np.ndarray, split: np.ndarray, rng: np.random.Generator) -> dict:
    tr, te = split == "val", split == "test"
    out = {}
    for name in ("claim_vec", "image_vec", "claim+image", "full_notebook_stack", "metadata_onehots", "full+onehots"):
        X = blocks[name]
        pipe = lr_pipeline().fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        acc = accuracy_score(y[te], pred)
        null = np.array(
            [
                accuracy_score(y[te], lr_pipeline().fit(X[tr], rng.permutation(y[tr])).predict(X[te]))
                for _ in range(N_PERM)
            ]
        )
        out[name] = {
            "n_features": X.shape[1],
            "accuracy": round(acc, 3),
            "balanced_accuracy": round(balanced_accuracy_score(y[te], pred), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "perm_p": round(float((np.sum(null >= acc) + 1) / (len(null) + 1)), 3),
        }
    return out


def regularization_grid(blocks: dict, y: np.ndarray, split: np.ndarray) -> dict:
    tr, te = split == "val", split == "test"
    out = {}
    for name in ("claim+image", "full+onehots"):
        X = blocks[name]
        grid = GridSearchCV(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring="balanced_accuracy",
        ).fit(X[tr], y[tr])
        pred = grid.predict(X[te])
        out[name] = {
            "best_C": grid.best_params_["clf__C"],
            "train_cv_balanced_accuracy": round(float(grid.best_score_), 3),
            "test_accuracy": round(accuracy_score(y[te], pred), 3),
            "test_balanced_accuracy": round(balanced_accuracy_score(y[te], pred), 3),
        }
    return out


def calibration(blocks: dict, y: np.ndarray, split: np.ndarray, block: str) -> dict:
    tr, te = split == "val", split == "test"
    X = blocks[block]
    pipe = lr_pipeline().fit(X[tr], y[tr])
    prob = pipe.predict_proba(X[te])[:, 1]
    bins = np.clip((prob * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[te][mask].mean() - prob[mask].mean())
    return {
        "block": block,
        "brier_score": round(float(brier_score_loss(y[te], prob)), 4),
        "brier_score_always_0.5": 0.25,
        "ece_10bin": round(float(ece), 4),
        "predicted_prob_mean": round(float(prob.mean()), 3),
        "predicted_prob_std": round(float(prob.std()), 3),
        "frac_predictions_in_0.4_0.6": round(float(((prob > 0.4) & (prob < 0.6)).mean()), 3),
    }


def error_breakdown(blocks: dict, df: pd.DataFrame, y: np.ndarray, split: np.ndarray, block: str) -> dict:
    tr, te = split == "val", split == "test"
    pred = lr_pipeline().fit(blocks[block][tr], y[tr]).predict(blocks[block][te])
    sub = df[te].copy()
    sub["correct"] = (pred == y[te]).astype(int)
    out = {"block": block}
    for col in ("modality", "claim_type"):
        out[f"by_{col}"] = {
            str(k): {"n": int(g["correct"].size), "accuracy": round(float(g["correct"].mean()), 3)}
            for k, g in sub.groupby(col)
        }
    return out


def similarity_eda(df: pd.DataFrame, blocks: dict) -> dict:
    """Same-label/same-paper cosine structure and 1-NN agreement per vector type."""
    labels = df["label_id"].astype(int).to_numpy()
    papers = df["paperid"].astype(str).to_numpy()
    out = {}
    for name in ("claim_vec", "image_vec"):
        V = blocks[name].astype(np.float32)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        sims = V @ V.T
        np.fill_diagonal(sims, -2.0)  # exclude self everywhere below

        same_label = labels[:, None] == labels[None, :]
        same_paper = papers[:, None] == papers[None, :]
        iu = np.triu_indices(len(df), k=1)
        out[name] = {
            "mean_cosine_same_label": round(float(sims[iu][same_label[iu]].mean()), 4),
            "mean_cosine_diff_label": round(float(sims[iu][~same_label[iu]].mean()), 4),
            "mean_cosine_same_paper": round(float(sims[iu][same_paper[iu]].mean()), 4),
            "mean_cosine_diff_paper": round(float(sims[iu][~same_paper[iu]].mean()), 4),
            "nn1_label_agreement": round(float((labels[sims.argmax(axis=1)] == labels).mean()), 3),
            "nn1_same_paper_rate": round(float((papers[sims.argmax(axis=1)] == papers).mean()), 3),
        }
    return out


def near_duplicate_claims(df: pd.DataFrame, blocks: dict) -> dict:
    labels = df["label_id"].astype(int).to_numpy()
    splits = df["split"].astype(str).to_numpy()
    V = blocks["claim_vec"].astype(np.float32)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    sims = V @ V.T
    np.fill_diagonal(sims, -2.0)
    iu = np.triu_indices(len(df), k=1)
    out = {}
    for thresh in (0.90, 0.95):
        mask = sims[iu] >= thresh
        i, j = iu[0][mask], iu[1][mask]
        out[f"cosine>={thresh}"] = {
            "n_pairs": int(mask.sum()),
            "opposite_label_fraction": round(float((labels[i] != labels[j]).mean()), 3) if mask.any() else None,
            "cross_split_pairs": int((splits[i] != splits[j]).sum()),
        }
    return out


def paper_label_balance(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Do labels cluster within papers more than chance would predict?"""
    labels = df["label_id"].astype(int).to_numpy()
    papers = df["paperid"].astype(str).to_numpy()
    by_paper = defaultdict(list)
    for p, l in zip(papers, labels):
        by_paper[p].append(l)
    multi = {p: v for p, v in by_paper.items() if len(v) >= 2}

    def imbalance(groups: dict) -> float:
        return float(np.mean([abs(np.mean(v) - 0.5) for v in groups.values()]))

    observed = imbalance(multi)
    null = []
    for _ in range(500):
        shuffled = rng.permutation(labels)
        groups = defaultdict(list)
        for p, l in zip(papers, shuffled):
            groups[p].append(l)
        null.append(imbalance({p: v for p, v in groups.items() if len(v) >= 2}))
    null = np.array(null)
    return {
        "papers_with_2plus_pairs": len(multi),
        "mean_within_paper_label_imbalance": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "perm_p": round(float((np.sum(null >= observed) + 1) / (len(null) + 1)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-parquet", default="data/processed/sciver_pair_features.parquet", type=Path)
    parser.add_argument("--out", default="docs/roadmap_followup_results.json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(args.features_parquet)
    for col in ("claim_vec", "evidence_text_vec", "pair_text_vec", "image_vec"):
        df = df[df[col].notna()]
    df = df.reset_index(drop=True)
    y = df["label_id"].astype(int).to_numpy()
    split = df["split"].astype(str).to_numpy()
    blocks = build_blocks(df)

    results = {
        "population": {
            "n_pairs": len(df),
            "train_val": int((split == "val").sum()),
            "test": int((split == "test").sum()),
            "claim_types": {str(k): int(v) for k, v in df["claim_type"].value_counts().items()},
            "modalities": {str(k): int(v) for k, v in df["modality"].value_counts().items()},
        },
        "expanded_baseline_all_pairs": expanded_baseline(blocks, y, split, rng),
        "regularization_grid": regularization_grid(blocks, y, split),
        "calibration": calibration(blocks, y, split, "full+onehots"),
        "error_breakdown": error_breakdown(blocks, df, y, split, "full+onehots"),
        "similarity_eda": similarity_eda(df, blocks),
        "near_duplicate_claims": near_duplicate_claims(df, blocks),
        "paper_label_balance": paper_label_balance(df, rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
