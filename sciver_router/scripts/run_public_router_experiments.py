"""Run the public three-VLM SciVer router experiment grid.

The script trains from the sanitized public dataset only. TDA and NLP feature
columns are expected to already be embedded in that CSV, so no private sidecar
paths are required for reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_router_models import compare_models
from router_model_lib import (
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_TDA_NLP,
    SELECTION_POLICY_REGRET_FIRST,
    SELECTION_POLICY_TOP1_FIRST,
)
import train_model_router_ridge
import train_model_router_torch_mlp
import train_model_router_xgboost


DEFAULT_INPUT = "Data/derived/model_router/public_router/router_dataset_public.csv"
DEFAULT_OUT_ROOT = "Data/derived/model_router/public_router"
PUBLIC_MODELS = ["qwen3vl", "pixtral12b", "kimi_vl_a3b"]
FEATURE_SETS = [
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA_NLP,
]
FEATURE_LABELS = {
    FEATURE_SET_BASE: "base",
    FEATURE_SET_BASE_TDA: "chart_tda",
    FEATURE_SET_BASE_NLP: "claim_nlp",
    FEATURE_SET_BASE_TDA_NLP: "chart_tda_claim_nlp",
}
POLICIES = [SELECTION_POLICY_REGRET_FIRST, SELECTION_POLICY_TOP1_FIRST]
FAMILIES = ["ridge_elasticnet", "xgboost", "torch_mlp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=215)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--models", nargs="+", default=PUBLIC_MODELS)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    parser.add_argument("--selection-policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--torch-device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lbfgs-max-iter", type=int, default=300)
    parser.add_argument("--xgboost-candidate-limit", type=int, default=None)
    parser.add_argument("--xgboost-n-jobs", type=int, default=4)
    parser.add_argument("--torch-candidate-limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a run directory when metrics.json already exists.",
    )
    return parser.parse_args()


def run_dir(out_root: Path, policy: str, family: str, feature_set: str) -> Path:
    return out_root / "runs" / policy / f"{family}_{FEATURE_LABELS[feature_set]}"


def run_training(args: argparse.Namespace, family: str, feature_set: str, policy: str) -> Path:
    input_path = Path(args.input)
    out_dir = run_dir(Path(args.out_root), policy, family, feature_set)
    if args.skip_existing and (out_dir / "metrics.json").exists():
        print(f"[public-router] reuse existing {family} {feature_set} {policy} -> {out_dir}")
        return out_dir
    common = {
        "input_path": input_path,
        "out_dir": out_dir,
        "repo_root": Path(args.repo_root),
        "seed": args.seed,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "cv_folds": args.cv_folds,
        "bootstrap_repeats": args.bootstrap_repeats,
        "feature_set": feature_set,
        "tda_features_path": None,
        "nlp_features_path": None,
        "model_names": args.models,
        "selection_policy": policy,
    }
    print(f"[public-router] training {family} {feature_set} {policy} -> {out_dir}")
    if family == "ridge_elasticnet":
        train_model_router_ridge.train_router(**common)
    elif family == "xgboost":
        train_model_router_xgboost.train_router(
            **common,
            candidate_limit=args.xgboost_candidate_limit,
            n_jobs=args.xgboost_n_jobs,
        )
    elif family == "torch_mlp":
        train_model_router_torch_mlp.train_router(
            **common,
            candidate_limit=args.torch_candidate_limit,
            requested_device=args.torch_device,
            requested_dtype=args.torch_dtype,
            lbfgs_max_iter=args.torch_lbfgs_max_iter,
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return out_dir


def write_comparison(out_root: Path, policy: str, run_dirs: list[Path], labels: list[str]) -> None:
    comparison_dir = out_root / "comparisons" / policy
    summary = compare_models(run_dirs, comparison_dir, labels)
    print(
        "[public-router] comparison",
        policy,
        json.dumps(summary["best_by_mean_regret"], sort_keys=True),
    )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_policy_summaries: dict[str, list[dict[str, str]]] = {}
    for policy in args.selection_policies:
        policy_run_dirs: list[Path] = []
        policy_labels: list[str] = []
        for feature_set in args.feature_sets:
            for family in args.families:
                directory = run_training(args, family, feature_set, policy)
                policy_run_dirs.append(directory)
                policy_labels.append(f"{family}_{FEATURE_LABELS[feature_set]}_{policy}")
        write_comparison(out_root, policy, policy_run_dirs, policy_labels)
        all_policy_summaries[policy] = [
            {"label": label, "model_dir": directory.as_posix()}
            for label, directory in zip(policy_labels, policy_run_dirs)
        ]

    manifest = {
        "input": args.input,
        "out_root": args.out_root,
        "models": args.models,
        "families": args.families,
        "feature_sets": args.feature_sets,
        "selection_policies": args.selection_policies,
        "runs": all_policy_summaries,
    }
    (out_root / "public_router_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
