#!/usr/bin/env python3
"""Train a PyTorch MLP router over matched SciVer model-result data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from router_model_lib import (
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_TDA_NLP,
    FEATURE_SET_CHOICES,
    MODEL_NAMES,
    SELECTION_POLICY_CHOICES,
    SELECTION_POLICY_REGRET_FIRST,
    candidate_metadata,
    candidate_sort_key,
    prepare_router_data,
    write_bootstrap_outputs,
    write_common_training_outputs,
    write_json,
    write_overfitting_curve_plot,
)
from torch_mlp_router_lib import (
    DEVICE_CHOICES,
    DTYPE_CHOICES,
    candidate_grid,
    candidate_id,
    encoded_feature_count,
    evaluate_candidate_cv_torch,
    evaluate_overfitting_curve_torch,
    fit_predict_torch,
    fit_torch_model,
    resolve_runtime,
    save_torch_bundle,
)


DEFAULT_INPUT = "Data/derived/model_router/public_router/router_dataset_public.csv"
DEFAULT_OUT_DIR = "Data/derived/model_router/public_router/runs/regret_first/torch_mlp_base"
DEFAULT_TDA_FEATURES = ""
DEFAULT_NLP_FEATURES = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=215)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default=FEATURE_SET_BASE)
    parser.add_argument("--tda-features", default=DEFAULT_TDA_FEATURES)
    parser.add_argument("--nlp-features", default=DEFAULT_NLP_FEATURES)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--selection-policy", choices=SELECTION_POLICY_CHOICES, default=SELECTION_POLICY_REGRET_FIRST)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="float32")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Optional development/test limit. Omit for the full specified MLP grid.",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=300,
        help="Maximum PyTorch LBFGS iterations per fit. The architecture/activation/alpha grid matches sklearn MLP.",
    )
    parser.add_argument("--lbfgs-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def update_run_config(out_dir: Path, updates: dict[str, Any]) -> None:
    config_path = out_dir / "router_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config.update(updates)
    write_json(config_path, config)


def train_router(
    input_path: Path,
    out_dir: Path,
    repo_root: Path,
    seed: int = 215,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    cv_folds: int = 10,
    bootstrap_repeats: int = 1000,
    candidate_limit: int | None = None,
    feature_set: str = FEATURE_SET_BASE,
    tda_features_path: Path | None = None,
    nlp_features_path: Path | None = None,
    model_names: list[str] | None = None,
    selection_policy: str = SELECTION_POLICY_REGRET_FIRST,
    requested_device: str = "auto",
    requested_dtype: str = "float32",
    lbfgs_max_iter: int = 300,
    lbfgs_tolerance: float = 1e-7,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = resolve_runtime(requested_device, requested_dtype)
    prepared = prepare_router_data(
        input_path,
        repo_root,
        seed,
        validation_size,
        test_size,
        feature_set=feature_set,
        tda_features_path=tda_features_path,
        nlp_features_path=nlp_features_path,
        model_names=model_names,
    )
    df = prepared["df"]
    train_df = prepared["train_df"]
    validation_df = prepared["validation_df"]
    test_df = prepared["test_df"]
    numeric_columns = prepared["numeric_columns"]
    categorical_columns = prepared["categorical_columns"]
    model_names = prepared["model_names"]
    target_columns = prepared["target_columns"]
    input_dim = encoded_feature_count(train_df, numeric_columns, categorical_columns)
    candidates = candidate_grid(seed, input_dim, len(target_columns), candidate_limit=candidate_limit)

    cv_metric_rows: list[dict[str, Any]] = []
    cv_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    cv_fit_seconds = 0.0
    for candidate in candidates:
        metrics, predictions_df, elapsed = evaluate_candidate_cv_torch(
            train_df,
            candidate,
            numeric_columns,
            categorical_columns,
            cv_folds,
            runtime,
            model_names,
            max_iter=lbfgs_max_iter,
            tolerance=lbfgs_tolerance,
        )
        cv_metric_rows.append(metrics)
        cv_predictions_by_candidate[metrics["candidate_id"]] = predictions_df
        cv_fit_seconds += elapsed

    cv_metrics_df = pd.DataFrame(cv_metric_rows).sort_values(
        by=["mean_regret", "top1_hit_rate", "parameter_count", "candidate_id"],
        ascending=[True, False, True, True],
    )
    selected_candidate_row = sorted(
        cv_metrics_df.to_dict("records"),
        key=lambda row: candidate_sort_key(row, selection_policy),
    )[0]
    candidate_lookup = {candidate_id(candidate): candidate for candidate in candidates}
    final_candidate = candidate_lookup[selected_candidate_row["candidate_id"]]

    _, _, validation_predictions, validation_metrics, validation_fit_seconds = fit_predict_torch(
        train_df,
        validation_df,
        final_candidate,
        numeric_columns,
        categorical_columns,
        "validation",
        runtime,
        model_names,
        max_iter=lbfgs_max_iter,
        tolerance=lbfgs_tolerance,
    )
    validation_metrics.update(candidate_metadata(final_candidate, selected_candidate_row["candidate_id"]))

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_preprocessor, final_model, test_predictions, test_metrics, final_fit_seconds = fit_predict_torch(
        train_validation_df,
        test_df,
        final_candidate,
        numeric_columns,
        categorical_columns,
        "test",
        runtime,
        model_names,
        max_iter=lbfgs_max_iter,
        tolerance=lbfgs_tolerance,
    )
    test_metrics.update(candidate_metadata(final_candidate, selected_candidate_row["candidate_id"]))
    save_torch_bundle(
        out_dir / "final_model.pt",
        final_preprocessor,
        final_model,
        final_candidate,
        numeric_columns,
        categorical_columns,
        model_names,
        runtime,
    )

    all_preprocessor, all_model, all_fit_seconds = fit_torch_model(
        df,
        final_candidate,
        numeric_columns,
        categorical_columns,
        runtime,
        target_columns,
        max_iter=lbfgs_max_iter,
        tolerance=lbfgs_tolerance,
    )
    save_torch_bundle(
        out_dir / "final_model_all_data.pt",
        all_preprocessor,
        all_model,
        final_candidate,
        numeric_columns,
        categorical_columns,
        model_names,
        runtime,
    )

    overfitting_df = evaluate_overfitting_curve_torch(
        train_df,
        validation_df,
        candidates,
        numeric_columns,
        categorical_columns,
        runtime,
        model_names,
        max_iter=lbfgs_max_iter,
        tolerance=lbfgs_tolerance,
    )
    overfitting_df.to_csv(out_dir / "overfitting_curve.csv", index=False)
    write_overfitting_curve_plot(
        overfitting_df,
        out_dir / "overfitting_curve.png",
        "PyTorch MLP Overfitting Curve",
    )

    bootstrap_summary = write_bootstrap_outputs(test_predictions, out_dir, bootstrap_repeats, seed)
    training_seconds = time.perf_counter() - started
    runtime_info = runtime.to_dict()
    runtime_info.update(
        {
            "lbfgs_max_iter": int(lbfgs_max_iter),
            "lbfgs_tolerance": float(lbfgs_tolerance),
            "cv_fit_seconds": round(float(cv_fit_seconds), 6),
            "validation_fit_seconds": round(float(validation_fit_seconds), 6),
            "final_fit_seconds": round(float(final_fit_seconds), 6),
            "all_data_fit_seconds": round(float(all_fit_seconds), 6),
            "training_seconds": round(float(training_seconds), 6),
        }
    )
    test_metrics["torch_runtime"] = runtime_info

    metrics_json = write_common_training_outputs(
        out_dir=out_dir,
        input_path=input_path,
        repo_root=repo_root,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
        cv_folds=cv_folds,
        model_family="torch_mlp",
        df=df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        split_counts=prepared["split_counts"],
        group_counts=prepared["group_counts"],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        candidates=candidates,
        cv_metrics_df=cv_metrics_df,
        selected_candidate_row=selected_candidate_row,
        cv_predictions=cv_predictions_by_candidate[selected_candidate_row["candidate_id"]],
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
        feature_set=prepared["feature_set"],
        tda_features_path=Path(prepared["tda_features_path"]) if prepared["tda_features_path"] else None,
        nlp_features_path=Path(prepared["nlp_features_path"]) if prepared["nlp_features_path"] else None,
        tda_feature_columns=prepared["tda_feature_columns"],
        nlp_feature_columns=prepared["nlp_feature_columns"],
        tda_coverage=prepared["tda_coverage"],
        nlp_coverage=prepared["nlp_coverage"],
        model_names=model_names,
        selection_policy=selection_policy,
    )
    metrics_json["validation_summary"] = [validation_metrics]
    metrics_json["candidate_limit"] = candidate_limit
    metrics_json["torch_runtime"] = runtime_info
    write_json(out_dir / "metrics.json", metrics_json)
    update_run_config(
        out_dir,
        {
            "torch_runtime": runtime_info,
            "candidate_limit": candidate_limit,
            "model_file": (out_dir / "final_model.pt").as_posix(),
            "all_data_model_file": (out_dir / "final_model_all_data.pt").as_posix(),
        },
    )
    return metrics_json


def main() -> None:
    args = parse_args()
    tda_features_path = (
        Path(args.tda_features)
        if args.feature_set in {FEATURE_SET_BASE_TDA, FEATURE_SET_BASE_TDA_NLP} and args.tda_features
        else None
    )
    nlp_features_path = (
        Path(args.nlp_features)
        if args.feature_set in {FEATURE_SET_BASE_NLP, FEATURE_SET_BASE_TDA_NLP} and args.nlp_features
        else None
    )
    metrics = train_router(
        input_path=Path(args.input),
        out_dir=Path(args.out_dir),
        repo_root=Path(args.repo_root),
        seed=args.seed,
        validation_size=args.validation_size,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        bootstrap_repeats=args.bootstrap_repeats,
        candidate_limit=args.candidate_limit,
        feature_set=args.feature_set,
        tda_features_path=tda_features_path,
        nlp_features_path=nlp_features_path,
        model_names=args.models,
        selection_policy=args.selection_policy,
        requested_device=args.device,
        requested_dtype=args.dtype,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_tolerance=args.lbfgs_tolerance,
    )
    print(json.dumps(metrics["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
