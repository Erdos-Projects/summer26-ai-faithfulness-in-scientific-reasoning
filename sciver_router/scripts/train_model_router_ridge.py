#!/usr/bin/env python3
"""Train a Ridge/ElasticNet router over matched SciVer model-result data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from router_model_lib import (
    DIMENSION_COLUMNS,
    FEATURE_SET_CHOICES,
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_TDA_NLP,
    MODEL_NAMES,
    SELECTION_POLICY_CHOICES,
    SELECTION_POLICY_REGRET_FIRST,
    SELECTION_POLICY_TOP1_FIRST,
    add_engineered_features,
    bootstrap_prediction_metrics,
    build_preprocessor,
    candidate_metadata,
    candidate_sort_key,
    evaluate_candidate_cv,
    evaluate_overfitting_curve,
    feature_columns,
    fit_all_data_model,
    fit_predict,
    group_kfold_splits,
    grouped_train_validation_test_split,
    prepare_router_data,
    prediction_frame,
    routing_metrics,
    write_bootstrap_outputs,
    write_common_training_outputs,
    write_json,
    write_overfitting_curve_plot,
)


DEFAULT_INPUT = "Data/derived/model_router/public_router/router_dataset_public.csv"
DEFAULT_OUT_DIR = "Data/derived/model_router/public_router/runs/regret_first/ridge_elasticnet_base"
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
    return parser.parse_args()


def candidate_grid(seed: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    order = 0
    for alpha in [100.0, 10.0, 1.0, 0.1, 0.01]:
        order += 1
        candidates.append(
            {
                "kind": "ridge",
                "alpha": alpha,
                "l1_ratio": None,
                "seed": seed,
                "complexity_score": 1.0 / alpha,
                "complexity_order": order,
                "complexity_label": f"ridge a={alpha}",
            }
        )
    for alpha in [1.0, 0.1, 0.01, 0.001]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            order += 1
            candidates.append(
                {
                    "kind": "elasticnet",
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "seed": seed,
                    "complexity_score": (1.0 / alpha) + l1_ratio,
                    "complexity_order": order,
                    "complexity_label": f"enet a={alpha} l1={l1_ratio}",
                }
            )
    return candidates


def candidate_id(candidate: dict[str, Any]) -> str:
    if candidate["kind"] == "ridge":
        return f"ridge_alpha={candidate['alpha']}"
    return f"elasticnet_alpha={candidate['alpha']}_l1={candidate['l1_ratio']}"


def make_model(candidate: dict[str, Any]) -> MultiOutputRegressor:
    if candidate["kind"] == "ridge":
        estimator = Ridge(alpha=float(candidate["alpha"]))
    else:
        estimator = ElasticNet(
            alpha=float(candidate["alpha"]),
            l1_ratio=float(candidate["l1_ratio"]),
            max_iter=20000,
            random_state=int(candidate["seed"]),
        )
    return MultiOutputRegressor(estimator)


def make_pipeline(
    candidate: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)),
            ("model", make_model(candidate)),
        ]
    )


def ridge_candidate_sort_key(
    metric_row: dict[str, Any],
    selection_policy: str = SELECTION_POLICY_REGRET_FIRST,
) -> tuple[float, float, int, float, str]:
    kind_penalty = 0 if metric_row.get("kind") == "ridge" else 1
    if selection_policy == SELECTION_POLICY_REGRET_FIRST:
        return (
            float(metric_row["mean_regret"]),
            -float(metric_row["top1_hit_rate"]),
            kind_penalty,
            float(metric_row.get("complexity_score", 0.0)),
            str(metric_row.get("candidate_id", "")),
        )
    if selection_policy == SELECTION_POLICY_TOP1_FIRST:
        return (
            -float(metric_row["top1_hit_rate"]),
            float(metric_row["mean_regret"]),
            kind_penalty,
            float(metric_row.get("complexity_score", 0.0)),
            str(metric_row.get("candidate_id", "")),
        )
    raise ValueError(f"Unknown selection policy {selection_policy!r}; expected one of {SELECTION_POLICY_CHOICES}.")


def export_coefficients(pipeline: Pipeline, out_path: Path, target_columns: list[str]) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    rows: list[dict[str, Any]] = []
    for target, estimator in zip(target_columns, model.estimators_):
        for feature, coefficient in zip(feature_names, estimator.coef_):
            rows.append(
                {
                    "target": target,
                    "model": target.removeprefix("acc_"),
                    "feature": feature,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )
    coef_df = pd.DataFrame(rows).sort_values(["target", "abs_coefficient"], ascending=[True, False])
    coef_df.to_csv(out_path, index=False)
    return coef_df


def train_router(
    input_path: Path,
    out_dir: Path,
    repo_root: Path,
    seed: int = 215,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    cv_folds: int = 10,
    bootstrap_repeats: int = 1000,
    feature_set: str = FEATURE_SET_BASE,
    tda_features_path: Path | None = None,
    nlp_features_path: Path | None = None,
    model_names: list[str] | None = None,
    selection_policy: str = SELECTION_POLICY_REGRET_FIRST,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    candidates = candidate_grid(seed)

    cv_metric_rows: list[dict[str, Any]] = []
    cv_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        metrics, predictions_df = evaluate_candidate_cv(
            train_df,
            candidate,
            make_pipeline,
            candidate_id,
            numeric_columns,
            categorical_columns,
            cv_folds,
            model_names,
        )
        cv_metric_rows.append(metrics)
        cv_predictions_by_candidate[metrics["candidate_id"]] = predictions_df

    cv_metrics_df = pd.DataFrame(cv_metric_rows).sort_values(
        by=["mean_regret", "top1_hit_rate", "kind", "complexity_score"],
        ascending=[True, False, True, True],
    )
    candidate_lookup = {candidate_id(candidate): candidate for candidate in candidates}

    best_by_kind_rows = []
    validation_rows = []
    validation_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    for kind, group in cv_metrics_df.groupby("kind", sort=False):
        best_cv = sorted(group.to_dict("records"), key=lambda row: ridge_candidate_sort_key(row, selection_policy))[0]
        selected_candidate = candidate_lookup[best_cv["candidate_id"]]
        _, validation_predictions, validation_metrics = fit_predict(
            train_df,
            validation_df,
            selected_candidate,
            make_pipeline,
            numeric_columns,
            categorical_columns,
            "validation",
            model_names,
        )
        validation_metrics.update(candidate_metadata(selected_candidate, best_cv["candidate_id"]))
        best_by_kind_rows.append(best_cv)
        validation_rows.append(validation_metrics)
        validation_predictions_by_candidate[best_cv["candidate_id"]] = validation_predictions

    validation_metrics_df = pd.DataFrame(validation_rows).sort_values(
        by=["mean_regret", "top1_hit_rate", "kind", "complexity_score"],
        ascending=[True, False, True, True],
    )
    final_validation_row = sorted(
        validation_metrics_df.to_dict("records"),
        key=lambda row: ridge_candidate_sort_key(row, selection_policy),
    )[0]
    final_candidate = candidate_lookup[final_validation_row["candidate_id"]]

    cv_predictions = cv_predictions_by_candidate[final_validation_row["candidate_id"]]
    validation_predictions = validation_predictions_by_candidate[final_validation_row["candidate_id"]]

    train_validation_df = pd.concat([train_df, validation_df], ignore_index=True)
    final_pipeline, test_predictions, test_metrics = fit_predict(
        train_validation_df,
        test_df,
        final_candidate,
        make_pipeline,
        numeric_columns,
        categorical_columns,
        "test",
        model_names,
    )
    test_metrics.update(candidate_metadata(final_candidate, final_validation_row["candidate_id"]))

    overfitting_df = evaluate_overfitting_curve(
        train_df,
        validation_df,
        candidates,
        make_pipeline,
        candidate_id,
        numeric_columns,
        categorical_columns,
        model_names,
    )
    overfitting_df.to_csv(out_dir / "overfitting_curve.csv", index=False)
    write_overfitting_curve_plot(
        overfitting_df,
        out_dir / "overfitting_curve.png",
        "Ridge/ElasticNet Overfitting Curve",
    )

    export_coefficients(final_pipeline, out_dir / "coefficients.csv", target_columns)
    joblib.dump(final_pipeline, out_dir / "final_model.joblib")
    all_data_pipeline = fit_all_data_model(
        df,
        final_candidate,
        make_pipeline,
        numeric_columns,
        categorical_columns,
        out_dir / "final_model_all_data.joblib",
        model_names,
    )
    export_coefficients(all_data_pipeline, out_dir / "coefficients_all_data.csv", target_columns)

    bootstrap_summary = write_bootstrap_outputs(test_predictions, out_dir, bootstrap_repeats, seed)

    metrics_json = write_common_training_outputs(
        out_dir=out_dir,
        input_path=input_path,
        repo_root=repo_root,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
        cv_folds=cv_folds,
        model_family="ridge_elasticnet",
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
        selected_candidate_row=final_validation_row,
        cv_predictions=cv_predictions,
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
    metrics_json["best_cv_by_kind"] = best_by_kind_rows
    metrics_json["validation_summary"] = validation_metrics_df.to_dict("records")
    write_json(out_dir / "metrics.json", metrics_json)
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
        feature_set=args.feature_set,
        tda_features_path=tda_features_path,
        nlp_features_path=nlp_features_path,
        model_names=args.models,
        selection_policy=args.selection_policy,
    )
    print(json.dumps(metrics["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
