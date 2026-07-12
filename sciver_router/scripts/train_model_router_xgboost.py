#!/usr/bin/env python3
"""Train an XGBoost tree-based router over matched SciVer model-result data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from router_model_lib import (
    FEATURE_SET_CHOICES,
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_TDA_NLP,
    MODEL_NAMES,
    SELECTION_POLICY_CHOICES,
    SELECTION_POLICY_REGRET_FIRST,
    build_preprocessor,
    candidate_metadata,
    candidate_sort_key,
    evaluate_candidate_cv,
    evaluate_overfitting_curve,
    fit_all_data_model,
    fit_predict,
    prepare_router_data,
    write_bootstrap_outputs,
    write_common_training_outputs,
    write_json,
    write_overfitting_curve_plot,
)


DEFAULT_INPUT = "Data/derived/model_router/public_router/router_dataset_public.csv"
DEFAULT_OUT_DIR = "Data/derived/model_router/public_router/runs/regret_first/xgboost_base"
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
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="Optional development/test limit. Omit for the full specified XGBoost grid.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        dest="candidate_ids",
        default=None,
        help="Optional candidate_id filter. May be supplied more than once.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker count inside each XGBoost regressor.")
    return parser.parse_args()


def candidate_grid(
    seed: int,
    candidate_limit: int | None = None,
    n_jobs: int = 1,
    candidate_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    order = 0
    for n_estimators in [50, 100, 200]:
        for max_depth in [1, 2, 3]:
            for learning_rate in [0.03, 0.05, 0.1]:
                for subsample in [0.8, 1.0]:
                    for colsample_bytree in [0.8, 1.0]:
                        for reg_lambda in [1.0, 5.0, 10.0]:
                            for reg_alpha in [0.0, 0.1]:
                                order += 1
                                complexity = float(n_estimators * max_depth)
                                candidates.append(
                                    {
                                        "kind": "xgboost",
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "reg_lambda": reg_lambda,
                                        "reg_alpha": reg_alpha,
                                        "seed": seed,
                                        "n_jobs": n_jobs,
                                        "complexity_score": complexity,
                                        "complexity_order": order,
                                        "complexity_label": f"{n_estimators} trees d={max_depth}",
                                    }
                                )
    if candidate_ids is not None:
        wanted = set(candidate_ids)
        candidates = [candidate for candidate in candidates if candidate_id(candidate) in wanted]
        found = {candidate_id(candidate) for candidate in candidates}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Unknown XGBoost candidate_id values: {', '.join(missing)}")
    if candidate_limit is not None:
        if candidate_limit < 1:
            raise ValueError("--candidate-limit must be positive when provided.")
        return candidates[:candidate_limit]
    return candidates


def candidate_id(candidate: dict[str, Any]) -> str:
    return (
        f"xgb_n={candidate['n_estimators']}"
        f"_depth={candidate['max_depth']}"
        f"_lr={candidate['learning_rate']}"
        f"_sub={candidate['subsample']}"
        f"_col={candidate['colsample_bytree']}"
        f"_l2={candidate['reg_lambda']}"
        f"_l1={candidate['reg_alpha']}"
    )


def make_model(candidate: dict[str, Any]) -> MultiOutputRegressor:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required for scripts/train_model_router_xgboost.py. "
            "Install project requirements with `pip install -r requirements.txt`. "
            "On macOS, XGBoost may also require the OpenMP runtime: "
            "`brew install libomp`."
        ) from exc

    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(candidate["n_estimators"]),
        max_depth=int(candidate["max_depth"]),
        learning_rate=float(candidate["learning_rate"]),
        subsample=float(candidate["subsample"]),
        colsample_bytree=float(candidate["colsample_bytree"]),
        reg_lambda=float(candidate["reg_lambda"]),
        reg_alpha=float(candidate["reg_alpha"]),
        random_state=int(candidate["seed"]),
        n_jobs=int(candidate.get("n_jobs", 1)),
        tree_method="hist",
        verbosity=0,
    )
    return MultiOutputRegressor(estimator)


def make_pipeline(
    candidate: dict[str, Any], numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
            ("model", make_model(candidate)),
        ]
    )


def export_feature_importances(pipeline: Pipeline, out_path: Path, target_columns: list[str]) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    rows: list[dict[str, Any]] = []
    for target, estimator in zip(target_columns, model.estimators_):
        importances = getattr(estimator, "feature_importances_", [])
        for feature, importance in zip(feature_names, importances):
            rows.append(
                {
                    "target": target,
                    "model": target.removeprefix("acc_"),
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])
    frame.to_csv(out_path, index=False)
    return frame


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
    n_jobs: int = 1,
    candidate_ids: list[str] | None = None,
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
    candidates = candidate_grid(seed, candidate_limit=candidate_limit, n_jobs=n_jobs, candidate_ids=candidate_ids)

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
        by=["mean_regret", "top1_hit_rate", "complexity_score", "candidate_id"],
        ascending=[True, False, True, True],
    )
    selected_candidate_row = sorted(
        cv_metrics_df.to_dict("records"),
        key=lambda row: candidate_sort_key(row, selection_policy),
    )[0]
    candidate_lookup = {candidate_id(candidate): candidate for candidate in candidates}
    final_candidate = candidate_lookup[selected_candidate_row["candidate_id"]]

    _, validation_predictions, validation_metrics = fit_predict(
        train_df,
        validation_df,
        final_candidate,
        make_pipeline,
        numeric_columns,
        categorical_columns,
        "validation",
        model_names,
    )
    validation_metrics.update(candidate_metadata(final_candidate, selected_candidate_row["candidate_id"]))

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
    test_metrics.update(candidate_metadata(final_candidate, selected_candidate_row["candidate_id"]))

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
        "XGBoost Overfitting Curve",
    )

    joblib.dump(final_pipeline, out_dir / "final_model.joblib")
    export_feature_importances(final_pipeline, out_dir / "feature_importances.csv", target_columns)
    all_data_pipeline = fit_all_data_model(
        df,
        final_candidate,
        make_pipeline,
        numeric_columns,
        categorical_columns,
        out_dir / "final_model_all_data.joblib",
        model_names,
    )
    export_feature_importances(all_data_pipeline, out_dir / "feature_importances_all_data.csv", target_columns)

    bootstrap_summary = write_bootstrap_outputs(test_predictions, out_dir, bootstrap_repeats, seed)
    metrics_json = write_common_training_outputs(
        out_dir=out_dir,
        input_path=input_path,
        repo_root=repo_root,
        seed=seed,
        validation_size=validation_size,
        test_size=test_size,
        cv_folds=cv_folds,
        model_family="xgboost",
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
        candidate_limit=args.candidate_limit,
        n_jobs=args.n_jobs,
        candidate_ids=args.candidate_ids,
        feature_set=args.feature_set,
        tda_features_path=tda_features_path,
        nlp_features_path=nlp_features_path,
        model_names=args.models,
        selection_policy=args.selection_policy,
    )
    print(json.dumps(metrics["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
