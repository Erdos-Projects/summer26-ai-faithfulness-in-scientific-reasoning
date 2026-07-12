"""Shared utilities for classical SciVer router models."""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "Data" / ".cache" / "matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_MODEL_NAMES = ["qwen3vl", "pixtral12b", "kimi_vl_a3b"]
MODEL_NAMES = DEFAULT_MODEL_NAMES
TARGET_COLUMNS = [f"acc_{model}" for model in MODEL_NAMES]
SELECTION_POLICY_REGRET_FIRST = "regret_first"
SELECTION_POLICY_TOP1_FIRST = "top1_first"
SELECTION_POLICY_CHOICES = [SELECTION_POLICY_REGRET_FIRST, SELECTION_POLICY_TOP1_FIRST]
DIMENSION_COLUMNS = [
    "dim_AS",
    "dim_AT",
    "dim_MCr",
    "dim_QLl",
    "dim_QLq",
    "dim_VO",
    "dim_CL",
    "dim_GS",
    "dim_KNf",
    "dim_MA",
    "dim_MCu",
    "dim_VL",
]
BASE_NUMERIC_COLUMNS = DIMENSION_COLUMNS + [
    "context_char_count",
    "claim_char_count",
    "caption_char_count",
    "claim_word_count",
    "caption_word_count",
    "claim_numeric_token_count",
    "caption_numeric_token_count",
    "claim_percent_count",
    "caption_percent_count",
    "claim_math_symbol_count",
    "caption_math_symbol_count",
    "image_width",
    "image_height",
    "image_aspect_ratio",
]
BASE_CATEGORICAL_COLUMNS = ["claim_type", "evidence_kind", "vtype"]
STATIC_LEAKAGE_COLUMNS = {
    "label",
    "best_accuracy",
    "best_models",
    "tie_count",
    "best_margin",
}
LEAKAGE_PREFIXES = ("acc_", "majority_correct_", "parse_fail_count_", "model_id_", "run_started_at_utc_")
MODEL_METADATA_PREFIXES = ("majority_correct_", "parse_fail_count_", "model_id_", "run_started_at_utc_")
LEAKAGE_COLUMNS = STATIC_LEAKAGE_COLUMNS | {
    f"{prefix}{model}" for model in MODEL_NAMES for prefix in MODEL_METADATA_PREFIXES
}
FEATURE_SET_BASE = "base"
FEATURE_SET_BASE_TDA = "base_tda"
FEATURE_SET_BASE_NLP = "base_nlp"
FEATURE_SET_BASE_TDA_NLP = "base_tda_nlp"
FEATURE_SET_CHOICES = (
    FEATURE_SET_BASE,
    FEATURE_SET_BASE_TDA,
    FEATURE_SET_BASE_NLP,
    FEATURE_SET_BASE_TDA_NLP,
)
TDA_METADATA_COLUMNS = {
    "tda_parse_ok",
    "tda_preprocess_max_size",
    "tda_threshold_count",
    "tda_source_image_width",
    "tda_source_image_height",
}
NLP_METADATA_COLUMNS = {
    "nlp_parse_ok",
    "nlp_svd_components",
    "nlp_tda_components",
    "nlp_tda_max_points",
    "nlp_max_features",
    "nlp_extraction_error",
}
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
MATH_SYMBOL_RE = re.compile(r"[=<>±≈∼~+\-*/^%αβγλΣ∑√≤≥]")

CandidateGrid = list[dict[str, Any]]
CandidateIdFn = Callable[[dict[str, Any]], str]
MakePipelineFn = Callable[[dict[str, Any], list[str], list[str]], Pipeline]


def normalize_model_names(model_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    names = list(model_names or MODEL_NAMES)
    if not names:
        raise ValueError("At least one router target model is required.")
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate model names are not allowed: {', '.join(sorted(duplicates))}")
    return names


def target_columns_for(model_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    return [f"acc_{model}" for model in normalize_model_names(model_names)]


def model_names_from_columns(df: pd.DataFrame) -> list[str]:
    return [column.removeprefix("acc_") for column in df.columns if column.startswith("acc_")]


def leakage_columns_for(
    model_names: list[str] | tuple[str, ...] | None = None,
    columns: list[str] | pd.Index | None = None,
) -> set[str]:
    names = normalize_model_names(model_names)
    leakage = set(STATIC_LEAKAGE_COLUMNS)
    leakage.update(f"{prefix}{model}" for model in names for prefix in MODEL_METADATA_PREFIXES)
    leakage.update(target_columns_for(names))
    if columns is not None:
        leakage.update(str(column) for column in columns if str(column).startswith(LEAKAGE_PREFIXES))
    return leakage


def text_or_empty(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_numbers(text: str) -> int:
    return len(NUMBER_RE.findall(text))


def count_percents(text: str) -> int:
    return text.count("%")


def count_math_symbols(text: str) -> int:
    return len(MATH_SYMBOL_RE.findall(text))


def resolve_image_path(row: pd.Series, repo_root: Path) -> Path | None:
    raw_paths = [
        text.strip()
        for text in text_or_empty(row.get("image_paths")).split(";")
        if text.strip()
    ]
    candidates: list[Path] = []
    for raw in raw_paths:
        raw_path = Path(raw)
        candidates.append(repo_root / raw_path)
        if raw.startswith("./SciVer/images/"):
            candidates.append(repo_root / "Data" / "images" / raw.removeprefix("./SciVer/images/"))
        elif raw.startswith("SciVer/images/"):
            candidates.append(repo_root / "Data" / "images" / raw.removeprefix("SciVer/images/"))
        candidates.append(repo_root / "Data" / "images" / raw_path.name)

    annotation_image_file = text_or_empty(row.get("annotation_image_file"))
    if annotation_image_file:
        candidates.append(repo_root / "Data" / "images" / annotation_image_file)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def image_dimensions(row: pd.Series, repo_root: Path) -> tuple[float, float, float]:
    path = resolve_image_path(row, repo_root)
    if path is None:
        return np.nan, np.nan, np.nan
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return np.nan, np.nan, np.nan
    aspect_ratio = width / height if height else np.nan
    return float(width), float(height), float(aspect_ratio)


def add_engineered_features(df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    engineered = df.copy()
    if "claim" in engineered.columns:
        claim_text = engineered["claim"].map(text_or_empty)
        if "claim_char_count" not in engineered.columns:
            engineered["claim_char_count"] = claim_text.str.len()
        if "claim_word_count" not in engineered.columns:
            engineered["claim_word_count"] = claim_text.map(count_words)
        if "claim_numeric_token_count" not in engineered.columns:
            engineered["claim_numeric_token_count"] = claim_text.map(count_numbers)
        if "claim_percent_count" not in engineered.columns:
            engineered["claim_percent_count"] = claim_text.map(count_percents)
        if "claim_math_symbol_count" not in engineered.columns:
            engineered["claim_math_symbol_count"] = claim_text.map(count_math_symbols)

    if "caption_1" in engineered.columns or "caption_2" in engineered.columns:
        caption_text = (
            engineered.get("caption_1", pd.Series("", index=engineered.index)).map(text_or_empty)
            + " "
            + engineered.get("caption_2", pd.Series("", index=engineered.index)).map(text_or_empty)
        )
        if "caption_char_count" not in engineered.columns:
            engineered["caption_char_count"] = caption_text.str.len()
        if "caption_word_count" not in engineered.columns:
            engineered["caption_word_count"] = caption_text.map(count_words)
        if "caption_numeric_token_count" not in engineered.columns:
            engineered["caption_numeric_token_count"] = caption_text.map(count_numbers)
        if "caption_percent_count" not in engineered.columns:
            engineered["caption_percent_count"] = caption_text.map(count_percents)
        if "caption_math_symbol_count" not in engineered.columns:
            engineered["caption_math_symbol_count"] = caption_text.map(count_math_symbols)

    image_columns = ["image_width", "image_height", "image_aspect_ratio"]
    if not all(column in engineered.columns for column in image_columns):
        dimensions = engineered.apply(lambda row: image_dimensions(row, repo_root), axis=1)
        engineered[image_columns] = pd.DataFrame(dimensions.tolist(), index=engineered.index)
    return engineered


def tda_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith("tda_")
        and column not in TDA_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def nlp_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column.startswith("nlp_")
        and column not in NLP_METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def feature_columns(
    df: pd.DataFrame,
    feature_set: str = FEATURE_SET_BASE,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature set {feature_set!r}. Expected one of {FEATURE_SET_CHOICES}.")
    numeric_columns = [column for column in BASE_NUMERIC_COLUMNS if column in df.columns]
    if feature_set in {FEATURE_SET_BASE_TDA, FEATURE_SET_BASE_TDA_NLP}:
        numeric_columns.extend(tda_feature_columns(df))
    if feature_set in {FEATURE_SET_BASE_NLP, FEATURE_SET_BASE_TDA_NLP}:
        numeric_columns.extend(nlp_feature_columns(df))
    categorical_columns = [column for column in BASE_CATEGORICAL_COLUMNS if column in df.columns]
    forbidden = leakage_columns_for(model_names, df.columns)
    for column in numeric_columns + categorical_columns:
        if column in forbidden or any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            raise ValueError(f"Leakage column selected as feature: {column}")
    return numeric_columns, categorical_columns


def merge_tda_features(df: pd.DataFrame, tda_features_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not tda_features_path.exists():
        raise FileNotFoundError(
            f"Missing TDA feature file: {tda_features_path}. "
            "Generate it with `python scripts/extract_chart_tda_features.py`."
        )
    tda_df = pd.read_csv(tda_features_path)
    if "item_id" not in tda_df.columns:
        raise ValueError(f"TDA feature file must contain item_id: {tda_features_path}")
    duplicate_ids = tda_df["item_id"][tda_df["item_id"].duplicated()].unique()
    if len(duplicate_ids):
        example = ", ".join(map(str, duplicate_ids[:5]))
        raise ValueError(f"TDA feature file has duplicate item_id values, including: {example}")

    keep_columns = ["item_id"] + [column for column in tda_df.columns if column.startswith("tda_")]
    merged = df.drop(columns=[column for column in df.columns if column.startswith("tda_")], errors="ignore").merge(
        tda_df[keep_columns], on="item_id", how="left"
    )
    status = merged.get("tda_parse_ok")
    coverage = {
        "tda_features_path": tda_features_path.as_posix(),
        "rows": int(len(merged)),
        "rows_with_tda_record": int(merged["tda_parse_ok"].notna().sum()) if status is not None else 0,
        "rows_parse_ok": int((merged["tda_parse_ok"] == 1).sum()) if status is not None else 0,
        "rows_parse_failed": int((merged["tda_parse_ok"] == 0).sum()) if status is not None else 0,
        "tda_numeric_feature_count": int(len(tda_feature_columns(merged))),
    }
    return merged, coverage


def precomputed_tda_coverage(df: pd.DataFrame) -> dict[str, Any]:
    status = df.get("tda_parse_ok")
    return {
        "tda_features_path": "precomputed_in_input_dataset",
        "rows": int(len(df)),
        "rows_with_tda_record": int(status.notna().sum()) if status is not None else int(len(df)),
        "rows_parse_ok": int((status == 1).sum()) if status is not None else int(len(df)),
        "rows_parse_failed": int((status == 0).sum()) if status is not None else 0,
        "tda_numeric_feature_count": int(len(tda_feature_columns(df))),
    }


def merge_nlp_features(df: pd.DataFrame, nlp_features_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not nlp_features_path.exists():
        raise FileNotFoundError(
            f"Missing NLP feature file: {nlp_features_path}. "
            "Generate it with `python scripts/extract_claim_nlp_features.py`."
        )
    nlp_df = pd.read_csv(nlp_features_path)
    if "item_id" not in nlp_df.columns:
        raise ValueError(f"NLP feature file must contain item_id: {nlp_features_path}")
    duplicate_ids = nlp_df["item_id"][nlp_df["item_id"].duplicated()].unique()
    if len(duplicate_ids):
        example = ", ".join(map(str, duplicate_ids[:5]))
        raise ValueError(f"NLP feature file has duplicate item_id values, including: {example}")

    keep_columns = ["item_id"] + [column for column in nlp_df.columns if column.startswith("nlp_")]
    merged = df.drop(columns=[column for column in df.columns if column.startswith("nlp_")], errors="ignore").merge(
        nlp_df[keep_columns], on="item_id", how="left"
    )
    status = merged.get("nlp_parse_ok")
    coverage = {
        "nlp_features_path": nlp_features_path.as_posix(),
        "rows": int(len(merged)),
        "rows_with_nlp_record": int(merged["nlp_parse_ok"].notna().sum()) if status is not None else 0,
        "rows_parse_ok": int((merged["nlp_parse_ok"] == 1).sum()) if status is not None else 0,
        "rows_parse_failed": int((merged["nlp_parse_ok"] == 0).sum()) if status is not None else 0,
        "nlp_numeric_feature_count": int(len(nlp_feature_columns(merged))),
    }
    return merged, coverage


def precomputed_nlp_coverage(df: pd.DataFrame) -> dict[str, Any]:
    status = df.get("nlp_parse_ok")
    return {
        "nlp_features_path": "precomputed_in_input_dataset",
        "rows": int(len(df)),
        "rows_with_nlp_record": int(status.notna().sum()) if status is not None else int(len(df)),
        "rows_parse_ok": int((status == 1).sum()) if status is not None else int(len(df)),
        "rows_parse_failed": int((status == 0).sum()) if status is not None else 0,
        "nlp_numeric_feature_count": int(len(nlp_feature_columns(df))),
    }


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def grouped_train_validation_test_split(
    df: pd.DataFrame,
    group_col: str = "paper_id",
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 215,
) -> pd.Series:
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    group_sizes = df.groupby(group_col).size().sample(frac=1, random_state=seed)
    total_rows = len(df)
    train_target = round(total_rows * (1 - validation_size - test_size))
    validation_target = round(total_rows * validation_size)

    assignments: dict[str, str] = {}
    counts = Counter()
    for group, size in group_sizes.items():
        if counts["train"] < train_target:
            split = "train"
        elif counts["validation"] < validation_target:
            split = "validation"
        else:
            split = "test"
        assignments[str(group)] = split
        counts[split] += int(size)

    split_series = df[group_col].astype(str).map(assignments)
    if split_series.isna().any():
        raise ValueError("Some rows were not assigned to a split.")
    if set(split_series) != {"train", "validation", "test"}:
        raise ValueError(f"Expected train/validation/test splits, got {sorted(set(split_series))}.")
    return split_series


def group_kfold_splits(df: pd.DataFrame, group_col: str, requested_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = df[group_col].nunique()
    n_splits = min(requested_folds, unique_groups)
    if n_splits < 2:
        raise ValueError("At least two unique groups are required for grouped CV.")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(df, groups=df[group_col]))


def best_models_for_row(row: pd.Series) -> list[str]:
    value = text_or_empty(row.get("best_models"))
    return [item for item in value.split(";") if item]


def sanitize_predictions(predictions: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(finite, 0.0, 1.0)


def selected_model_from_predictions(
    predictions: np.ndarray,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    names = normalize_model_names(model_names)
    clipped = sanitize_predictions(predictions)
    if clipped.ndim != 2 or clipped.shape[1] != len(names):
        raise ValueError(
            f"Prediction array has shape {clipped.shape}; expected one column for each of {len(names)} models."
        )
    indices = np.argmax(clipped, axis=1)
    return [names[index] for index in indices]


def prediction_frame(
    df: pd.DataFrame,
    predictions: np.ndarray,
    split_name: str,
    fold: int | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    names = normalize_model_names(model_names)
    clipped = sanitize_predictions(predictions)
    selected_models = selected_model_from_predictions(clipped, names)
    rows: list[dict[str, Any]] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        selected = selected_models[idx]
        selected_accuracy = float(row[f"acc_{selected}"])
        best_accuracy = float(row["best_accuracy"])
        best_models = best_models_for_row(row)
        record = {
            "item_id": row["item_id"],
            "paper_id": row["paper_id"],
            "split": split_name,
            "source_split": row.get("split", ""),
            "fold": "" if fold is None else fold,
            "claim_type": row.get("claim_type", ""),
            "best_models": row["best_models"],
            "selected_model": selected,
            "selected_observed_accuracy": selected_accuracy,
            "best_accuracy": best_accuracy,
            "tie_aware_hit": selected in best_models,
            "regret": round(best_accuracy - selected_accuracy, 6),
        }
        for model_index, model in enumerate(names):
            record[f"observed_acc_{model}"] = float(row[f"acc_{model}"])
            record[f"pred_acc_{model}"] = round(float(clipped[idx, model_index]), 6)
        rows.append(record)
    return pd.DataFrame(rows)


def routing_metrics(
    frame: pd.DataFrame,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = normalize_model_names(model_names)
    if frame.empty:
        return {}
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "top1_hit_rate": round(float(frame["tie_aware_hit"].mean()), 6),
        "mean_regret": round(float(frame["regret"].mean()), 6),
        "median_regret": round(float(frame["regret"].median()), 6),
        "selected_model_counts": {
            str(key): int(value) for key, value in frame["selected_model"].value_counts().sort_index().items()
        },
    }
    for model in names:
        observed = frame[f"observed_acc_{model}"]
        predicted = frame[f"pred_acc_{model}"]
        metrics[f"mae_{model}"] = round(float(mean_absolute_error(observed, predicted)), 6)
        metrics[f"rmse_{model}"] = round(float(mean_squared_error(observed, predicted) ** 0.5), 6)

    metrics["by_claim_type"] = {
        str(key): {
            "rows": int(len(group)),
            "top1_hit_rate": round(float(group["tie_aware_hit"].mean()), 6),
            "mean_regret": round(float(group["regret"].mean()), 6),
        }
        for key, group in frame.groupby("claim_type")
    }
    if "source_split" in frame.columns:
        metrics["by_source_split"] = {
            str(key): {
                "rows": int(len(group)),
                "top1_hit_rate": round(float(group["tie_aware_hit"].mean()), 6),
                "mean_regret": round(float(group["regret"].mean()), 6),
            }
            for key, group in frame.groupby("source_split")
        }
    return metrics


def routing_metrics_for_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    metrics = {
        "rows": int(len(frame)),
        "top1_hit_rate": round(float(frame["tie_aware_hit"].mean()), 6),
        "mean_regret": round(float(frame["regret"].mean()), 6),
        "median_regret": round(float(frame["regret"].median()), 6),
        "selected_model_counts": {
            str(key): int(value) for key, value in frame["selected_model"].value_counts().sort_index().items()
        },
    }
    if "source_split" in frame.columns:
        metrics["by_source_split"] = {
            str(key): {
                "rows": int(len(group)),
                "top1_hit_rate": round(float(group["tie_aware_hit"].mean()), 6),
                "mean_regret": round(float(group["regret"].mean()), 6),
            }
            for key, group in frame.groupby("source_split")
        }
    return metrics


def candidate_sort_key(
    metric_row: dict[str, Any],
    selection_policy: str = SELECTION_POLICY_REGRET_FIRST,
) -> tuple[float, float, float, str]:
    if selection_policy == SELECTION_POLICY_REGRET_FIRST:
        return (
            float(metric_row["mean_regret"]),
            -float(metric_row["top1_hit_rate"]),
            float(metric_row.get("complexity_score", 0.0)),
            str(metric_row.get("candidate_id", "")),
        )
    if selection_policy == SELECTION_POLICY_TOP1_FIRST:
        return (
            -float(metric_row["top1_hit_rate"]),
            float(metric_row["mean_regret"]),
            float(metric_row.get("complexity_score", 0.0)),
            str(metric_row.get("candidate_id", "")),
        )
    raise ValueError(f"Unknown selection policy {selection_policy!r}; expected one of {SELECTION_POLICY_CHOICES}.")


def candidate_metadata(candidate: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"candidate_id": candidate_id}
    for key, value in candidate.items():
        if isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
            metadata[key] = value
    metadata.setdefault("kind", candidate.get("kind", "model"))
    metadata.setdefault("complexity_score", candidate.get("complexity_score", 0.0))
    metadata.setdefault("complexity_label", candidate_id)
    return metadata


def evaluate_candidate_cv(
    train_df: pd.DataFrame,
    candidate: dict[str, Any],
    make_pipeline: MakePipelineFn,
    candidate_id_fn: CandidateIdFn,
    numeric_columns: list[str],
    categorical_columns: list[str],
    cv_folds: int,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    splits = group_kfold_splits(train_df, "paper_id", cv_folds)
    fold_frames: list[pd.DataFrame] = []
    x = train_df[numeric_columns + categorical_columns]
    y = train_df[target_columns]

    for fold_index, (fit_indices, holdout_indices) in enumerate(splits, start=1):
        pipeline = make_pipeline(candidate, numeric_columns, categorical_columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pipeline.fit(x.iloc[fit_indices], y.iloc[fit_indices])
            predictions = pipeline.predict(x.iloc[holdout_indices])
        fold_df = train_df.iloc[holdout_indices].copy()
        fold_frames.append(prediction_frame(fold_df, predictions, "cv", fold=fold_index, model_names=names))

    predictions_df = pd.concat(fold_frames, ignore_index=True)
    candidate_id = candidate_id_fn(candidate)
    metrics = routing_metrics(predictions_df, names)
    metrics.update(candidate_metadata(candidate, candidate_id))
    metrics["cv_folds"] = len(splits)
    return metrics, predictions_df


def fit_predict(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate: dict[str, Any],
    make_pipeline: MakePipelineFn,
    numeric_columns: list[str],
    categorical_columns: list[str],
    split_name: str,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any]]:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    pipeline = make_pipeline(candidate, numeric_columns, categorical_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipeline.fit(fit_df[numeric_columns + categorical_columns], fit_df[target_columns])
        predictions = pipeline.predict(eval_df[numeric_columns + categorical_columns])
    frame = prediction_frame(eval_df, predictions, split_name, model_names=names)
    metrics = routing_metrics(frame, names)
    return pipeline, frame, metrics


def baseline_predictions(
    df: pd.DataFrame,
    strategy: str,
    train_df: pd.DataFrame,
    seed: int,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    if strategy.startswith("always_"):
        selected_model = strategy.removeprefix("always_")
        if selected_model not in names:
            raise ValueError(f"Unknown always baseline model {selected_model!r}; expected one of {names}.")
        selected = [selected_model] * len(df)
    elif strategy == "train_global_best":
        means = train_df[target_columns].mean()
        selected_model = names[int(np.argmax(means.to_numpy()))]
        selected = [selected_model] * len(df)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        selected = [names[index] for index in rng.integers(0, len(names), size=len(df))]
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    rows: list[dict[str, Any]] = []
    for (_, row), model in zip(df.iterrows(), selected):
        selected_accuracy = float(row[f"acc_{model}"])
        best_accuracy = float(row["best_accuracy"])
        best_models = best_models_for_row(row)
        rows.append(
            {
                "item_id": row["item_id"],
                "paper_id": row["paper_id"],
                "source_split": row.get("split", ""),
                "claim_type": row["claim_type"],
                "selected_model": model,
                "selected_observed_accuracy": selected_accuracy,
                "best_accuracy": best_accuracy,
                "tie_aware_hit": model in best_models,
                "regret": round(best_accuracy - selected_accuracy, 6),
            }
        )
    return pd.DataFrame(rows)


def baseline_metrics(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    names = normalize_model_names(model_names)
    output: dict[str, Any] = {}
    for strategy in [*(f"always_{model}" for model in names), "train_global_best", "random"]:
        output[strategy] = {
            "validation": routing_metrics_for_baseline(
                baseline_predictions(validation_df, strategy, train_df, seed, names)
            ),
            "test": routing_metrics_for_baseline(baseline_predictions(test_df, strategy, train_df, seed + 1, names)),
        }
    return output


def prepare_router_data(
    input_path: Path,
    repo_root: Path,
    seed: int,
    validation_size: float,
    test_size: float,
    feature_set: str = FEATURE_SET_BASE,
    tda_features_path: Path | None = None,
    nlp_features_path: Path | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    raw_df = pd.read_csv(input_path)
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    missing_targets = [column for column in target_columns if column not in raw_df.columns]
    if missing_targets:
        raise ValueError(f"Input dataset is missing target columns: {', '.join(missing_targets)}")
    target_null_counts = raw_df[target_columns].isna().sum()
    null_targets = {column: int(count) for column, count in target_null_counts.items() if int(count)}
    if null_targets:
        raise ValueError(f"Input dataset has missing target values: {null_targets}")

    df = add_engineered_features(raw_df, repo_root)
    tda_coverage: dict[str, Any] = {}
    nlp_coverage: dict[str, Any] = {}
    if feature_set in {FEATURE_SET_BASE_TDA, FEATURE_SET_BASE_TDA_NLP}:
        if tda_features_path is not None:
            df, tda_coverage = merge_tda_features(df, tda_features_path)
        elif tda_feature_columns(df):
            tda_coverage = precomputed_tda_coverage(df)
        else:
            raise ValueError(f"--tda-features is required when --feature-set {feature_set}.")
    if feature_set in {FEATURE_SET_BASE_NLP, FEATURE_SET_BASE_TDA_NLP}:
        if nlp_features_path is not None:
            df, nlp_coverage = merge_nlp_features(df, nlp_features_path)
        elif nlp_feature_columns(df):
            nlp_coverage = precomputed_nlp_coverage(df)
        else:
            raise ValueError(f"--nlp-features is required when --feature-set {feature_set}.")
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature set {feature_set!r}. Expected one of {FEATURE_SET_CHOICES}.")

    numeric_columns, categorical_columns = feature_columns(df, feature_set=feature_set, model_names=names)
    df["router_split"] = grouped_train_validation_test_split(
        df,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )
    train_df = df[df["router_split"] == "train"].copy()
    validation_df = df[df["router_split"] == "validation"].copy()
    test_df = df[df["router_split"] == "test"].copy()
    split_counts = {str(key): int(value) for key, value in df["router_split"].value_counts().sort_index().items()}
    group_counts = {
        str(split): int(part["paper_id"].nunique())
        for split, part in df.groupby("router_split")
    }
    return {
        "df": df,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "split_counts": split_counts,
        "group_counts": group_counts,
        "model_names": names,
        "target_columns": target_columns,
        "feature_set": feature_set,
        "tda_features_path": "" if tda_features_path is None else tda_features_path.as_posix(),
        "nlp_features_path": "" if nlp_features_path is None else nlp_features_path.as_posix(),
        "tda_feature_columns": [column for column in numeric_columns if column.startswith("tda_")],
        "nlp_feature_columns": [column for column in numeric_columns if column.startswith("nlp_")],
        "tda_coverage": tda_coverage,
        "nlp_coverage": nlp_coverage,
    }


def evaluate_overfitting_curve(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    candidates: CandidateGrid,
    make_pipeline: MakePipelineFn,
    candidate_id_fn: CandidateIdFn,
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    names = normalize_model_names(model_names)
    rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates, start=1):
        candidate_id = candidate_id_fn(candidate)
        pipeline, train_predictions, train_metrics = fit_predict(
            train_df,
            train_df,
            candidate,
            make_pipeline,
            numeric_columns,
            categorical_columns,
            "overfit_train",
            names,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions = pipeline.predict(validation_df[numeric_columns + categorical_columns])
        validation_predictions = prediction_frame(validation_df, predictions, "overfit_validation", model_names=names)
        validation_metrics = routing_metrics(validation_predictions, names)
        row = candidate_metadata(candidate, candidate_id)
        row.update(
            {
                "complexity_order": int(candidate.get("complexity_order", order)),
                "train_top1_hit_rate": train_metrics["top1_hit_rate"],
                "validation_top1_hit_rate": validation_metrics["top1_hit_rate"],
                "top1_hit_rate_gap": round(
                    train_metrics["top1_hit_rate"] - validation_metrics["top1_hit_rate"], 6
                ),
                "train_mean_regret": train_metrics["mean_regret"],
                "validation_mean_regret": validation_metrics["mean_regret"],
                "mean_regret_gap": round(
                    validation_metrics["mean_regret"] - train_metrics["mean_regret"], 6
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["complexity_order", "candidate_id"])


def write_overfitting_curve_plot(curve_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if curve_df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = curve_df.sort_values(["complexity_order", "candidate_id"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, min(24, len(plot_df) * 0.35)), 9), sharex=True)
    axes[0].plot(x, plot_df["train_mean_regret"], label="train", marker="o", linewidth=1)
    axes[0].plot(x, plot_df["validation_mean_regret"], label="validation", marker="o", linewidth=1)
    axes[0].set_ylabel("Mean regret")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, plot_df["train_top1_hit_rate"], label="train", marker="o", linewidth=1)
    axes[1].plot(x, plot_df["validation_top1_hit_rate"], label="validation", marker="o", linewidth=1)
    axes[1].set_ylabel("Top-1 hit rate")
    axes[1].set_xlabel("Candidate complexity order")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    if len(plot_df) <= 45:
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot_df["complexity_label"], rotation=90, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def bootstrap_prediction_metrics(
    frame: pd.DataFrame,
    group_col: str = "paper_id",
    n_bootstrap: int = 1000,
    seed: int = 215,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        empty = pd.DataFrame(columns=["replicate", "rows", "unique_groups", "top1_hit_rate", "mean_regret", "median_regret"])
        return empty, {}
    groups = np.array(sorted(frame[group_col].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    by_group = {group: part for group, part in frame.assign(_group=frame[group_col].astype(str)).groupby("_group")}
    for replicate in range(1, n_bootstrap + 1):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([by_group[group] for group in sampled_groups], ignore_index=True)
        rows.append(
            {
                "replicate": replicate,
                "rows": int(len(sampled)),
                "unique_groups": int(len(set(sampled_groups))),
                "top1_hit_rate": float(sampled["tie_aware_hit"].mean()),
                "mean_regret": float(sampled["regret"].mean()),
                "median_regret": float(sampled["regret"].median()),
            }
        )
    metrics_df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "group_col": group_col,
        "source_rows": int(len(frame)),
        "source_groups": int(len(groups)),
        "interval_method": "percentile",
        "metrics": {},
    }
    for metric in ["top1_hit_rate", "mean_regret", "median_regret"]:
        values = metrics_df[metric].to_numpy()
        summary["metrics"][metric] = {
            "mean": float(np.mean(values)),
            "p025": float(np.percentile(values, 2.5)),
            "p500": float(np.percentile(values, 50.0)),
            "p975": float(np.percentile(values, 97.5)),
        }
    return metrics_df, summary


def write_bootstrap_outputs(
    test_predictions: pd.DataFrame,
    out_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    bootstrap_df, bootstrap_summary = bootstrap_prediction_metrics(
        test_predictions,
        group_col="paper_id",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    bootstrap_df.to_csv(out_dir / "bootstrap_metrics.csv", index=False)
    write_json(out_dir / "bootstrap_summary.json", bootstrap_summary)
    return bootstrap_summary


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_for_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_common_training_outputs(
    *,
    out_dir: Path,
    input_path: Path,
    repo_root: Path,
    seed: int,
    validation_size: float,
    test_size: float,
    cv_folds: int,
    model_family: str,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_counts: dict[str, int],
    group_counts: dict[str, int],
    numeric_columns: list[str],
    categorical_columns: list[str],
    candidates: CandidateGrid,
    cv_metrics_df: pd.DataFrame,
    selected_candidate_row: dict[str, Any],
    cv_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    feature_set: str = FEATURE_SET_BASE,
    tda_features_path: Path | None = None,
    tda_feature_columns: list[str] | None = None,
    tda_coverage: dict[str, Any] | None = None,
    nlp_features_path: Path | None = None,
    nlp_feature_columns: list[str] | None = None,
    nlp_coverage: dict[str, Any] | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
    selection_policy: str = SELECTION_POLICY_REGRET_FIRST,
) -> dict[str, Any]:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    split_columns = ["item_id", "paper_id", "split", "claim_type", "best_models", "router_split"]
    df[split_columns].to_csv(out_dir / "splits.csv", index=False)
    cv_predictions.to_csv(out_dir / "cv_predictions.csv", index=False)
    validation_predictions.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    metrics_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": model_family,
        "feature_set": feature_set,
        "tda_coverage": tda_coverage or {},
        "nlp_coverage": nlp_coverage or {},
        "input_path": input_path.as_posix(),
        "rows": int(len(df)),
        "split_counts": split_counts,
        "group_counts": group_counts,
        "model_names": names,
        "selection_policy": selection_policy,
        "target_columns": target_columns,
        "candidate_cv_summary": cv_metrics_df.to_dict("records"),
        "selected_candidate": selected_candidate_row,
        "cv_metrics": routing_metrics(cv_predictions, names),
        "validation_metrics": routing_metrics(validation_predictions, names),
        "test_metrics": test_metrics,
        "bootstrap_summary": bootstrap_summary,
        "evaluation_note": (
            "Internal grouped holdout estimate over the combined val+test router dataset; "
            "not untouched official SciVer test performance."
        ),
    }
    write_json(out_dir / "metrics.json", metrics_json)
    write_json(out_dir / "baselines.json", baseline_metrics(train_df, validation_df, test_df, seed, names))
    write_json(
        out_dir / "feature_columns.json",
        {
            "feature_set": feature_set,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "tda_feature_columns": tda_feature_columns or [],
            "nlp_feature_columns": nlp_feature_columns or [],
            "model_names": names,
            "target_columns": target_columns,
            "excluded_leakage_columns": sorted(leakage_columns_for(names, df.columns)),
            "excluded_tda_metadata_columns": sorted(TDA_METADATA_COLUMNS),
            "excluded_nlp_metadata_columns": sorted(NLP_METADATA_COLUMNS),
        },
    )
    write_json(
        out_dir / "router_run_config.json",
        {
            "input": input_path.as_posix(),
            "out_dir": out_dir.as_posix(),
            "repo_root": repo_root.as_posix(),
            "feature_set": feature_set,
            "tda_features": "" if tda_features_path is None else tda_features_path.as_posix(),
            "nlp_features": "" if nlp_features_path is None else nlp_features_path.as_posix(),
            "tda_coverage": tda_coverage or {},
            "nlp_coverage": nlp_coverage or {},
            "seed": seed,
            "validation_size": validation_size,
            "test_size": test_size,
            "cv_folds_requested": cv_folds,
            "cv_folds_used": min(cv_folds, train_df["paper_id"].nunique()),
            "bootstrap_repeats": int(bootstrap_summary.get("n_bootstrap", 0)),
            "model_names": names,
            "selection_policy": selection_policy,
            "candidate_grid": candidates,
        },
    )
    return metrics_json


def fit_all_data_model(
    df: pd.DataFrame,
    candidate: dict[str, Any],
    make_pipeline: MakePipelineFn,
    numeric_columns: list[str],
    categorical_columns: list[str],
    out_path: Path,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> Pipeline:
    names = normalize_model_names(model_names)
    target_columns = target_columns_for(names)
    pipeline = make_pipeline(candidate, numeric_columns, categorical_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pipeline.fit(df[numeric_columns + categorical_columns], df[target_columns])
    joblib.dump(pipeline, out_path)
    return pipeline
