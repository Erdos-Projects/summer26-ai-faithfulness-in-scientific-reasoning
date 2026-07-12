#!/usr/bin/env python3
"""Compare expanded SciVer router model outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "Data" / ".cache" / "matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MODEL_DIRS = [
    "Data/derived/model_router/public_router/runs/regret_first/ridge_elasticnet_base",
    "Data/derived/model_router/public_router/runs/regret_first/xgboost_base",
]
DEFAULT_OUT_DIR = "Data/derived/model_router/public_router/comparisons/regret_first_default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", action="append", dest="model_dirs", default=None)
    parser.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        default=None,
        help="Optional label for each --model-dir, in the same order.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def derive_run_label(model_dir: Path, model_family: str, feature_set: str | None) -> str:
    family = "ridge" if model_family == "ridge_elasticnet" else model_family
    if feature_set == "base_tda" or model_dir.name.endswith("_tda"):
        suffix = "tda"
    else:
        suffix = "base"
    return f"{family}_{suffix}"


def model_row(model_dir: Path, run_label: str | None = None) -> dict[str, Any]:
    metrics = load_json(model_dir / "metrics.json")
    bootstrap = load_json(model_dir / "bootstrap_summary.json")
    config_path = model_dir / "router_run_config.json"
    config = load_json(config_path) if config_path.exists() else {}
    test_metrics = metrics["test_metrics"]
    selected = metrics["selected_candidate"]
    intervals = bootstrap.get("metrics", {})
    top1 = intervals.get("top1_hit_rate", {})
    regret = intervals.get("mean_regret", {})
    model_family = metrics.get("model_family", model_dir.name)
    feature_set = metrics.get("feature_set") or config.get("feature_set") or "base"
    torch_runtime = metrics.get("torch_runtime") or test_metrics.get("torch_runtime") or config.get("torch_runtime", {})
    label = run_label or derive_run_label(model_dir, model_family, feature_set)
    return {
        "run_label": label,
        "model_family": model_family,
        "feature_set": feature_set,
        "selection_policy": metrics.get("selection_policy", config.get("selection_policy", "regret_first")),
        "torch_dtype": torch_runtime.get("torch_dtype", ""),
        "effective_device": torch_runtime.get("effective_device", ""),
        "requested_device": torch_runtime.get("requested_device", ""),
        "mps_available": torch_runtime.get("mps_available", ""),
        "training_seconds": torch_runtime.get("training_seconds", ""),
        "runtime_fallback_reason": torch_runtime.get("fallback_reason", ""),
        "model_dir": model_dir.as_posix(),
        "rows": metrics.get("rows"),
        "test_rows": test_metrics.get("rows"),
        "selected_candidate_id": selected.get("candidate_id"),
        "test_top1_hit_rate": test_metrics.get("top1_hit_rate"),
        "test_mean_regret": test_metrics.get("mean_regret"),
        "test_median_regret": test_metrics.get("median_regret"),
        "top1_p025": top1.get("p025"),
        "top1_p975": top1.get("p975"),
        "mean_regret_p025": regret.get("p025"),
        "mean_regret_p975": regret.get("p975"),
    }


def write_metric_plot(frame: pd.DataFrame, metric: str, lower: str, upper: str, out_path: Path, ylabel: str) -> None:
    values = frame[metric].astype(float)
    lower_err = (values - frame[lower].astype(float)).clip(lower=0)
    upper_err = (frame[upper].astype(float) - values).clip(lower=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["run_label"], values, yerr=[lower_err, upper_err], capsize=5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compare_models(model_dirs: list[Path], out_dir: Path, model_labels: list[str] | None = None) -> dict[str, Any]:
    if model_labels is not None and len(model_labels) != len(model_dirs):
        raise ValueError("--model-label must be supplied the same number of times as --model-dir.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        model_row(model_dir, None if model_labels is None else model_labels[index])
        for index, model_dir in enumerate(model_dirs)
    ]
    frame = pd.DataFrame(rows).sort_values(["test_mean_regret", "test_top1_hit_rate"], ascending=[True, False])
    frame.to_csv(out_dir / "model_comparison_metrics.csv", index=False)
    write_metric_plot(
        frame,
        "test_top1_hit_rate",
        "top1_p025",
        "top1_p975",
        out_dir / "top1_hit_rate_comparison.png",
        "Test top-1 hit rate",
    )
    write_metric_plot(
        frame,
        "test_mean_regret",
        "mean_regret_p025",
        "mean_regret_p975",
        out_dir / "mean_regret_comparison.png",
        "Test mean regret",
    )
    summary = {
        "model_dirs": [path.as_posix() for path in model_dirs],
        "model_labels": frame["run_label"].tolist(),
        "best_by_mean_regret": frame.iloc[0].to_dict() if not frame.empty else {},
        "rows": frame.to_dict("records"),
    }
    (out_dir / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    model_dirs = [Path(path) for path in (args.model_dirs or DEFAULT_MODEL_DIRS)]
    summary = compare_models(model_dirs, Path(args.out_dir), args.model_labels)
    print(json.dumps(summary["best_by_mean_regret"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
