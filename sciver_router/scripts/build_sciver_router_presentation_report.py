"""Build a public presentation HTML report for the three-VLM SciVer router."""

from __future__ import annotations

import base64
import html
import io
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
import numpy as np
import pandas as pd

from router_model_lib import nlp_feature_columns, tda_feature_columns


DATASET = Path("Data/derived/model_router/public_router/router_dataset_public.csv")
AUDIT = Path("Data/derived/model_router/public_router/router_dataset_public_audit.json")
RUNS_ROOT = Path("Data/derived/model_router/public_router/runs")
COMPARISON_ROOT = Path("Data/derived/model_router/public_router/comparisons")
OUT = Path("docs/sciver_router_presentation.html")
MODEL_LABELS = {
    "qwen3vl": "Qwen3-VL",
    "pixtral12b": "Pixtral12B",
    "kimi_vl_a3b": "Kimi-VL-A3B",
}
ACC_COLUMNS = [f"acc_{model}" for model in MODEL_LABELS]
POLICY_LABELS = {
    "regret_first": "Mean-Regret-First Selection",
    "top1_first": "Top-1-Hit-First Selection",
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def ci_pct(row: pd.Series) -> str:
    return f"{pct(row['top1_p025'])} - {pct(row['top1_p975'])}"


def ci_num(row: pd.Series) -> str:
    return f"{num(row['mean_regret_p025'], 3)} - {num(row['mean_regret_p975'], 3)}"


def fig_to_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def file_to_uri(path: Path) -> str:
    require(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def table_html(df: pd.DataFrame, classes: str = "") -> str:
    return df.to_html(index=False, escape=True, classes=f"table {classes}".strip(), border=0)


def chart_block(title: str, uri: str, caption: str) -> str:
    return f"""
    <figure class="figure-card">
      <h3>{html.escape(title)}</h3>
      <img class="zoomable" src="{uri}" alt="{html.escape(title)}" loading="lazy" />
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def plot_accuracy_distributions(df: pd.DataFrame) -> str:
    values = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels = [MODEL_LABELS[column.removeprefix("acc_")] for column in ACC_COLUMNS]
    width = 0.055
    offsets = np.linspace(-width, width, len(ACC_COLUMNS))
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    for offset, column, label in zip(offsets, ACC_COLUMNS, labels):
        counts = [
            int(np.isclose(df[column].dropna().astype(float).to_numpy(), value, atol=1e-9).sum())
            for value in values
        ]
        ax.bar(values + offset, counts, width=width, label=label)
    ax.set_xlabel("Observed five-run accuracy")
    ax.set_ylabel("Examples")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:.1f}" for value in values])
    ax.set_xlim(-0.08, 1.08)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_uri(fig)


def plot_best_model_counts(df: pd.DataFrame) -> str:
    counts = df["best_models"].value_counts().sort_values(ascending=True)
    labels = [label.replace("qwen3vl", "Qwen").replace("pixtral12b", "Pixtral").replace("kimi_vl_a3b", "Kimi") for label in counts.index]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(labels, counts.values, color="#4477aa")
    ax.set_xlabel("Examples")
    ax.set_title("Observed best model or tie")
    ax.grid(axis="x", alpha=0.25)
    return fig_to_uri(fig)


def plot_feature_counts(df: pd.DataFrame) -> str:
    counts = {
        "12 cognitive dimensions": len([column for column in df.columns if column.startswith("dim_")]),
        "Base scalar features": len(
            [
                column
                for column in df.columns
                if column
                in {
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
                }
            ]
        ),
        "Chart TDA router features": len(tda_feature_columns(df)),
        "Claim NLP router features": len(nlp_feature_columns(df)),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b"])
    ax.set_ylabel("Router-used feature columns")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_uri(fig)


def plot_metric_comparison(metrics: pd.DataFrame, title: str) -> str:
    ordered = metrics.sort_values(["test_mean_regret", "test_top1_hit_rate"], ascending=[True, False]).copy()
    labels = ordered["run_label"].str.replace("_regret_first", "", regex=False).str.replace("_top1_first", "", regex=False)
    fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(ordered))
    ax1.bar(x - 0.2, ordered["test_top1_hit_rate"], width=0.4, color="#4c78a8", label="Top-1 hit rate")
    ax1.set_ylabel("Top-1 hit rate")
    ax1.set_ylim(0, max(0.62, ordered["test_top1_hit_rate"].max() + 0.08))
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, ordered["test_mean_regret"], width=0.4, color="#f58518", label="Mean regret")
    ax2.set_ylabel("Mean regret")
    ax2.set_ylim(0, max(0.42, ordered["test_mean_regret"].max() + 0.06))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right")
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.2)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)
    return fig_to_uri(fig)


def pretty_run_label(row: pd.Series) -> str:
    family = {
        "ridge_elasticnet": "Ridge",
        "xgboost": "XGBoost",
        "torch_mlp": "Torch MLP",
    }.get(str(row.get("model_family", "")), str(row.get("model_family", "")))
    feature = {
        "base": "base",
        "base_tda": "chart TDA",
        "base_nlp": "claim NLP",
        "base_tda_nlp": "chart+claim",
    }.get(str(row.get("feature_set", "")), str(row.get("feature_set", "")))
    policy = "regret" if row.get("selection_policy") == "regret_first" else "top-1"
    return f"{family} {feature}\n{policy}"


def plot_bootstrap_intervals(
    frame: pd.DataFrame,
    title: str,
    *,
    sort_by: str,
    top_n: int | None = None,
) -> str:
    if sort_by == "top1":
        ordered = frame.sort_values(["test_top1_hit_rate", "test_mean_regret"], ascending=[False, True]).copy()
    elif sort_by == "regret":
        ordered = frame.sort_values(["test_mean_regret", "test_top1_hit_rate"], ascending=[True, False]).copy()
    else:
        ordered = frame.copy()
    if top_n is not None:
        ordered = ordered.head(top_n).copy()

    labels = [pretty_run_label(row) for _, row in ordered.iterrows()]
    x = np.arange(len(ordered))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8.8, 1.2 * len(ordered)), 7.2), sharex=True)

    top1 = ordered["test_top1_hit_rate"].astype(float)
    top1_lower = (top1 - ordered["top1_p025"].astype(float)).clip(lower=0)
    top1_upper = (ordered["top1_p975"].astype(float) - top1).clip(lower=0)
    ax1.errorbar(x, top1, yerr=[top1_lower, top1_upper], fmt="o", color="#4c78a8", capsize=5)
    ax1.set_ylabel("Test top-1 hit rate")
    ax1.set_ylim(0, max(0.65, float((top1 + top1_upper).max()) + 0.04))
    ax1.grid(axis="y", alpha=0.25)

    regret = ordered["test_mean_regret"].astype(float)
    regret_lower = (regret - ordered["mean_regret_p025"].astype(float)).clip(lower=0)
    regret_upper = (ordered["mean_regret_p975"].astype(float) - regret).clip(lower=0)
    ax2.errorbar(x, regret, yerr=[regret_lower, regret_upper], fmt="o", color="#f58518", capsize=5)
    ax2.set_ylabel("Test mean regret")
    ax2.set_ylim(0, max(0.4, float((regret + regret_upper).max()) + 0.04))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    return fig_to_uri(fig)


def best_family_runs(regret_metrics: pd.DataFrame) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    ordered = regret_metrics.sort_values(["test_mean_regret", "test_top1_hit_rate"], ascending=[True, False])
    for family in ["ridge_elasticnet", "xgboost", "torch_mlp"]:
        row = ordered.loc[ordered["model_family"] == family].head(1)
        if not row.empty:
            runs[family] = Path(str(row.iloc[0]["model_dir"]))
    return runs


def zone_columns_for_overfit(frame: pd.DataFrame, family: str) -> list[str]:
    if family == "xgboost":
        return ["n_estimators", "max_depth"]
    if family == "torch_mlp":
        return ["hidden_layer_count"]
    if "kind" in frame.columns:
        return ["kind"]
    return []


def zone_label(values: tuple, family: str) -> str:
    if family == "xgboost":
        return f"{int(values[0])} trees, depth {int(values[1])}"
    if family == "torch_mlp":
        layers = int(values[0])
        return "1 hidden layer" if layers == 1 else f"{layers} hidden layers"
    return str(values[0])


def add_overfit_zones(ax: plt.Axes, frame: pd.DataFrame, family: str) -> None:
    zone_cols = zone_columns_for_overfit(frame, family)
    if not zone_cols:
        return
    zones = frame[zone_cols].apply(lambda row: tuple(row), axis=1).tolist()
    starts = [0]
    for index in range(1, len(zones)):
        if zones[index] != zones[index - 1]:
            starts.append(index)
    starts.append(len(frame))
    for start, end in zip(starts[:-1], starts[1:]):
        if start > 0:
            boundary = (frame.iloc[start - 1]["complexity_order"] + frame.iloc[start]["complexity_order"]) / 2
            ax.axhline(boundary, color="#8a93a3", linestyle=":", linewidth=0.8, alpha=0.65)
        center = (frame.iloc[start]["complexity_order"] + frame.iloc[end - 1]["complexity_order"]) / 2
        if family != "xgboost" or (end - start) >= 20:
            ax.text(
                1.02,
                center,
                zone_label(zones[start], family),
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8,
                color="#5b6577",
            )


def plot_overfitting_curve(run_dir: Path, family: str) -> str:
    frame = pd.read_csv(run_dir / "overfitting_curve.csv").sort_values("complexity_order").reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    selected_id = metrics["selected_candidate"]["candidate_id"]
    selected = frame.loc[frame["candidate_id"] == selected_id]
    selected_order = float(selected.iloc[0]["complexity_order"]) if not selected.empty else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 7.0), sharey=True)
    y = frame["complexity_order"].astype(float)
    ax1.plot(frame["train_top1_hit_rate"], y, color="#4c78a8", linewidth=1.5, label="train")
    ax1.plot(frame["validation_top1_hit_rate"], y, color="#4c78a8", linestyle="--", linewidth=1.5, label="validation")
    ax1.set_xlabel("Top-1 hit rate")
    ax1.set_ylabel("Candidate complexity order")
    ax1.grid(axis="x", alpha=0.25)
    ax1.legend(frameon=False, loc="lower right")

    ax2.plot(frame["train_mean_regret"], y, color="#f58518", linewidth=1.5, label="train")
    ax2.plot(frame["validation_mean_regret"], y, color="#f58518", linestyle="--", linewidth=1.5, label="validation")
    ax2.set_xlabel("Mean regret")
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(frameon=False, loc="lower right")

    for ax in [ax1, ax2]:
        add_overfit_zones(ax, frame, family)
        if selected_order is not None:
            ax.axhline(selected_order, color="#b23b3b", linestyle="-", linewidth=1.1, alpha=0.85)

    title = {
        "ridge_elasticnet": "Ridge/ElasticNet overfitting curve",
        "xgboost": "XGBoost overfitting curve",
        "torch_mlp": "PyTorch MLP overfitting curve",
    }.get(family, "Overfitting curve")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    return fig_to_uri(fig)


def selected_overfitting_table(run_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for family, run_dir in run_dirs.items():
        frame = pd.read_csv(run_dir / "overfitting_curve.csv")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        selected_id = metrics["selected_candidate"]["candidate_id"]
        selected = frame.loc[frame["candidate_id"] == selected_id]
        if selected.empty:
            continue
        row = selected.iloc[0]
        top1_gap = float(row["top1_hit_rate_gap"])
        regret_gap = float(row["mean_regret_gap"])
        if top1_gap > 0.20 or regret_gap > 0.12:
            reading = "strong overfitting signal"
        elif top1_gap > 0.10 or regret_gap > 0.06:
            reading = "moderate overfitting signal"
        else:
            reading = "limited overfitting signal"
        rows.append(
            {
                "router family": {
                    "ridge_elasticnet": "Ridge/ElasticNet",
                    "xgboost": "XGBoost",
                    "torch_mlp": "PyTorch MLP",
                }.get(family, family),
                "selected run": run_dir.name,
                "selected candidate": selected_id,
                "train top-1": pct(row["train_top1_hit_rate"]),
                "validation top-1": pct(row["validation_top1_hit_rate"]),
                "top-1 gap": pct(top1_gap),
                "train regret": num(row["train_mean_regret"], 3),
                "validation regret": num(row["validation_mean_regret"], 3),
                "regret gap": num(regret_gap, 3),
                "reading": reading,
            }
        )
    return pd.DataFrame(rows)


def load_policy_metrics(policy: str) -> pd.DataFrame:
    path = COMPARISON_ROOT / policy / "model_comparison_metrics.csv"
    require(path)
    return pd.read_csv(path)


def compact_metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output = output[
        [
            "run_label",
            "model_family",
            "feature_set",
            "selection_policy",
            "test_top1_hit_rate",
            "test_mean_regret",
            "test_median_regret",
            "top1_p025",
            "top1_p975",
            "mean_regret_p025",
            "mean_regret_p975",
            "selected_candidate_id",
        ]
    ]
    output["test_top1_hit_rate"] = output["test_top1_hit_rate"].map(pct)
    output["test_mean_regret"] = output["test_mean_regret"].map(lambda x: num(x, 3))
    output["test_median_regret"] = output["test_median_regret"].map(lambda x: num(x, 3))
    output["top1 95% CI"] = output.apply(lambda row: f"{pct(row['top1_p025'])} - {pct(row['top1_p975'])}", axis=1)
    output["regret 95% CI"] = output.apply(lambda row: f"{num(row['mean_regret_p025'], 3)} - {num(row['mean_regret_p975'], 3)}", axis=1)
    output = output.drop(columns=["top1_p025", "top1_p975", "mean_regret_p025", "mean_regret_p975"])
    return output.rename(
        columns={
            "run_label": "run",
            "model_family": "router",
            "feature_set": "features",
            "selection_policy": "selection",
            "test_top1_hit_rate": "test top-1",
            "test_mean_regret": "test mean regret",
            "test_median_regret": "test median regret",
            "selected_candidate_id": "selected candidate",
        }
    )


def baseline_table(best_run_dir: Path) -> pd.DataFrame:
    data = json.loads((best_run_dir / "baselines.json").read_text(encoding="utf-8"))
    rows = []
    for name, sections in data.items():
        test = sections["test"]
        rows.append(
            {
                "baseline": name,
                "type": "stochastic" if name == "random" else "deterministic",
                "test top-1": pct(test["top1_hit_rate"]),
                "test mean regret": num(test["mean_regret"], 3),
                "selected models": ", ".join(f"{key}: {value}" for key, value in test["selected_model_counts"].items()),
            }
        )
    return pd.DataFrame(rows).sort_values(["type", "test mean regret", "baseline"])


def selected_model_table(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        counts = metrics["test_metrics"].get("selected_model_counts", {})
        rows.append(
            {
                "run": run_dir.parent.name + "/" + run_dir.name,
                "qwen3vl": counts.get("qwen3vl", 0),
                "pixtral12b": counts.get("pixtral12b", 0),
                "kimi_vl_a3b": counts.get("kimi_vl_a3b", 0),
            }
        )
    return pd.DataFrame(rows)


def methods_table(splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group = splits.groupby("router_split")
    for split_name in ["train", "validation", "test"]:
        if split_name in group.groups:
            part = group.get_group(split_name)
            rows.append(
                {
                    "router split": split_name,
                    "rows": len(part),
                    "paper groups": part["paper_id"].nunique(),
                }
            )
    rows.extend(
        [
            {"router split": "grouped CV", "rows": "10 folds inside train", "paper groups": "paper_id kept intact"},
            {"router split": "bootstrap", "rows": "1,000 held-out resamples", "paper groups": "resampled with replacement"},
            {"router split": "seed", "rows": 215, "paper groups": ""},
        ]
    )
    return pd.DataFrame(rows)


def feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature family": "12 cognitive dimensions",
                "public columns": len([column for column in df.columns if column.startswith("dim_")]),
                "meaning": "Human-coded chart difficulty annotations included as precomputed public features; deployment would require these annotations or an automatic proxy.",
            },
            {
                "feature family": "Base scalar features",
                "public columns": len(
                    [
                        column
                        for column in df.columns
                        if column
                        in {
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
                        }
                    ]
                ),
                "meaning": "Simple size, token, numeric, and image-shape summaries.",
            },
            {
                "feature family": "Chart TDA",
                "public columns": len([column for column in df.columns if column.startswith("tda_") and pd.api.types.is_numeric_dtype(df[column])]),
                "meaning": "Topological summaries of grayscale and edge-map chart structure.",
            },
            {
                "feature family": "Claim NLP",
                "public columns": len([column for column in df.columns if column.startswith("nlp_") and pd.api.types.is_numeric_dtype(df[column])]),
                "meaning": "TF-IDF/SVD semantics, cue counts, alignment, numeric overlap, and token-level TDA.",
            },
        ]
    )


def build_html() -> str:
    require(DATASET)
    require(AUDIT)
    df = pd.read_csv(DATASET)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    regret = load_policy_metrics("regret_first")
    top1 = load_policy_metrics("top1_first")
    all_metrics = pd.concat([regret, top1], ignore_index=True)
    best = regret.sort_values(["test_mean_regret", "test_top1_hit_rate"], ascending=[True, False]).iloc[0]
    best_dir = Path(best["model_dir"])
    family_runs = best_family_runs(regret)
    splits = pd.read_csv(best_dir / "splits.csv")

    dataset_summary = pd.DataFrame(
        [
            {"quantity": "public examples", "value": len(df)},
            {"quantity": "validation-origin rows", "value": int((df["split"] == "val").sum())},
            {"quantity": "test-origin rows", "value": int((df["split"] == "test").sum())},
            {"quantity": "paper groups", "value": df["paper_id"].nunique()},
            {"quantity": "direct claims", "value": int((df["claim_type"] == "direct").sum())},
            {"quantity": "analytical claims", "value": int((df["claim_type"] == "analytical").sum())},
            {"quantity": "public feature columns", "value": audit["column_count"]},
        ]
    )
    mean_accuracy = pd.DataFrame(
        {
            "VLM": [MODEL_LABELS[column.removeprefix("acc_")] for column in ACC_COLUMNS],
            "mean observed accuracy": [pct(df[column].mean()) for column in ACC_COLUMNS],
        }
    )

    images = {
        "accuracy": plot_accuracy_distributions(df),
        "best": plot_best_model_counts(df),
        "features": plot_feature_counts(df),
        "regret": plot_metric_comparison(regret, "Regret-first policy: top-1 and mean regret by router"),
        "top1": plot_metric_comparison(top1, "Top-1-first policy: top-1 and mean regret by router"),
        "bootstrap_ridge": plot_bootstrap_intervals(
            all_metrics.loc[all_metrics["model_family"] == "ridge_elasticnet"],
            "Ridge/ElasticNet bootstrap intervals",
            sort_by="regret",
        ),
        "bootstrap_xgb": plot_bootstrap_intervals(
            all_metrics.loc[all_metrics["model_family"] == "xgboost"],
            "XGBoost bootstrap intervals",
            sort_by="regret",
        ),
        "bootstrap_mlp": plot_bootstrap_intervals(
            all_metrics.loc[all_metrics["model_family"] == "torch_mlp"],
            "PyTorch MLP bootstrap intervals",
            sort_by="regret",
        ),
        "bootstrap_regret_top5": plot_bootstrap_intervals(
            regret,
            "Mean-regret-first policy: five best point estimates",
            sort_by="regret",
            top_n=5,
        ),
        "bootstrap_top1_top5": plot_bootstrap_intervals(
            top1,
            "Top-1-first policy: five best point estimates",
            sort_by="top1",
            top_n=5,
        ),
    }
    if "ridge_elasticnet" in family_runs:
        images["overfit_ridge"] = plot_overfitting_curve(family_runs["ridge_elasticnet"], "ridge_elasticnet")
    if "xgboost" in family_runs:
        images["overfit_xgb"] = plot_overfitting_curve(family_runs["xgboost"], "xgboost")
    if "torch_mlp" in family_runs:
        images["overfit_mlp"] = plot_overfitting_curve(family_runs["torch_mlp"], "torch_mlp")
    selected_counts = selected_model_table(
        [
            RUNS_ROOT / "regret_first" / "xgboost_chart_tda",
            RUNS_ROOT / "regret_first" / "ridge_elasticnet_base",
            RUNS_ROOT / "top1_first" / "torch_mlp_chart_tda",
        ]
    )
    overfit_summary = selected_overfitting_table(family_runs)

    best_baselines = baseline_table(best_dir)
    best_top1 = float(best["test_top1_hit_rate"])
    best_regret = float(best["test_mean_regret"])
    baseline_payload = json.loads((best_dir / "baselines.json").read_text(encoding="utf-8"))
    train_global_best_regret = float(baseline_payload["train_global_best"]["test"]["mean_regret"])
    train_global_best_top1 = float(baseline_payload["train_global_best"]["test"]["top1_hit_rate"])
    random_regret = float(baseline_payload["random"]["test"]["mean_regret"])
    random_top1 = float(baseline_payload["random"]["test"]["top1_hit_rate"])
    two_way_ties = int((df["tie_count"] == 2).sum())
    three_way_ties = int((df["tie_count"] == 3).sum())
    single_winners = int((df["tie_count"] == 1).sum())
    best_top1_ci = ci_pct(best)
    best_regret_ci = ci_num(best)
    method_summary = methods_table(splits)
    features_summary = feature_dictionary(df)

    css = """
    :root { color-scheme: light; --ink:#172033; --muted:#5b6577; --line:#d9dee7; --bg:#f7f8fb; --card:#ffffff; --blue:#315f9f; --orange:#c96d1b; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); line-height: 1.55; font-size: 17px; }
    header { background: #102033; color: #fff; padding: 44px 7vw 34px; }
    header h1 { margin: 0 0 8px; font-size: clamp(34px, 5vw, 58px); letter-spacing: 0; }
    header p { max-width: 1040px; margin: 8px 0 0; font-size: 20px; color: #dce6f5; }
    main { width: min(1240px, 92vw); margin: 28px auto 70px; }
    section { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 26px; margin: 20px 0; box-shadow: 0 2px 10px rgba(15, 31, 52, 0.04); }
    h2 { margin: 0 0 14px; font-size: 28px; }
    h3 { margin: 0 0 10px; font-size: 21px; }
    p { margin: 10px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }
    .callout { border-left: 5px solid var(--blue); background:#eef4fc; padding: 14px 16px; border-radius: 6px; }
    .metric { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-top: 14px; }
    .metric div { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fbfcff; }
    .metric strong { display:block; font-size:28px; color:var(--blue); }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:12px 0; }
    table.table { border-collapse: collapse; width: 100%; min-width: 760px; font-size: 14px; }
    table.table th, table.table td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
    table.table th { background:#eef1f7; position:sticky; top:0; }
    figure.figure-card { margin:0; border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }
    figure img { width:100%; height:auto; display:block; border-radius:6px; cursor: zoom-in; }
    figcaption { color:var(--muted); font-size:14px; margin-top:8px; }
    ul { margin-top: 8px; }
    li { margin: 7px 0; }
    code { background:#eef1f7; border-radius:4px; padding:2px 5px; }
    .modal { display:none; position:fixed; inset:0; z-index:50; background:rgba(8,15,28,0.88); align-items:center; justify-content:center; padding:28px; }
    .modal.open { display:flex; }
    .modal img { max-width:96vw; max-height:92vh; box-shadow:0 8px 36px rgba(0,0,0,.5); cursor:zoom-out; background:#fff; }
    footer { color:var(--muted); font-size:14px; padding: 0 0 32px; }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SciVer Router Presentation</title>
  <style>{css}</style>
</head>
<body>
<header>
  <h1>SciVer Router Models For Local VLMs</h1>
  <p>Public-review-safe summary of a three-candidate router over Qwen3-VL, Pixtral12B, and Kimi-VL-A3B using cognitive dimensions, chart TDA features, and claim NLP features.</p>
</header>
<main>
  <section>
    <h2>Abstract</h2>
    <p>We study whether a lightweight router can choose among three local vision-language models for SciVer chart-claim verification. The router is trained on sanitized derived features and observed five-run local VLM accuracies, not raw claims, captions, images, prompts, or responses. We compare Ridge/ElasticNet regression, XGBoost, and PyTorch MLP regressors that predict each candidate VLM's accuracy and route to the highest predicted value. The best grouped-holdout point estimate is XGBoost with chart TDA features under mean-regret-first selection: {pct(best_top1)} top-1 hit rate and {num(best_regret, 3)} mean regret. Bootstrap intervals and overfitting curves show the result is promising but still uncertain on an 811-example dataset.</p>
    <div class="metric">
      <div><span>Rows</span><strong>{len(df)}</strong></div>
      <div><span>Best point-estimate top-1</span><strong>{pct(best_top1)}</strong></div>
      <div><span>Best point-estimate regret</span><strong>{num(best_regret, 3)}</strong></div>
      <div><span>Held-out router rows</span><strong>{int((splits["router_split"] == "test").sum())}</strong></div>
    </div>
  </section>

  <section>
    <h2>Read This First</h2>
    <div class="callout">Headline result: XGBoost with chart TDA has the best point estimate on the grouped held-out router split: {pct(best_top1)} top-1 hit rate ({best_top1_ci} bootstrap interval) and {num(best_regret, 3)} mean regret ({best_regret_ci} bootstrap interval). This headline is selected after comparing the public router grid, so the bootstrap interval is conditional on the selected run and does not adjust for model, feature-set, or selection-policy search. Treat it as internal model-development evidence, not an official untouched SciVer benchmark claim.</div>
    <p>The public source labels <code>val</code> and <code>test</code> are original SciVer split-origin labels retained for reporting. Router evaluation uses the separate grouped <code>router_split</code> in each run's <code>splits.csv</code>.</p>
  </section>

  <section>
    <h2>What SciVer Evaluates</h2>
    <p><a href="https://arxiv.org/abs/2506.15569">SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification</a> introduces a benchmark for verifying scientific claims against multimodal evidence from papers, including figures and tables. The released <a href="https://huggingface.co/datasets/chengyewang/SciVer">SciVer dataset</a> is licensed CC BY 4.0. This router study focuses on the chart-claim subset where each example can be represented by chart-derived difficulty features and claim-side semantic features.</p>
    <div class="callout">Public reproducibility here means reproducing router training and evaluation from sanitized derived features. It does not mean rerunning private local VLM inference, exposing raw SciVer claims/captions, or publishing raw model outputs.</div>
  </section>

  <section>
    <h2>Key Definitions</h2>
    <ul>
      <li><strong>Observed five-run accuracy:</strong> for each chart-claim item and VLM, the fraction of five local runs that produced the correct SciVer label. Because each target uses only five repeats, per-item best-model labels and regret values are noisy, especially when VLMs tie.</li>
      <li><strong>Router target:</strong> train one regression target per VLM accuracy, then select the VLM with the highest predicted accuracy.</li>
      <li><strong>Top-1 hit rate:</strong> fraction of held-out examples where the routed VLM is one of the observed best VLMs, counting ties as hits.</li>
      <li><strong>Regret:</strong> observed best accuracy minus the selected VLM's observed accuracy. Lower is better; zero means the router picked a best VLM.</li>
      <li><strong>Train-global-best baseline:</strong> always pick the VLM with the highest mean accuracy on the router training split.</li>
      <li><strong>Grouped 10-fold CV:</strong> split the training paper groups into 10 folds; for each candidate, train on 9 folds, predict the held-out fold, concatenate out-of-fold predictions, and rank candidates by the chosen CV policy. Every row from the same <code>paper_id</code> stays in one fold.</li>
      <li><strong>Grouped bootstrap interval:</strong> after model selection, repeatedly resample held-out test paper groups to estimate uncertainty in the final test top-1 hit rate and mean regret. These intervals capture held-out group sampling uncertainty, not uncertainty from the five-repeat VLM accuracy estimates.</li>
    </ul>
  </section>

  <section>
    <h2>Data Generation And Public Dataset</h2>
    <p>We ran the three local VLMs on analytical/direct single-chart examples, used five repeats per example, and then sanitized the derived router dataset. The public dataset keeps IDs, paper groups, source split, three observed accuracy columns, best-model target columns, twelve cognitive dimensions, engineered scalar features, chart TDA features, and claim NLP features.</p>
    <div class="grid">
      <div class="table-wrap">{table_html(dataset_summary)}</div>
      <div class="table-wrap">{table_html(mean_accuracy)}</div>
    </div>
    <p>Best-model ties are common in this dataset: {single_winners} examples have a single best VLM, {two_way_ties} have two-way ties, and {three_way_ties} have three-way ties. This is why top-1 hit rate is tie-aware rather than ordinary single-label accuracy.</p>
    <div class="grid">
      {chart_block("Observed VLM Accuracy Distributions", images["accuracy"], "Because each item has five local repeats, observed accuracy can only take the six discrete values [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. Each group of bars shows how many chart-claim items fall at that accuracy value for each VLM.")}
      {chart_block("Observed Best VLM Or Tie", images["best"], "Counts show which VLM or tie-set has the highest observed five-run accuracy for an item. Best-model targets are recomputed only over Qwen3-VL, Pixtral12B, and Kimi-VL-A3B.")}
      {chart_block("Feature Families In The Public Dataset", images["features"], "Counts show numeric feature columns used by the router after excluding TDA/NLP status metadata. Cognitive dimensions are human-coded annotations; chart TDA and claim NLP features are precomputed derived features.")}
    </div>
  </section>

  <section>
    <h2>Feature Construction</h2>
    <ul>
      <li><strong>12 cognitive dimensions:</strong> human-coded chart difficulty annotations such as visual load, arithmetic transformation, graphical scale, and knowledge needs. They are available in this public modeling table as precomputed features; a live deployment would need those annotations or an automatic proxy.</li>
      <li><strong>Base scalar features:</strong> claim/caption length, numeric-token counts, percent counts, math-symbol counts, context length, and image dimensions.</li>
      <li><strong>Chart TDA:</strong> persistent homology summaries from grayscale and edge filtrations, designed to capture visual structural complexity.</li>
      <li><strong>Claim NLP:</strong> TF-IDF/SVD semantic coordinates, claim-caption alignment, reasoning cue counts, numeric overlap, and token-point-cloud TDA. The public NLP features were fit once on the full feature-extraction corpus before the router split, so they should be read as transductive exploratory features rather than a strict train-fold-only text pipeline.</li>
    </ul>
    <div class="table-wrap">{table_html(features_summary)}</div>
    <p>The public sidecar feature files can contain more rows than the 811-row modeling table because they are exported before complete-case filtering over the three public VLM accuracy columns. Router training uses the merged complete-case public dataset.</p>
    <div class="callout">Acknowledgment: the 12 cognitive-difficulty dimensions were engineered by the <a href="https://github.com/Erdos-Projects/summer26-ai-faithfulness-in-scientific-reasoning">AI Faithfulness in Scientific Reasoning</a> project. This release attributes that contribution and does not claim to relicense upstream project code or documentation.</div>
  </section>

  <section>
    <h2>Router Workflow</h2>
    <ol>
      <li>Build the matched, sanitized three-VLM router dataset.</li>
      <li>Construct base, chart-TDA, claim-NLP, and combined feature sets.</li>
      <li>Split by <code>paper_id</code> into grouped train/validation/test sets.</li>
      <li>Select hyperparameters by grouped 10-fold CV using either mean-regret-first or top-1-first ranking.</li>
      <li>Refit the selected hyperparameters on train+validation and evaluate once on grouped test.</li>
      <li>Train a final all-data model for future use after evaluation artifacts are written.</li>
    </ol>
    <div class="table-wrap">{table_html(method_summary)}</div>
    <p>Candidate ranking is policy-dependent: <code>regret_first</code> sorts by lower grouped-CV mean regret, then higher top-1 hit rate, then lower complexity and stable candidate ID; <code>top1_first</code> swaps the first two criteria. The held-out test split is not used during candidate ranking inside any run, but the headline “best” run is chosen post-hoc after comparing the public grid of router families, feature sets, and selection policies.</p>
    <p>The grouped CV implementation is deliberately stricter than row-random CV: it first restricts model selection to the router training split, then assigns unique <code>paper_id</code> groups to 10 folds. For each hyperparameter candidate, the script trains on nine paper-group folds, predicts the remaining fold, repeats this for all folds, and computes CV top-1 hit rate and mean regret from the concatenated out-of-fold predictions. This prevents related examples from the same paper from appearing on both sides of a CV fold.</p>
  </section>

  <section>
    <h2>Evaluation Results</h2>
    <p>The strongest point-estimate result in the public artifact tree is <strong>{html.escape(str(best["run_label"]))}</strong>, with {pct(best_top1)} held-out top-1 hit rate and {num(best_regret, 3)} mean regret. The deterministic train-global-best baseline reaches {pct(train_global_best_top1)} top-1 and {num(train_global_best_regret, 3)} regret; the seeded random baseline reaches {pct(random_top1)} top-1 and {num(random_regret, 3)} regret. Baseline comparisons are shown as point estimates; baseline deltas are not separately uncertainty-tested. This keeps deterministic and stochastic baselines separate.</p>
    <div class="grid">
      {chart_block("Regret-First Model Comparison", images["regret"], "Bars show held-out top-1 hit rate and mean regret for all router families/features after hyperparameters are selected by lower grouped-CV mean regret first.")}
      {chart_block("Top-1-First Model Comparison", images["top1"], "Bars show the same held-out metrics after selecting candidates by higher grouped-CV top-1 hit rate first. Changing the selection priority does not displace XGBoost + chart TDA as the best point-estimate run.")}
    </div>
    <h3>Regret-First Metrics</h3>
    <div class="table-wrap">{table_html(compact_metrics_table(regret))}</div>
    <h3>Top-1-First Metrics</h3>
    <div class="table-wrap">{table_html(compact_metrics_table(top1))}</div>
    <h3>Baselines For The Best Run's Split</h3>
    <div class="table-wrap">{table_html(best_baselines)}</div>
    <h3>Selected-Model Counts</h3>
    <div class="table-wrap">{table_html(selected_counts)}</div>
  </section>

  <section>
    <h2>Overfitting Curves</h2>
    <p>Overfitting curves compare training and validation metrics across candidate complexity. A widening train-validation gap, especially high training top-1 paired with lower validation top-1 or low training regret paired with higher validation regret, suggests the model is fitting patterns that do not generalize to held-out paper groups.</p>
    <div class="grid">
      {chart_block("Ridge/ElasticNet Overfitting Curve", images["overfit_ridge"], "Curves show train versus grouped-validation top-1 and mean regret across regularization candidates for the best regret-first Ridge/ElasticNet run. The solid red horizontal line marks the selected candidate. Dotted horizontal separators mark Ridge versus ElasticNet candidate zones when both appear.")}
      {chart_block("XGBoost Overfitting Curve", images["overfit_xgb"], "Curves show train versus grouped-validation metrics across the XGBoost grid. Dotted horizontal separators split tree-count/depth zones such as 50 trees depth 1, 50 trees depth 2, and so on; the right-side labels mark those zones. The solid red line marks the selected candidate.")}
      {chart_block("PyTorch MLP Overfitting Curve", images["overfit_mlp"], "Curves show train versus grouped-validation metrics across the PyTorch MLP grid. Dotted horizontal separators split one-hidden-layer and two-hidden-layer candidates; the solid red line marks the selected candidate.")}
    </div>
    <h3>Selected-Candidate Overfitting Summary</h3>
    <div class="table-wrap">{table_html(overfit_summary)}</div>
    <p>The selected Ridge/ElasticNet and XGBoost candidates show smaller train-validation gaps than the selected PyTorch MLP candidate. This suggests limited-to-moderate overfitting risk for the linear and tree routers, while the MLP has the clearest overfitting signal: it can fit the training split much better than it validates on held-out paper groups. These curves diagnose the model-selection stage; final uncertainty is still judged on the separate grouped test split.</p>
  </section>

  <section>
    <h2>Bootstrap Uncertainty</h2>
    <p>We resample only the held-out router test split because the bootstrap is estimating uncertainty for final reported performance after hyperparameters and model families have already been selected. Resampling training or validation rows would mix model-fitting/model-selection data into the uncertainty estimate. The implementation samples held-out <code>paper_id</code> groups with replacement 1,000 times, recomputes top-1 hit rate and mean regret for each resample, and reports percentile intervals. It does not refit the router and it does not account for five-repeat target noise or post-hoc winner selection across the public grid.</p>
    <div class="grid">
      {chart_block("Ridge/ElasticNet Bootstrap Intervals", images["bootstrap_ridge"], "Each point is a held-out point estimate; whiskers are 95% grouped-bootstrap intervals from 1,000 resamples of test paper groups. The figure includes Ridge/ElasticNet runs across feature sets and both selection policies.")}
      {chart_block("XGBoost Bootstrap Intervals", images["bootstrap_xgb"], "Each point is a held-out point estimate; whiskers are 95% grouped-bootstrap intervals. XGBoost + chart TDA has the best mean-regret point estimate, but intervals overlap with nearby candidates.")}
      {chart_block("PyTorch MLP Bootstrap Intervals", images["bootstrap_mlp"], "Each point is a held-out point estimate; whiskers are 95% grouped-bootstrap intervals. MLP variants show substantial interval overlap, consistent with the small 120-row grouped holdout.")}
      {chart_block("Mean-Regret-First Top Five", images["bootstrap_regret_top5"], "Only the five best point-estimate runs under the mean-regret-first policy are shown, sorted by lower held-out mean regret. Top panel shows tie-aware top-1; bottom panel shows mean regret.")}
      {chart_block("Top-1-First Top Five", images["bootstrap_top1_top5"], "Only the five best point-estimate runs under the top-1-first policy are shown, sorted by higher held-out top-1 hit rate. Intervals are grouped-bootstrap intervals over held-out paper groups.")}
    </div>
  </section>

  <section>
    <h2>Supported Conclusions</h2>
    <ul>
      <li><strong>Chart TDA helps most clearly for XGBoost as a point estimate:</strong> XGBoost + chart TDA yielded the best grouped-holdout point estimate in this run, with mean regret {num(best_regret, 3)} and top-1 hit rate {pct(best_top1)}, but bootstrap intervals overlap with nearby candidates.</li>
      <li><strong>Claim NLP alone does not win in this run:</strong> XGBoost + claim NLP reaches {pct(float(regret.loc[regret["run_label"] == "xgboost_claim_nlp_regret_first", "test_top1_hit_rate"].iloc[0]))} top-1 and {num(float(regret.loc[regret["run_label"] == "xgboost_claim_nlp_regret_first", "test_mean_regret"].iloc[0]), 3)} regret, below chart TDA.</li>
      <li><strong>The router is behaviorally different from always choosing one VLM:</strong> the best router routes held-out items across all three VLMs, while train-global-best always selects Qwen3-VL. This is a routing-diversity observation, not a claim that the performance difference is statistically separated.</li>
      <li><strong>Top-1-first selection is not automatically better:</strong> the best top-1-first run ties the regret-first best point estimate, while several top-1-first NLP variants increase regret.</li>
      <li><strong>Uncertainty remains substantial:</strong> bootstrap intervals are broad on 120 held-out examples and do not include five-repeat target noise or post-hoc grid-selection uncertainty, so conclusions should be presented as internal model-development point estimates.</li>
    </ul>
  </section>

  <section>
    <h2>Reproducibility</h2>
    <p>Primary public data artifact: <code>{DATASET.as_posix()}</code>. Public reviewers can reproduce router modeling from the bundled sanitized CSV without rebuilding private local VLM inference traces.</p>
    <pre><code>python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python scripts/verify_public_router_release.py --root .
.venv/bin/python scripts/run_public_router_experiments.py
.venv/bin/python scripts/build_sciver_router_presentation_report.py
.venv/bin/python scripts/verify_public_router_release.py --root .</code></pre>
    <p>Use <code>--skip-existing</code> only for a quick check of bundled generated artifacts. For a true rerun, start from a fresh public release copy and omit it.</p>
    <p>See <code>docs/reproduce_router_experiments.md</code> and <code>docs/public_data_statement.md</code> for the public/private boundary. Generated <code>.joblib</code> and <code>.pt</code> model files are included as artifacts, but loading serialized model files from untrusted sources can execute code; rerunning scripts is the recommended validation path. For exact-version reproduction, use <code>requirements-lock.txt</code>.</p>
  </section>
</main>
<footer>
  <main>Generated from derived public router artifacts. No raw claims, captions, prompts, responses, images, SQLite annotations, or SciVer gold labels are rendered in this report.</main>
</footer>
<div class="modal" id="image-modal"><img alt="zoomed figure" /></div>
<script>
const modal = document.getElementById('image-modal');
const modalImg = modal.querySelector('img');
document.querySelectorAll('img.zoomable').forEach(img => {{
  img.addEventListener('click', () => {{
    modalImg.src = img.src;
    modal.classList.add('open');
  }});
}});
modal.addEventListener('click', () => {{
  modal.classList.remove('open');
  modalImg.src = '';
}});
document.addEventListener('keydown', event => {{
  if (event.key === 'Escape') {{
    modal.classList.remove('open');
    modalImg.src = '';
  }}
}});
</script>
</body>
</html>
"""


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html_text = "\n".join(line.rstrip() for line in build_html().splitlines()) + "\n"
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
