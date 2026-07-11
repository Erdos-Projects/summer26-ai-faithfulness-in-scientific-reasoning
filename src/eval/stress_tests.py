"""Robustness and leakage stress tests for the CharXiv evaluation framework.

Each function returns a tidy DataFrame and (optionally) writes it under ``results/stress_tests/``
with the split strategy + date in the filename. They reuse the shared CV core in
:mod:`src.eval.core` and the preprocessing pipeline in :mod:`src.features.preprocessing`.

Each test operates on a single target's failure label in ``df["y"]`` (set it from the matching
``y__<model>`` column before calling), so a classifier per target reuses the same code.

Tests provided:
- :func:`shuffled_target_test`  — adversarial leakage / overfit check (AUC must fall to ~0.5).
- :func:`dropped_feature_test`  — where does the signal live (demand dims vs metadata baseline)?
- :func:`cohort_eval`           — per-cohort performance (distribution-shift / subgroup robustness).
- :func:`outlier_sensitivity`   — StandardScaler vs RobustScaler, and excluding demand outliers.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from src.features.preprocessing import REPO_ROOT, build_preprocessor
from src.features.transformers import DEMAND_DIMS
from .core import evaluate, fold_predictions, per_fold_metrics, summarize

STRESS_DIR = REPO_ROOT / "results" / "stress_tests"
TODAY = date.today().isoformat()


def _save(df, name, strategy):
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    path = STRESS_DIR / f"{name}__{strategy}__{TODAY}.csv"
    df.to_csv(path, index=False)
    return path


def shuffled_target_test(df, make_pre, clf_factory, splitter, strategy, n_repeats=3,
                         seed=0, save=True):
    """Permute the target and re-evaluate. Real AUC should beat shuffled; shuffled should be ~0.50.

    If the shuffled runs score well above 0.50, there is leakage or the split is being overfit.
    """
    rng = np.random.default_rng(seed)
    real, _, _ = evaluate(df, make_pre, clf_factory, splitter)
    rows = [{"setting": "real", "roc_auc": real["roc_auc"], "roc_auc_sd": real["roc_auc_sd"]}]
    for r in range(n_repeats):
        d = df.copy()
        d["y"] = rng.permutation(d["y"].to_numpy())
        s, _, _ = evaluate(d, make_pre, clf_factory, splitter)
        rows.append({"setting": f"shuffled_{r}", "roc_auc": s["roc_auc"], "roc_auc_sd": s["roc_auc_sd"]})
    out = pd.DataFrame(rows)
    out["passes_leakage_check"] = bool(
        out.loc[out.setting.str.startswith("shuffled"), "roc_auc"].max() < 0.55
        <= out.loc[out.setting == "real", "roc_auc"].iloc[0])
    if save:
        _save(out, "shuffled_target", strategy)
    return out


def dropped_feature_test(df, configs, clf_factory, splitter, strategy, save=True):
    """Re-evaluate under several feature configs. ``configs`` maps name -> make_pre factory."""
    rows = []
    for name, make_pre in configs.items():
        s, _, _ = evaluate(df, make_pre, clf_factory, splitter)
        rows.append({"config": name, "roc_auc": s["roc_auc"], "roc_auc_sd": s["roc_auc_sd"]})
    out = pd.DataFrame(rows)
    if save:
        _save(out, "dropped_feature", strategy)
    return out


def cohort_eval(pred, df, by, strategy, save=True):
    """Slice held-out predictions by a cohort column and report per-cohort ROC-AUC / failure rate.

    ``pred`` is the output of :func:`core.fold_predictions`; ``by`` is a column of ``df`` (used for
    *slicing only*, never as a feature) — e.g. ``category``, ``inst_category``, or ``year``.
    """
    p = pred.copy()
    p[by] = df.iloc[p["pos"].to_numpy()][by].to_numpy()
    rows = []
    for val, g in p.groupby(by, observed=True):
        yt, yp = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:
            auc = np.nan
        rows.append({by: val, "n": len(g), "failure_rate": round(float(yt.mean()), 3),
                     "roc_auc": auc})
    out = pd.DataFrame(rows).sort_values("roc_auc").reset_index(drop=True)
    if save:
        _save(out, f"cohort_{by}", strategy)
    return out


def outlier_sensitivity(df, clf_factory, splitter, strategy, contamination=0.03, save=True):
    """Does performance hinge on the scaler or on atypical-demand items?

    Compares (a) StandardScaler, (b) RobustScaler, and (c) excluding IsolationForest demand-outlier
    *items* entirely, using the default pipeline (the demand dims).
    """
    std_pre = lambda: build_preprocessor(scaler="standard")
    rob_pre = lambda: build_preprocessor(scaler="robust")

    s_std, _, _ = evaluate(df, std_pre, clf_factory, splitter)
    s_rob, _, _ = evaluate(df, rob_pre, clf_factory, splitter)

    # Flag atypical demand profiles at the item level, then drop those items' rows.
    items = df.drop_duplicates("item_id").set_index("item_id")[DEMAND_DIMS]
    iso = IsolationForest(contamination=contamination, random_state=0).fit(items)
    outlier_items = set(items.index[iso.predict(items) == -1])
    kept = df[~df["item_id"].isin(outlier_items)].reset_index(drop=True)
    s_drop, _, _ = evaluate(kept, std_pre, clf_factory, splitter)

    out = pd.DataFrame([
        {"setting": "StandardScaler", "roc_auc": s_std["roc_auc"], "n_rows": len(df)},
        {"setting": "RobustScaler", "roc_auc": s_rob["roc_auc"], "n_rows": len(df)},
        {"setting": f"exclude_outlier_items ({len(outlier_items)} items)",
         "roc_auc": s_drop["roc_auc"], "n_rows": len(kept)},
    ])
    if save:
        _save(out, "outlier_sensitivity", strategy)
    return out
