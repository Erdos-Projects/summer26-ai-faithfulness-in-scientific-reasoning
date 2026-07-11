"""Item-level bootstrap of the held-out test KPIs for a fixed model.

The sealed 200-item test set is small, so a single point estimate of each key performance indicator
carries real sampling uncertainty. This helper quantifies that uncertainty without refitting the
model: it takes one target model's already-computed predictions on the test items, resamples the
distinct test items with replacement, and recomputes ROC-AUC, balanced accuracy, and F1 on each
resample — the same item-level convention used by the training bootstrap in
``notebooks/18_charxiv_bootstrap_stability.ipynb``.

Because the functions take predictions rather than an estimator, the same code serves each target
model's test evaluation. The KPI definitions are imported from :mod:`src.eval.core` so there is a
single source of truth for the metric set.

Target convention: ``y = 1`` is **failure** (the positive class), matching the rest of the project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .core import METRIC_COLS


def kpis(y_true, y_prob, threshold=0.5):
    """Point-estimate KPIs for one set of predictions.

    Returns a dict over :data:`src.eval.core.METRIC_COLS` (``roc_auc``, ``bal_acc``, ``f1``). ROC-AUC is
    ``nan`` if only one class is present (an undefined single-class resample).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_hat = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:                     # only one class present in this sample
        auc = np.nan
    return {"roc_auc": auc,
            "bal_acc": balanced_accuracy_score(y_true, y_hat),
            "f1": f1_score(y_true, y_hat, zero_division=0)}


def bootstrap_kpis(df, y_true, y_prob, n_boot=100, seed=20260618, threshold=0.5):
    """Item-level bootstrap of the KPIs for fixed predictions.

    Parameters
    ----------
    df : DataFrame
        The evaluated rows, aligned position-for-position with ``y_true`` and ``y_prob``. Must carry an
        ``item_id`` column; items are the resampling unit (one row per item).
    y_true, y_prob : array-like
        The true 0/1 failure labels and the model's predicted P(failure) for those same rows.
    n_boot : int
        Number of bootstrap resamples (default 100 — each resample only recomputes metrics on fixed
        predictions, so this is cheap).
    seed : int
        Base seed; resample ``b`` uses ``seed + b`` (same seed lineage as the fixed 800/200 split).

    Returns
    -------
    DataFrame
        One row per resample with columns ``sample_idx`` and the three KPI columns.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if not (len(df) == len(y_true) == len(y_prob)):
        raise ValueError("df, y_true, and y_prob must be the same length and row-aligned.")

    # Positional row indices grouped by item (one row per item here), so items are resampled as units.
    pos_by_item = pd.Series(np.arange(len(df)), index=df["item_id"].to_numpy())
    item_rows = {item: g.to_numpy() for item, g in pos_by_item.groupby(level=0)}
    unique_items = np.array(sorted(item_rows))

    rows = []
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        drawn = rng.choice(unique_items, size=len(unique_items), replace=True)
        pos = np.concatenate([item_rows[i] for i in drawn])
        rec = {"sample_idx": b}
        rec.update(kpis(y_true[pos], y_prob[pos], threshold=threshold))
        rows.append(rec)
    return pd.DataFrame(rows)


def percentile_ci(boot_df, point, alpha=0.05):
    """95 percent (by default) percentile confidence intervals from a bootstrap KPI table.

    Parameters
    ----------
    boot_df : DataFrame
        Output of :func:`bootstrap_kpis`.
    point : dict
        The single-test-set point estimate per KPI (for example from :func:`kpis`), reported alongside
        each interval so the notebook can compare the point value to the resampling distribution.
    alpha : float
        Two-sided miscoverage; ``0.05`` gives the 2.5th and 97.5th percentiles.

    Returns
    -------
    DataFrame
        One row per KPI with ``point``, ``boot_mean``, ``boot_sd``, ``ci_lo``, ``ci_hi``.
    """
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    rows = []
    for c in METRIC_COLS:
        vals = boot_df[c].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [lo_q, hi_q])
        rows.append({"kpi": c,
                     "point": round(float(point[c]), 4),
                     "boot_mean": round(float(vals.mean()), 4),
                     "boot_sd": round(float(vals.std(ddof=1)), 4),
                     "ci_lo": round(float(lo), 4),
                     "ci_hi": round(float(hi), 4)})
    return pd.DataFrame(rows)
