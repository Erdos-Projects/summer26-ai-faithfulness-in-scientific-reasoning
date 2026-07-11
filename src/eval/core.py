"""Shared cross-validation evaluation core for the CharXiv metric framework.

A single grouped-CV loop drives both the metric notebook and the stress tests. It is splitter-
agnostic (any object with ``.split(df) -> (train_idx, test_idx)``) and preprocessor-agnostic
(``make_pre() -> ColumnTransformer``). Preprocessing is fit **per fold** on the training rows only,
so there is no train/test contamination.

Target convention: ``y = 1`` is **failure** (the positive class), matching the rest of the project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

METRIC_COLS = ["roc_auc", "bal_acc", "f1"]


def fold_predictions(df, make_pre, clf_factory, splitter):
    """Fit per fold and collect held-out predictions.

    Returns a long DataFrame with columns ``fold, pos, y_true, y_prob`` where ``pos`` is the
    positional index into ``df`` (rows may repeat across repeats but not within a fold).
    """
    out = []
    y = df["y"].to_numpy()
    for fold, (tr, te) in enumerate(splitter.split(df)):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        pipe = Pipeline([("pre", make_pre()), ("clf", clf_factory())]).fit(d_tr, y[tr])
        prob = pipe.predict_proba(d_te)[:, 1]
        out.append(pd.DataFrame({"fold": fold, "pos": te, "y_true": y[te], "y_prob": prob}))
    return pd.concat(out, ignore_index=True)


def per_fold_metrics(pred, threshold=0.5):
    """Per-fold metric table from ``fold_predictions`` output."""
    rows = []
    for f, g in pred.groupby("fold"):
        yt, yp = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
        yhat = (yp >= threshold).astype(int)
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:                     # a fold with one class present
            auc = np.nan
        rows.append(dict(fold=f, roc_auc=auc,
                         bal_acc=balanced_accuracy_score(yt, yhat),
                         f1=f1_score(yt, yhat, zero_division=0)))
    return pd.DataFrame(rows)


def summarize(metrics):
    """Mean ± SD (population) across folds -> one row (Series)."""
    agg = {c: metrics[c].mean() for c in METRIC_COLS}
    agg.update({c + "_sd": metrics[c].std(ddof=0) for c in METRIC_COLS})
    return pd.Series(agg)


def evaluate(df, make_pre, clf_factory, splitter, threshold=0.5):
    """Convenience: predictions -> per-fold metrics -> summary. Returns (summary, per_fold, pred)."""
    pred = fold_predictions(df, make_pre, clf_factory, splitter)
    per_fold = per_fold_metrics(pred, threshold=threshold)
    return summarize(per_fold), per_fold, pred


def evaluate_targets(df, make_pre, clf_factory_for, splitter, targets, threshold=0.5):
    """Cross-validate one classifier per target model and stack the KPI summaries.

    ``df`` carries one failure-label column per target (``y__<model>``). For each target the working
    ``y`` is set from its column and :func:`evaluate` runs the grouped CV. ``clf_factory_for`` is a
    callable ``target -> factory`` so different targets can use different estimators (for example the
    random-answer control can stay on a default classifier while the others are tuned).

    Returns a tidy DataFrame with one row per target: ``target`` plus the ``METRIC_COLS`` means and
    their ``_sd`` columns.
    """
    rows = []
    for target in targets:
        d = df.copy()
        d["y"] = d[f"y__{target}"].to_numpy()
        summary, _, _ = evaluate(d, make_pre, clf_factory_for(target), splitter, threshold=threshold)
        rows.append({"target": target, **summary.to_dict()})
    return pd.DataFrame(rows)
