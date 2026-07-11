"""Nested, paper-grouped cross-validation for honest hyperparameter tuning.

This is the Checkpoint-4 adaptation of the recipe in ``charxiv_analysis/tuned_model_comparison.py``.
Hyperparameters are chosen in an inner paper-grouped loop and the receipt-of-truth ROC-AUC is measured
on an outer fold the tuner never saw, so there is no tuning-on-test leakage. The data API is the
Checkpoint-2 design-matrix DataFrame plus a ``make_pre`` preprocessing factory rather than a raw
feature matrix, so every transformer is still fit per fold inside the pipeline.

To avoid the nested-parallelism trap the estimators run single-threaded (their own ``n_jobs=1``) and
``GridSearchCV`` parallelizes the search (``n_jobs=-1``).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def _inner_folds(sub, inner_k, seed=0):
    """Explicit paper-grouped inner folds as ``(train_idx, test_idx)`` positions into ``sub``.

    Passing a precomputed fold list as ``cv`` sidesteps threading ``groups`` through the pipeline while
    keeping the inner split paper-grouped (no ``paperid`` straddles an inner fold).
    """
    y = sub["y"].to_numpy()
    groups = sub["paperid"].to_numpy()
    sgk = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    return list(sgk.split(np.zeros(len(sub)), y, groups))


def nested_cv_auc(df, make_pre, estimator, grid, splitter, inner_k=3, seed=0):
    """Nested outer-fold ROC-AUC for one estimator + grid.

    For each outer ``(tr, te)`` from ``splitter.split(df)``, an inner ``GridSearchCV`` (paper-grouped,
    ``inner_k`` folds) selects hyperparameters on the outer-train rows only; the refit best estimator
    is scored on the untouched outer-test rows. ``grid`` uses ``clf__``-prefixed keys.

    Returns a dict with the per-fold AUCs, their mean and population SD, the modal best parameters, and
    a Counter of how often each parameter set won.
    """
    y = df["y"].to_numpy()
    aucs, best_counter = [], Counter()
    for tr, te in splitter.split(df):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        pipe = Pipeline([("pre", make_pre()), ("clf", estimator)])
        inner = _inner_folds(d_tr.reset_index(drop=True), inner_k, seed=seed)
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        gs.fit(d_tr.reset_index(drop=True), y[tr])
        prob = gs.best_estimator_.predict_proba(d_te)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        best_counter[str(gs.best_params_)] += 1
    aucs = np.array(aucs, dtype=float)
    modal = best_counter.most_common(1)[0][0] if best_counter else "{}"
    return {
        "outer_aucs": aucs,
        "roc_auc": float(aucs.mean()),
        "roc_auc_sd": float(aucs.std(ddof=0)),
        "modal_best_params": modal,
        "best_param_counts": best_counter,
    }


def modal_params_dict(nested_result):
    """Parse the ``modal_best_params`` string back into a dict of ``clf__*`` -> value."""
    import ast
    return ast.literal_eval(nested_result["modal_best_params"])


def tune_table(configs, df, make_pre, splitter, inner_k=3, seed=0, extra_cols=None):
    """Run :func:`nested_cv_auc` for several ``configs`` and return a tidy DataFrame.

    ``configs`` maps a family name to ``(estimator, grid)``. ``extra_cols`` is an optional dict of
    constant columns (for example ``{"config": "tuned"}``) added to every row.
    """
    rows = []
    for name, (estimator, grid) in configs.items():
        res = nested_cv_auc(df, make_pre, estimator, grid, splitter, inner_k=inner_k, seed=seed)
        row = {"family": name, "roc_auc": res["roc_auc"], "roc_auc_sd": res["roc_auc_sd"],
               "modal_best_params": res["modal_best_params"]}
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return pd.DataFrame(rows)
