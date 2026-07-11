"""Interpretability diagnostics for the CharXiv failure-prediction models.

Three views, all keyed to the preprocessed feature names emitted by the pipeline's
``ColumnTransformer``:
  - :func:`logistic_coefficients` — signed coefficients of a fitted linear model (direction and size
    of each feature's contribution to the log-odds of failure).
  - :func:`permutation_importance_table` — model-agnostic permutation importance on a held-out frame,
    scored by ROC-AUC (the primary KPI), so it answers "how much does the metric drop if this feature
    is scrambled".
  - :func:`shap_summary` — a SHAP summary plot (requires the charxiv-model conda env).

Passing held-out rows to the permutation and SHAP functions keeps the read honest; in-sample
importance is optimistic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.preprocessing import CATEGORICAL
from src.features.transformers import DEMAND_DIMS

# Columns the preprocessor may consume; passed to permutation importance so only meaningful
# features are scrambled (item_id / paperid / the y__<model> labels / groups are ignored).
FEATURE_COLS = DEMAND_DIMS + CATEGORICAL


def feature_names(pipe):
    """Transformed feature names emitted by the fitted pipeline's preprocessor."""
    return list(pipe.named_steps["pre"].get_feature_names_out())


def logistic_coefficients(pipe):
    """Signed coefficients of a fitted linear ``clf`` step, sorted by absolute magnitude.

    Returns a DataFrame with columns ``feature``, ``coef``, ``abs_coef``.
    """
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise TypeError("logistic_coefficients expects a linear model with a coef_ attribute.")
    coef = np.asarray(clf.coef_).ravel()
    out = pd.DataFrame({"feature": feature_names(pipe), "coef": coef})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def permutation_importance_table(pipe, df_eval, n_repeats=20, scoring="roc_auc", seed=0):
    """Permutation importance on ``df_eval`` (ideally held-out rows), scored by ``scoring``.

    Only the model-relevant columns present in ``df_eval`` are permuted. Returns a DataFrame with
    ``feature`` (raw input column), ``importance_mean``, ``importance_sd``, sorted descending.
    """
    cols = [c for c in FEATURE_COLS if c in df_eval.columns]
    X = df_eval[cols]
    y = df_eval["y"].to_numpy()
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": cols, "importance_mean": r.importances_mean,
                        "importance_sd": r.importances_std})
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_summary(pipe, df_eval, out_png, max_display=15):
    """Write a SHAP summary plot for a fitted pipeline on ``df_eval``. Requires the shap package.

    The preprocessor transforms the frame, then a SHAP explainer runs on the fitted classifier over
    the transformed matrix. Returns the path written.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X = pre.transform(df_eval)
    names = feature_names(pipe)

    # TreeExplainer is preferred for the boosted / forest models this is called on. Some XGBoost /
    # shap version pairs cannot parse the tree dump, so fall back to the model-agnostic explainer on
    # the positive-class probability (slower, but correct and single-output).
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
    except Exception:
        f = lambda data: clf.predict_proba(data)[:, 1]
        explainer = shap.Explainer(f, X, feature_names=names)
        shap_values = explainer(X, silent=True)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=np.array(names), max_display=max_display,
                      show=False)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png
