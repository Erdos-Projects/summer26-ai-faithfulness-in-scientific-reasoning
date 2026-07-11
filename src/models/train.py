"""Fit and serialize the final CharXiv failure-prediction pipelines.

A thin wrapper over the shared preprocessing + estimator contract. ``fit_final`` fits the full
``Pipeline([("pre", make_pre()), ("clf", clf_factory())])`` on all supplied rows (always the 800-item
train split — never the sealed 200-item test set). ``fit_targets`` fits one such pipeline per target
model and returns a ``{target: pipeline}`` dict — the project trains a separate classifier per
target. ``save`` / ``load`` round-trip the fitted object (a pipeline or the dict) through ``joblib``.
"""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def fit_final(df, make_pre, clf_factory):
    """Fit the full pipeline on ``df`` (train rows) and return the fitted estimator.

    ``make_pre`` is a preprocessor factory (e.g. from ``build_preprocessor``); ``clf_factory`` returns
    a fresh classifier. Fitting the preprocessor here is safe because ``df`` is train-only.
    """
    pipe = Pipeline([("pre", make_pre()), ("clf", clf_factory())])
    return pipe.fit(df, df["y"].to_numpy())


def fit_targets(df, make_pre, clf_factory_for, targets):
    """Fit one pipeline per target and return ``{target: fitted_pipeline}``.

    ``clf_factory_for`` is a callable ``target -> factory`` so each target can use its own estimator.
    The working label is taken from ``df[f"y__{target}"]``; the feature block is shared.
    """
    models = {}
    for target in targets:
        d = df.copy()
        d["y"] = d[f"y__{target}"].to_numpy()
        models[target] = fit_final(d, make_pre, clf_factory_for(target))
    return models


def save(pipe, path):
    """Serialize a fitted pipeline with ``joblib.dump``; creates the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load(path):
    """Reload a pipeline serialized by :func:`save`."""
    return joblib.load(Path(path))
