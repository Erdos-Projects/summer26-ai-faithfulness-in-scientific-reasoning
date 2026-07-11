"""Model registry for the CharXiv failure-prediction checkpoint.

A single source of truth for the model families and their nested-cross-validation tuning grids. Each
family is a factory that returns a fresh, unfitted estimator; the estimator becomes the ``clf`` step
of ``Pipeline([("pre", make_pre()), ("clf", ...)])`` in :mod:`src.eval.core` and :mod:`src.models.tune`.

Two groups of families are defined.
  - Trivial and simple baselines (Dummy, DecisionTree, default LogisticRegression, default
    RandomForest) — the reference points reported in the baselines notebook.
  - Tunable families (regularized LogisticRegression, RandomForest, HistGradientBoosting, XGBoost) —
    each paired with a ``clf__``-prefixed grid in :data:`GRIDS`, mirroring the honest nested-CV recipe
    in ``charxiv_analysis/tuned_model_comparison.py``.

XGBoost is imported lazily and guarded so this module still imports in the plain ``.venv`` (which has
no XGBoost); :func:`has_xgboost` reports whether the family is available.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEED = 0

try:                                    # optional: only present in the charxiv-model conda env
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:                     # keep the module importable under the plain .venv
    XGBClassifier = None
    _HAS_XGBOOST = False


def has_xgboost() -> bool:
    """Whether the XGBoost family is importable in the current environment."""
    return _HAS_XGBOOST


# --- trivial and simple baselines ----------------------------------------------------
def dummy_most_frequent():
    return DummyClassifier(strategy="most_frequent")


def dummy_stratified():
    return DummyClassifier(strategy="stratified", random_state=SEED)


def logistic():
    """Default L2 LogisticRegression — the linear reference (matches the Checkpoint-3 baseline)."""
    return LogisticRegression(max_iter=2000, random_state=SEED)


def random_forest():
    """RandomForest reference — matches the Checkpoint 3 baseline (300 trees, min_samples_leaf 5)."""
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  n_jobs=1, random_state=SEED)


# --- tunable families ----------------------------------------------------------------
def logistic_regularized():
    """LogisticRegression with the ``saga`` solver so the grid can pick L2, L1, or elastic net."""
    return LogisticRegression(solver="saga", max_iter=5000, random_state=SEED)


def hist_gb():
    return HistGradientBoostingClassifier(random_state=SEED)


def xgboost():
    """XGBoost classifier (requires the charxiv-model conda env)."""
    if not _HAS_XGBOOST:
        raise ImportError("xgboost is not installed in this environment (use the charxiv-model env).")
    return XGBClassifier(
        n_jobs=1, random_state=SEED, eval_metric="logloss",
        tree_method="hist", verbosity=0,
    )


# Factory lookup for the reusable evaluation code.
FACTORIES = {
    "dummy_most_frequent": dummy_most_frequent,
    "dummy_stratified": dummy_stratified,
    "logistic": logistic,
    "random_forest": random_forest,
    "logistic_regularized": logistic_regularized,
    "hist_gb": hist_gb,
    "xgboost": xgboost,
}

# Trivial-to-simple reference families reported in the baselines notebook.
BASELINE_FAMILIES = ["dummy_most_frequent", "dummy_stratified", "logistic", "random_forest"]

# Families carried into the tuned experiments notebook.
TUNED_FAMILIES = ["logistic_regularized", "random_forest", "hist_gb", "xgboost"]


# --- nested-CV tuning grids (``clf__``-prefixed to target the pipeline's classifier step) --------
# Grids are deliberately small: this is light tuning meant to run quickly on a modest CPU. The
# winning family gets a more thorough search (and bootstrap validation) in a later checkpoint.
_C_GRID = [0.1, 1.0, 10.0]     # regularization strength; a short log grid keeps the saga fits cheap

# Ridge (L2), Lasso (L1), or elastic net — the best regularizer is chosen inside CV, not by hand.
# The saga solver was slow and finicky before, so the grid stays small (one l1_ratio for elastic net).
LOGISTIC_REGULARIZED_GRID = [
    {"clf__penalty": ["l2"], "clf__C": _C_GRID},
    {"clf__penalty": ["l1"], "clf__C": _C_GRID},
    {"clf__penalty": ["elasticnet"], "clf__C": _C_GRID, "clf__l1_ratio": [0.5]},
]

RANDOM_FOREST_GRID = {
    "clf__min_samples_leaf": [5, 20, 50],
    "clf__max_depth": [2, 4, 8],
}

HIST_GB_GRID = {
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_leaf_nodes": [15, 31],
    "clf__l2_regularization": [0.0, 1.0],
}

XGBOOST_GRID = {
    "clf__n_estimators": [300],
    "clf__max_depth": [2, 4, 8],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8],
    "clf__reg_lambda": [1.0],
}

GRIDS = {
    "logistic_regularized": LOGISTIC_REGULARIZED_GRID,
    "random_forest": RANDOM_FOREST_GRID,
    "hist_gb": HIST_GB_GRID,
    "xgboost": XGBOOST_GRID,
}
