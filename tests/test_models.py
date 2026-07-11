"""Tests for the Checkpoint-4 modeling code in ``src/models``.

Covers the model registry factories, the final-fit and serialization wrapper, the nested
cross-validation tuner, and the interpretability helpers. DB-backed tests skip if the DB is absent,
mirroring ``tests/test_pipeline.py`` and ``tests/test_splits.py``. The XGBoost family is only checked
when the package is importable, so the suite still passes in the plain ``.venv``.
"""
import numpy as np
import pytest

from src.features import preprocessing as p
from src.eval.core import METRIC_COLS, evaluate
from src.splits.splitters import ItemHoldoutSplit
from src.models import registry, tune, train, interpret


@pytest.fixture(scope="module")
def df():
    if not p.DB_PATH.exists():
        pytest.skip("annotations_charxiv.db not present")
    con = p.connect()
    train_ids, _ = p.get_split(con)
    d = p.make_design_matrix(con, train_ids)
    con.close()
    d["y"] = d[p.ycol(p.TARGET_MODELS[0])].to_numpy()   # the modeling code predicts a single target's y
    return d


def _make_pre():
    return p.build_preprocessor()


def test_every_factory_builds_an_estimator():
    """Each registered family (except XGBoost when absent) builds a fresh estimator."""
    for name, factory in registry.FACTORIES.items():
        if name == "xgboost" and not registry.has_xgboost():
            continue
        est = factory()
        assert hasattr(est, "fit"), f"{name} did not build a fittable estimator"


def test_grids_cover_the_tuned_families():
    """Every tuned family has a grid; every grid key targets the pipeline's clf step."""
    for name in registry.TUNED_FAMILIES:
        assert name in registry.GRIDS, f"{name} has no tuning grid"
    for grid in registry.GRIDS.values():
        blocks = grid if isinstance(grid, list) else [grid]
        for block in blocks:
            assert all(k.startswith("clf__") for k in block), "grid keys must be clf__-prefixed"


def test_fit_final_predicts_proba(df):
    """fit_final returns a fitted pipeline whose predict_proba has one column per class."""
    sub = df.head(300)
    pipe = train.fit_final(sub, _make_pre, registry.logistic)
    proba = pipe.predict_proba(sub)
    assert proba.shape == (len(sub), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_and_load_roundtrip(df, tmp_path):
    """A serialized pipeline reloads and reproduces its probabilities."""
    sub = df.head(300)
    pipe = train.fit_final(sub, _make_pre, registry.logistic)
    path = train.save(pipe, tmp_path / "model.joblib")
    reloaded = train.load(path)
    assert np.allclose(pipe.predict_proba(sub)[:, 1], reloaded.predict_proba(sub)[:, 1])


def test_nested_cv_auc_returns_one_auc_per_outer_fold(df):
    """Nested CV yields one AUC per outer fold, each a sensible probability, plus modal params."""
    spl = ItemHoldoutSplit(n_splits=3, n_repeats=1)
    grid = [{"clf__penalty": ["l2"], "clf__C": [1.0]}]     # trivial grid keeps the test fast
    res = tune.nested_cv_auc(df, _make_pre, registry.logistic_regularized(), grid, spl, inner_k=3)
    assert len(res["outer_aucs"]) == spl.get_n_splits(df)
    assert (res["outer_aucs"] > 0.4).all() and (res["outer_aucs"] < 1.0).all()
    assert res["modal_best_params"] == "{'clf__C': 1.0, 'clf__penalty': 'l2'}"


def test_evaluate_returns_all_metric_cols(df):
    """The shared evaluator returns every KPI column for a registry estimator."""
    spl = ItemHoldoutSplit(n_splits=3, n_repeats=1)
    summary, _, _ = evaluate(df, _make_pre, registry.logistic, spl)
    assert all(c in summary.index for c in METRIC_COLS)
    assert 0.4 < summary["roc_auc"] < 1.0


def test_interpret_helpers(df):
    """Coefficient and permutation-importance tables cover the transformed / input features."""
    sub = df.head(400)
    pipe = train.fit_final(sub, _make_pre, registry.logistic)
    coefs = interpret.logistic_coefficients(pipe)
    assert len(coefs) == len(interpret.feature_names(pipe))
    assert any(f.startswith("demand__") for f in coefs.feature)
    pi = interpret.permutation_importance_table(pipe, sub, n_repeats=2)
    assert "MCu" in set(pi.feature)
    assert {"importance_mean", "importance_sd"}.issubset(pi.columns)
