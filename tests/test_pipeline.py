"""Tests for the CharXiv feature pipeline (Checkpoint 2, 3 target models).

Covers the per-item design-matrix construction (one row per item with a failure label per target),
the feature toggles, and the fit-on-train discipline of the preprocessing pipeline. The 12 DeLeAn dims
are the only engineered features; item metadata is the baseline block, off by default. DB-backed tests
skip if the DB is absent.
"""
import numpy as np
import pytest

from src.features import preprocessing as p


@pytest.fixture(scope="module")
def con():
    if not p.DB_PATH.exists():
        pytest.skip("annotations_charxiv.db not present")
    c = p.connect()
    yield c
    c.close()


def test_design_matrix_is_one_row_per_item_with_target_labels(con):
    """One row per item, a 0/1 failure label per target, no model correctness among the features."""
    train_ids, test_ids = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    assert len(df) == len(train_ids)                         # one row per item, not stacked
    for col in p.TARGET_YCOLS:
        assert col in df.columns and df[col].isin([0, 1]).all()
    assert "human_correct" not in df.columns                 # Human is not a feature
    assert "model" not in df.columns                         # no model-identity column
    assert set(df.item_id).isdisjoint(set(test_ids))         # no test items


def test_pipeline_fit_transform_clean(con):
    """Pipeline fit on train then transform yields the expected width and no NaNs."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    pre = p.build_preprocessor()
    Xt = pre.fit_transform(df)
    assert Xt.shape[0] == len(df)
    assert not np.isnan(Xt).any()
    assert len(pre.get_feature_names_out()) == Xt.shape[1]
    assert not any(n.endswith("__CL") for n in pre.get_feature_names_out())   # CL dropped by default


def test_default_pipeline_features(con):
    """Default pipeline = the 11 kept demand dims; no Human, no model identity, no metadata."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    names = list(p.build_preprocessor().fit(df).get_feature_names_out())
    assert all(n.startswith("demand__") for n in names)
    assert not any(n.startswith("meta__") for n in names), "metadata off by default"
    assert not any("human" in n or n.startswith("model__") for n in names), "no Human / model identity"
    assert len(names) == 11                                   # 12 demand dims minus CL


def test_include_demand_dims_toggle(con):
    """include_demand_dims=False with metadata on yields the pure metadata baseline."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    names = list(p.build_preprocessor(include_demand_dims=False, include_metadata=True)
                 .fit(df).get_feature_names_out())
    assert not any(n.startswith("demand__") for n in names)
    assert all(n.startswith("meta__") for n in names) and len(names) > 0


def test_include_metadata_toggle(con):
    """include_metadata=True adds one-hot metadata columns alongside the demand dims."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    names = list(p.build_preprocessor(include_metadata=True).fit(df).get_feature_names_out())
    assert any(n.startswith("meta__") for n in names)
    assert any(n.startswith("demand__") for n in names)


def test_pipeline_handles_unknown_categories(con):
    """A metadata category unseen at fit time transforms without error."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    pre = p.build_preprocessor(include_metadata=True).fit(df)
    unseen = df.head(3).copy()
    unseen["category"] = "zzz"
    Xt = pre.transform(unseen)
    assert Xt.shape == (3, len(pre.get_feature_names_out()))
    assert not np.isnan(Xt).any()


def test_scaler_uses_train_statistics_only(con):
    """StandardScaler centers the demand block on TRAIN means (~zero-mean on train)."""
    train_ids, _ = p.get_split(con)
    df = p.make_design_matrix(con, train_ids)
    pre = p.build_preprocessor().fit(df)
    Xt = pre.transform(df)
    demand_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("demand__")]
    assert np.abs(Xt[:, demand_cols].mean(axis=0)).max() < 1e-6
