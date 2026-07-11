"""Tests for the CharXiv CV splitter — no-leakage guarantees for the item_holdout regime.

DB-gated (skipped if `annotations_charxiv.db` is absent), mirroring tests/test_pipeline.py.
"""
import numpy as np
import pytest

from src.features import preprocessing as p
from src.splits.splitters import ItemHoldoutSplit


@pytest.fixture(scope="module")
def df():
    if not p.DB_PATH.exists():
        pytest.skip("annotations_charxiv.db not present")
    con = p.connect()
    train_ids, _ = p.get_split(con)
    d = p.make_design_matrix(con, train_ids)
    con.close()
    d["y"] = d[p.ycol(p.TARGET_MODELS[0])].to_numpy()   # splitter stratifies on a target's label
    return d


def _disjoint(tr, te):
    return set(tr).isdisjoint(set(te))


def test_item_holdout_no_paper_straddle(df):
    spl = ItemHoldoutSplit(n_splits=5, n_repeats=2)
    n = 0
    for tr, te in spl.split(df):
        n += 1
        assert _disjoint(tr, te)
        tr_papers = set(df.iloc[tr]["paperid"])
        te_papers = set(df.iloc[te]["paperid"])
        assert tr_papers.isdisjoint(te_papers), "a paper straddles an item-holdout fold"
    assert n == spl.get_n_splits(df) == 10


def test_item_holdout_covers_only_train_items(df):
    """No split touches anything outside the supplied (train) frame."""
    con = p.connect()
    _, test_ids = p.get_split(con)
    con.close()
    assert set(df["item_id"]).isdisjoint(set(test_ids))          # frame is train-only
    for tr, te in ItemHoldoutSplit(n_splits=5, n_repeats=1).split(df):
        idx = np.concatenate([tr, te])
        assert idx.max() < len(df) and idx.min() >= 0
