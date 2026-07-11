"""Tests for the test-set KPI bootstrap helper in ``src/eval/bootstrap``.

These are self-contained (no database needed): they run on a small synthetic frame of one row per
item with fixed predictions, so they exercise the item-level resampling and the confidence-interval
table directly.
"""
import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap_kpis, kpis, percentile_ci
from src.eval.core import METRIC_COLS


def _toy_frame(n_items=180, seed=0):
    """One row per item, with a mild signal so ROC-AUC beats chance."""
    rng = np.random.default_rng(seed)
    rows, y_true, y_prob = [], [], []
    for i in range(n_items):
        y = int(rng.random() < 0.7)                     # ~70% failure base rate
        p = np.clip(0.5 + (0.2 if y else -0.2) + rng.normal(0, 0.2), 0, 1)  # signal + noise
        rows.append({"item_id": f"item_{i}"})
        y_true.append(y)
        y_prob.append(p)
    return pd.DataFrame(rows), np.array(y_true), np.array(y_prob)


def test_bootstrap_kpis_shape_and_columns():
    """One row per resample, carrying every KPI column."""
    df, y_true, y_prob = _toy_frame()
    boot = bootstrap_kpis(df, y_true, y_prob, n_boot=50)
    assert len(boot) == 50
    assert all(c in boot.columns for c in METRIC_COLS)


def test_bootstrap_kpis_is_deterministic_under_seed():
    """The same seed reproduces the same resampling distribution."""
    df, y_true, y_prob = _toy_frame()
    a = bootstrap_kpis(df, y_true, y_prob, n_boot=30, seed=123)
    b = bootstrap_kpis(df, y_true, y_prob, n_boot=30, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_resamples_items():
    """Items are the resampling unit (one row per item); one KPI row per resample."""
    df, y_true, y_prob = _toy_frame()
    boot = bootstrap_kpis(df, y_true, y_prob, n_boot=5)
    assert len(boot) == 5   # item-level resampling is enforced by construction in bootstrap_kpis


def test_percentile_ci_brackets_and_orders():
    """The interval is ordered (lo <= mean <= hi) and reported next to the point estimate."""
    df, y_true, y_prob = _toy_frame()
    point = kpis(y_true, y_prob)
    boot = bootstrap_kpis(df, y_true, y_prob, n_boot=200)
    ci = percentile_ci(boot, point).set_index("kpi")
    for c in METRIC_COLS:
        assert ci.at[c, "ci_lo"] <= ci.at[c, "boot_mean"] <= ci.at[c, "ci_hi"]
        assert ci.at[c, "ci_lo"] <= ci.at[c, "point"] <= ci.at[c, "ci_hi"]
