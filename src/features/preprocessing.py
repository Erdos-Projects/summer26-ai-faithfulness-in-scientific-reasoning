"""Design-matrix builder + configurable preprocessing pipeline for CharXiv (3 target models).

Reuses the shared loaders in ``charxiv_analysis/mann_whitney_analysis.py`` and the cached 800/200
split, then assembles a tidy per-item feature DataFrame and a scikit-learn ``ColumnTransformer``
that the modeling checkpoints fit inside cross-validation.

Task: predict **failure** (``y=1`` = incorrect) for each of the **3 target models** — GPT-4o,
Claude-3-5-Sonnet, and GPT-4o-Random — with a separate classifier per target. GPT-4o-Random is a
random-answer control that fails most items, so its classifier is expected to sit near chance.

Feature groups:
  - ``category`` / ``inst_category`` / ``year`` (item metadata)   -> one-hot (baseline; OFF by default)
  - the 12 DeLeAn demand dims (ordinal 0-5)                       -> ENGINEERED
        scale (drop CL) and optionally reduce; toggled by ``include_demand_dims``

The 12 DeLeAn dims are the only engineered features. The metadata block is the cheap baseline the
demand dims must beat. All builders use TRAIN items only; the 200-item holdout test set is never
read here.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .transformers import DEMAND_DIMS


def _scaler(name):
    """Numeric scaler by name: 'standard' (default) or 'robust' (for outlier stress tests)."""
    return {"standard": StandardScaler, "robust": RobustScaler}[name]()


# --- repo paths; reuse the charxiv_analysis loaders without packaging them -----------
REPO_ROOT = Path(__file__).resolve().parents[2]
CHARXIV_ANALYSIS = REPO_ROOT / "charxiv_analysis"
DB_PATH = REPO_ROOT / "charxiv_scoring" / "annotations_charxiv.db"
if str(CHARXIV_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(CHARXIV_ANALYSIS))
from mann_whitney_analysis import (  # noqa: E402  shared loaders + cached split
    load_rubric, make_or_load_split, load_correctness,
)

TARGET_MODELS = ["GPT-4o", "Claude-3-5-Sonnet", "GPT-4o-Random"]   # one classifier per target
CATEGORICAL = ["category", "inst_category", "year"]
DEFAULT_DROP_DIMS = ("CL",)   # near-constant (82% zeros); see feature-selection notebook


def ycol(model):
    """Column name holding the failure label for a target model in the design matrix."""
    return f"y__{model}"


TARGET_YCOLS = [ycol(m) for m in TARGET_MODELS]


def connect(db_path=DB_PATH):
    """Open a connection to the CharXiv DB."""
    return sqlite3.connect(str(db_path))


def get_split(con):
    """Return (train_ids, test_ids) from the cached 800/200 split."""
    rubric, _ = load_rubric(con)
    return make_or_load_split(set(rubric))


def make_design_matrix(con, item_ids):
    """Assemble the tidy per-item feature DataFrame for ``item_ids`` (use TRAIN ids).

    One row per item (800 rows on the 800-item train split). Columns: ``item_id``, ``paperid``,
    the 12 demand dims, the metadata columns, one failure label per target (``y__<model>``,
    1=failure), and ``groups`` (paperid). Each target model gets its own classifier fit on the
    shared feature block; only the label column differs.
    """
    item_ids = set(item_ids)
    rubric, _ = load_rubric(con)
    meta = pd.read_sql_query(
        "SELECT item_id, paperid, category, inst_category, year FROM item", con
    ).set_index("item_id")
    correctness = load_correctness(con, item_ids)   # {model: {item_id: 0/1 correct}}

    rows = []
    for item in sorted(item_ids):
        if item not in rubric:
            continue
        if any(item not in correctness.get(m, {}) for m in TARGET_MODELS):
            continue   # keep only items with a valid label for every target
        rec = {"item_id": item, "paperid": meta.at[item, "paperid"]}
        for m in TARGET_MODELS:
            rec[ycol(m)] = 0 if correctness[m][item] else 1   # failure = positive
        for d in DEMAND_DIMS:
            rec[d] = float(rubric[item][d])
        for c in CATEGORICAL:
            rec[c] = meta.at[item, c]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["groups"] = df["paperid"]
    return df


def build_preprocessor(*, drop_dims=DEFAULT_DROP_DIMS, scale=True,
                       scaler="standard", include_demand_dims=True, include_metadata=False):
    """Construct the CharXiv preprocessing ``ColumnTransformer``.

    Parameters
    ----------
    drop_dims : tuple[str]
        Demand dims to exclude (default drops the near-constant ``CL``).
    scale : bool
        Standardize the demand block (matters for linear models; harmless for trees).
    scaler : 'standard' | 'robust'
        Which scaler to use when ``scale`` is True. 'robust' is used by the outlier stress test.
    include_demand_dims : bool
        Include the 12 DeLeAn demand dims (the engineered block). Set False for the pure
        metadata-only baseline in the added-value ablation.
    include_metadata : bool
        One-hot the item metadata (``category``/``inst_category``/``year``) — the cheap baseline.
        **OFF by default** in the final model; the feature-selection ablation found it adds ~0 over
        the demand dims. Kept as a toggle so the baseline configuration can turn it on.
    """
    dims = [d for d in DEMAND_DIMS if d not in set(drop_dims or ())]

    transformers = []

    if include_metadata:
        transformers.append(
            ("meta", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL))

    if include_demand_dims and dims:                                    # engineered block (toggle)
        demand_block = Pipeline([("scale", _scaler(scaler))]) if scale else "passthrough"
        transformers.append(("demand", demand_block, dims))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
