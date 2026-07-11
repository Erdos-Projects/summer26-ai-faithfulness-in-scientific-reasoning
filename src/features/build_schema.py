"""Generate ``src/features/schema.json`` from the pipeline + the feature-selection decisions.

Run as a module so the schema is reproducible and cannot drift from the code:

    .venv/bin/python -m src.features.build_schema

It introspects the default preprocessor for the emitted feature names, reads the keep/drop
decisions logged by ``notebooks/13_charxiv_feature_selection.ipynb``
(``results/feature_selection.csv``), and pulls the categorical levels from the DB.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from . import preprocessing as P
from .transformers import DEMAND_DIMS

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema.json"
DECISIONS_CSV = P.REPO_ROOT / "results" / "feature_selection.csv"


def _default_config():
    """Read the live ``build_preprocessor`` defaults so the schema can't drift from the code."""
    params = inspect.signature(P.build_preprocessor).parameters
    cfg = {k: params[k].default for k in
           ["drop_dims", "scale", "scaler", "include_demand_dims", "include_metadata"]
           if k in params}
    cfg["drop_dims"] = list(cfg.get("drop_dims", ()))
    return cfg


def _levels(con):
    q = lambda col: [str(v) for (v,) in con.execute(
        f"SELECT DISTINCT {col} FROM item ORDER BY {col}")]
    return {"category": q("category"), "inst_category": q("inst_category"), "year": q("year")}


def build_schema():
    con = P.connect()
    levels = _levels(con)
    # Emitted feature names from the default pipeline (fit on TRAIN).
    train_ids, _ = P.get_split(con)
    df = P.make_design_matrix(con, train_ids)
    pre = P.build_preprocessor().fit(df)
    emitted = list(map(str, pre.get_feature_names_out()))
    con.close()

    decisions = {}
    if DECISIONS_CSV.exists():
        for _, r in pd.read_csv(DECISIONS_CSV).iterrows():
            decisions[r["feature"]] = {"decision": r["decision"], "reason": r["reason"]}

    def dec(name, default="keep"):
        return decisions.get(name, {"decision": default, "reason": ""})

    inputs = []
    for d in DEMAND_DIMS:                              # the 12 DeLeAn dims: the only engineered features
        inputs.append({"name": d, "role": "demand_dim_engineered", "dtype": "ordinal_int",
                       "range": [0, 5],
                       "transformation": "drop CL; else StandardScaler", **dec(d)})
    for c in ["category", "inst_category", "year"]:    # metadata: baseline candidate, expected dropped
        inputs.append({"name": c, "role": "item_metadata_baseline", "dtype": "categorical",
                       "levels": levels[c], "transformation": "OneHotEncoder (OFF by default)",
                       **dec(c, default="drop")})

    schema = {
        "generated_by": "src/features/build_schema.py (from build_preprocessor + feature_selection.csv)",
        "dataset": "charxiv_scoring/annotations_charxiv.db (reasoning task, TRAIN split)",
        "task": "predict failure per target model (GPT-4o, Claude-3-5-Sonnet, GPT-4o-Random); "
                "one classifier per target",
        "target": {"name": "y__<model>", "positive_class": "failure (incorrect = 1)"},
        "group_key": "paperid",
        "unit_of_analysis": "one row per item (800 train items); three per-target failure labels",
        "default_config": _default_config(),
        "inputs": inputs,
        "emitted_features_default": emitted,
        "n_emitted_features_default": len(emitted),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"wrote {SCHEMA_PATH} ({len(inputs)} inputs, {len(emitted)} emitted features)")
    return schema


if __name__ == "__main__":
    build_schema()
