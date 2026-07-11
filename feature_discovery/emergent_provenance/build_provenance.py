#!/usr/bin/env python3
"""Assemble the emergent-dimensions provenance package.

Reads the discovery-pass artifacts that already live in feature_discovery/
(no annotations_prod.db is present on disk -- everything was already exported
to CSV during the discovery/granularity passes) and derives the four tabular
provenance files. The candidate rubric is copied verbatim by build_provenance.sh.

Run from feature_discovery/:  python3 emergent_provenance/build_provenance.py
"""
import csv
import shutil
import statistics as st
from pathlib import Path

SRC = Path(".")                       # feature_discovery/
OUT = Path("emergent_provenance")
OUT.mkdir(exist_ok=True)

# The nine parent dimensions of the emergent codebook. `tally_depth` in
# dimension_counts.csv is a total-tally column, not a parent; `unassigned`
# in dimension_mapping.csv holds sub-features not yet bound to a parent.
PARENTS = [
    "localization_grounding",
    "value_readout",
    "referent_mismatch",
    "multi_element_reasoning",
    "gestalt_shape",
    "encoding_literacy",
    "legibility_render",
    "claim_linguistic_load",
    "scope_overreach",
]

# ---------------------------------------------------------------------------
# (1) Raw discovery-pass sub-feature list
#     Every sub-feature surfaced by the inductive discovery pass, with its
#     origin (seeded = carried from ADeLe/DeLeAn/Tolan; emergent = newly
#     induced from the 300 items) and the parent it was later folded into.
# ---------------------------------------------------------------------------
mapping = list(csv.DictReader(open(SRC / "dimension_mapping.csv")))
mapping.sort(key=lambda r: (r["origin"], r["parent_dimension"], r["sub_dimension"]))

with open(OUT / "sub_feature_list.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sub_dimension", "origin", "parent_dimension"])
    for r in mapping:
        w.writerow([r["sub_dimension"], r["origin"], r["parent_dimension"]])

n_sub = len(mapping)
n_emergent = sum(1 for r in mapping if r["origin"] == "emergent")
n_seeded = sum(1 for r in mapping if r["origin"] == "seeded")

# ---------------------------------------------------------------------------
# (2) Sub-feature -> parent dimension mapping (9 parents)
#     Verbatim structure of dimension_mapping.csv, re-emitted grouped by
#     parent so the 9-parent consolidation is legible.
# ---------------------------------------------------------------------------
order = {p: i for i, p in enumerate(PARENTS)}
by_parent = sorted(mapping, key=lambda r: (order.get(r["parent_dimension"], 99),
                                           r["sub_dimension"]))
with open(OUT / "sub_feature_to_parent_mapping.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parent_dimension", "sub_dimension", "origin"])
    for r in by_parent:
        w.writerow([r["parent_dimension"], r["sub_dimension"], r["origin"]])

parents_present = sorted({r["parent_dimension"] for r in mapping
                          if r["parent_dimension"] in PARENTS})
n_unassigned = sum(1 for r in mapping if r["parent_dimension"] == "unassigned")

# ---------------------------------------------------------------------------
# (3) Per-parent presence-count mean & std across discovery items
#     Recomputed from the raw per-item tallies in dimension_counts.csv
#     (N = 300 discovery items). Sample std (ddof=1) to match the convention
#     already used in dimension_granularity.csv's sd_count column.
# ---------------------------------------------------------------------------
counts = list(csv.DictReader(open(SRC / "dimension_counts.csv")))
n_items = len(counts)
with open(OUT / "per_parent_presence_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parent_dimension", "n_items", "presence_mean", "presence_std",
                "min", "max", "frac_items_present"])
    for p in PARENTS:
        vals = [int(r[p]) for r in counts]
        present = sum(1 for v in vals if v > 0)
        w.writerow([
            p, n_items,
            f"{st.mean(vals):.3f}",
            f"{st.stdev(vals):.3f}",
            min(vals), max(vals),
            f"{present / n_items:.3f}",
        ])

# ---------------------------------------------------------------------------
# (4) Saturation curve: items annotated vs cumulative unique features
#     saturation_log.csv records the item index at which each new sub-feature
#     first entered the codebook. Cumulative unique-feature count is the
#     running number of first-appearance events; here we emit one row per
#     discovery event (the natural saturation curve) with the running total.
# ---------------------------------------------------------------------------
sat = list(csv.DictReader(open(SRC / "saturation_log.csv")))
sat.sort(key=lambda r: (int(r["item_index"]), r["dimension_name"]))
with open(OUT / "saturation_curve.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["items_annotated", "new_dimension", "cumulative_unique_features"])
    for i, r in enumerate(sat, start=1):
        w.writerow([r["item_index"], r["dimension_name"], i])

n_events = len(sat)
last_item = max(int(r["item_index"]) for r in sat)

# ---------------------------------------------------------------------------
# (5) candidate_rubrics_v0.md -- copied by the shell wrapper (verbatim).
# The source draft is not distributed in the public repository; the committed
# candidate_rubrics_v0.md is the copy of record there.
# ---------------------------------------------------------------------------
rubrics_src = SRC / "Putative_emergent_features_rubrics.md"
if rubrics_src.exists():
    shutil.copyfile(rubrics_src, OUT / "candidate_rubrics_v0.md")

print(f"(1) sub_feature_list.csv           : {n_sub} sub-features "
      f"({n_seeded} seeded, {n_emergent} emergent)")
print(f"(2) sub_feature_to_parent_mapping  : {len(parents_present)} parents "
      f"+ {n_unassigned} unassigned")
print(f"(3) per_parent_presence_stats.csv  : {len(PARENTS)} parents over "
      f"N={n_items} items")
print(f"(4) saturation_curve.csv           : {n_events} discovery events, "
      f"last new feature at item {last_item}")
if rubrics_src.exists():
    print("(5) candidate_rubrics_v0.md         : copied from "
          "Putative_emergent_features_rubrics.md")
else:
    print("(5) candidate_rubrics_v0.md         : source draft not present; "
          "committed copy left as-is")
