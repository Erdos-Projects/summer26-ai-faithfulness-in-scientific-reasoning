# Emergent Dimensions — Provenance Package

This folder traces the **emergent figure-verification demand dimensions** from the
raw inductive discovery pass through to the consolidated 9-parent codebook and the
first candidate scoring rubrics. It exists so the provenance chain

> raw discovered sub-features → parent consolidation → per-parent load → saturation evidence → candidate rubric

can be audited end to end without re-running the discovery pass.

**Source of record.** The discovery pass ran over **300 claim+figure items**
(200 SciVer single-figure-evidence + 100 SciClaimEval figure-evidence). No
`annotations_prod.db` is present in this checkout — the annotation tallies were
already exported to CSV during the discovery and granularity passes, so every file
here is derived from those CSVs (see *Regenerating* below), not from a database.

## Files

| # | File | What it is | What it demonstrates |
|---|------|-----------|----------------------|
| 1 | `sub_feature_list.csv` | The **raw discovery-pass sub-feature list**: all 112 sub-dimensions (23 seeded, 89 emergent), each tagged `origin` and the parent it was folded into. | The full inductive yield of the discovery pass, before consolidation — every distinct demand the annotators named. |
| 2 | `sub_feature_to_parent_mapping.csv` | The **sub-feature → parent mapping**, grouped by parent. 108 sub-features map onto the **9 parent dimensions**; 4 remain `unassigned`. | How 112 fine-grained sub-features collapse to the 9-dimension consolidated codebook. |
| 3 | `per_parent_presence_stats.csv` | Per-parent **presence-count mean & std** across the 300 discovery items, plus min/max and fraction of items where the parent is present. | The load/salience of each parent dimension — which demands are pervasive (e.g. `localization_grounding`, mean 5.54) vs. sparse (`scope_overreach`, present in 43% of items). |
| 4 | `saturation_curve.csv` | **Items annotated vs. cumulative unique features**: one row per discovery event (first appearance of a sub-feature), with running unique count. | Discovery saturation — the codebook keeps gaining features up to item 300 (last new feature at item 300, 89 emergent total), evidence the sample was **not** fully saturated. |
| 5 | `candidate_rubrics_v0.md` | The **v0 candidate scoring rubrics** (VL / GS / MA), verbatim copy of `../Putative_emergent_features_rubrics.md`, in ADeLe/DeLeAn levels-0–5 format. | The first operationalization step: turning consolidated parents into scoreable 0–5 demand rubrics. |
| — | `build_provenance.py` | The generator script (see below). | Reproducibility of files 1–5. |

## Column notes

- **`origin`** — `seeded`: a dimension carried in from ADeLe/DeLeAn (arXiv 2503.06378)
  and the Tolan et al. (2021) visual-processing branch; `emergent`: newly induced from
  the discovery items.
- **`presence_std`** — sample standard deviation (ddof = 1), matching the `sd_count`
  convention in `../dimension_granularity.csv`. The `presence_mean` and `presence_std`
  columns reproduce that file's `mean_count` / `sd_count` exactly.
- **`cumulative_unique_features`** — the count of distinct sub-features seen at or before
  that `items_annotated` index; equals the row number because each row is a first
  appearance. Only the 89 emergent dimensions generate events (seeded dimensions are in
  the codebook from item 0).

## The 9 parent dimensions

`localization_grounding`, `value_readout`, `referent_mismatch`,
`multi_element_reasoning`, `gestalt_shape`, `encoding_literacy`,
`legibility_render`, `claim_linguistic_load`, `scope_overreach`.

(The `tally_depth` column in `../dimension_counts.csv` is a total-tally aggregate, not a
parent dimension, and is excluded here.)

## Source files (in `../`, the discovery output)

- `dimension_mapping.csv` → files 1 & 2
- `dimension_counts.csv` (per-item parent tallies, N=300) → file 3
- `saturation_log.csv` → file 4
- `Putative_emergent_features_rubrics.md` → file 5
- cross-check reference: `dimension_granularity.csv`

## Regenerating

From `feature_discovery/`:

```bash
python3 emergent_provenance/build_provenance.py
```

Pure standard library, deterministic. It re-reads the source CSVs above and rewrites
files 1–4 plus the rubric copy.
