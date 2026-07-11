# Demand-Level Annotation for Figure-Based Scientific Claim Verification: Inductive Discovery Pass — Methods

**Date of run:** 2026-06-12.
**Scope:** This document records the methodology of the inductive
codebook-discovery pass only. Codebook consolidation, large-scale scoring,
and difficulty regression are subsequent phases not covered here.

---

## 1. Objective

We extend the DeLeAn demand-annotation methodology of ADeLe (Zhou et al.,
arXiv 2503.06378) from text-only items to multimodal items consisting of a
natural-language scientific claim paired with a single evidence figure. The
goal of this pass is to inductively construct a candidate codebook of
*demand dimensions* — properties describing what verifying a claim against a
figure requires of a verifier (visual decoding, reasoning, knowledge) —
together with operationalized 0–5 anchors. The pass produces a candidate
codebook for later consolidation; it does not produce final item scores and
does not fit any model.

Two design constraints govern every dimension:

1. **Demands, not verdicts.** Each dimension describes what an item requires,
   never whether its claim is true. A supported claim and its refuted twin
   over near-identical figures must receive near-identical demand profiles.
2. **Operationalized anchors.** Each dimension's levels are tied to
   observable criteria a human or a weaker model could apply from the text
   and figure alone, not to judgments of sophistication.

### 1.1 Lineage and theoretical grounding
The codebook restores the visual-processing branch of the Tolan et al.
(2021) capability taxonomy that ADeLe discarded as inapplicable to its
text-only battery, and subdivides it into figure-intrinsic dimensions. The
seed dimensions are calibrated against three prior results: FUGU (arXiv
2510.21740), which localizes VLM data-visualization failures to the
vision–language handoff (exact-value readout, multi-point integration, and
lookup-vs-gestalt failing through distinct mechanisms); Verma & Fan (CogSci
2025), which found surface features such as chart type explain only ~16% of
reliable human difficulty variance, motivating treating chart type as a
control rather than a signal dimension; and the DeLeAn rubric format of ADeLe
Appendix 10 (per-dimension construct description plus level descriptions with
observable criteria, anchors spaced so demand rarity increases roughly an
order of magnitude per level).

---

## 2. Data

All items are single-figure-evidence claim-verification instances. SciVer
multi-evidence items were never downloaded, sampled, or read. Two sources
were used, both already present locally under the project root.

### 2.1 SciVer (chengyewang/SciVer)
The combined `valset.json` (1000 items) and `testset.json` (2000 items) were
filtered to `type == "chart"`, yielding **817 single-figure items** across
**622 unique papers**. (Items with no top-level `type` field are
multi-evidence by construction — they carry `item1`/`item2` keys — and were
excluded.) Each item provides a `paperid`, an `image_path` to a rendered
figure PNG, a `claim`, a boolean `label`, a `claim_type`
(`direct`/`analytical` in the single-figure pool; `parallel`/`sequential`
occur only among the excluded multi-evidence items), and the originating
`item` (figure number) used to recover the caption from the per-paper JSON
under `SciVer/papers/`. In this pool the `claim` equals the unperturbed
`origin_statement` exactly when `label` is true and the perturbed statement
exactly when false; labels were therefore withheld from annotators (Section
4.2) to prevent verdict leakage.

### 2.2 SciClaimEval (alabnii/sciclaimeval-shared-task)
`dev_task1_release.json` (747 items) was filtered to `evi_type == "figure"`,
yielding **265 figure-evidence items** across **54 unique papers**, with
labels Supported (149) / Refuted (116) and modification operations Legend
Swap (126), Category Swap (46), Graph Flip (40), Supported_claim_only (33),
and Graph Swap (20). Items carry a `claim_id_pair` that links a
supported-original and refuted-modified twin sharing one claim; 116 pairs of
size 2 exist plus one large group (size 33) of unpaired supported-only items.
Each item provides `paper_id`, `evi_path`, `claim`, `caption`, `label`,
`operation`, `domain` (peerj/ml/nlp), and a dataset-supplied `use_context`
flag.

---

## 3. Splits (authoritative for all future phases)

For each source we created a fresh, paper-grouped 80/20 train/test split,
grouped so that no paper appears on both sides, and saved it as the
authoritative split for the project. Splits are deterministic under a fixed
seed (`SEED = 20260612`); the random number generator is seeded once and
papers are shuffled before greedy assignment of whole papers to the test side
until the 20% item target is reached.

| Source | Pool items | Pool papers | Train items | Test items | Train papers | Test papers | Paper overlap |
|---|---|---|---|---|---|---|---|
| SciVer single-figure | 817 | 622 | 654 | 163 | 497 | 125 | 0 |
| SciClaimEval figure | 265 | 54 | 212 | 53 | 46 | 8 | 0 |

Files: `sciver_singlefig_split.json`, `sciclaimeval_split.json`. Each record
carries a boolean `used_in_feature_discovery`. Each file's `meta` block
records the description, creation date, seed, pool/train/test sizes, and
paper counts. The prior SciVer split that existed elsewhere in the repository
was treated as superseded and not reused.

**Discovery items were drawn only from the train side of each split. The test
sides were never sampled, read, or modified.**

---

## 4. Discovery sample

### 4.1 Sampling and stratification
- **SciVer: 200 items** from the train side, stratified to be balanced across
  the two single-figure reasoning subsets and label-balanced within each:
  50 items per (claim_type ∈ {direct, analytical}) × (label ∈ {true, false})
  cell, giving 100 direct / 100 analytical and 100 true / 100 false. The
  brief's `parallel`/`sequential` strata do not exist in the single-figure
  pool and were unreachable by construction.
- **SciClaimEval: 100 items** from the train side. Ten supported/refuted twin
  pairs (20 items) were selected first as purity controls, spread across
  modification operations; the remaining 80 items were drawn by proportional
  stratification over `operation`. The realized operation distribution of the
  100-item sample is Legend Swap 41, Category Swap 21, Graph Flip 17,
  Supported_claim_only 13, Graph Swap 8.

The 10 twin pairs span Legend Swap (3), Category Swap (3), Graph Flip (2),
and Graph Swap (2). Twin membership is recorded in the manifest via a
`twin_id` field.

### 4.2 Figures, manifest, and provenance records
Each sampled item's evidence figure was copied to
`feature_discovery/figures/<item_id>.png` (300 PNGs). Three provenance
artifacts were written:

- `manifest.csv` — one row per item: `item_id`, `source`, `claim_text`,
  `label`, `figure_path`, `paper_id`, `subset_or_operation`, `twin_id`,
  dataset-supplied context flag, an annotator-judged `needs_context_judged`
  field (filled during annotation), and the caption.
- `discovery_items.csv` — `item_id`, `source`, `paper_id`, `figure_path`
  (300 rows).
- `used_in_feature_discovery = true` set on each of the 300 items inside the
  two split JSON files (200 SciVer, 100 SciClaimEval; all on train sides;
  zero on test sides). Future phases can recover the exact discovery-exposed
  set from the split files alone.

---

## 5. Codebook and seed dimensions

The codebook (`codebook.md`, finalized as `candidate_features.md`) follows
the DeLeAn format: per dimension a taxonomy tag (figure-intrinsic /
claim-intrinsic / relational), a one-line human-applicable definition, and
0/2/4 anchors with observable criteria. It was initialized with **23 seeded
dimensions** tagged `origin = seeded`:

- *Figure-intrinsic (11):* chart_type, data_elements_to_integrate
  (FUGU-calibrated element counts), panel_count_layout, visual_density_clutter,
  legend_complexity, uncertainty_elements, axis_complexity, label_indirection,
  encoding_convention_literacy, near_tie_discrimination, marks_clipped_occluded.
- *Claim-intrinsic (4):* negation_present, quantifier_strength,
  numeric_specificity, claim_length_atomic_assertions.
- *Relational (8):* readout_precision, lookup_vs_gestalt, vlat_task_type,
  cross_panel_synthesis, quantitative_reasoning, attention_scan_demand,
  domain_knowledge_demand, caption_sufficiency.

Two dimensions are categorical (chart_type, vlat_task_type) and recorded as
category strings rather than 0–5 loadings.

---

## 6. Discovery protocol

### 6.1 Batching
The 300 items were processed in **30 batches of 10**. Batches mix the two
sources; twin-pair members are kept within a single batch so divergence can
be checked. Batch composition is deterministic (`batches.json`, same seed).

### 6.2 Per-item, per-batch annotation
For each item an annotator viewed the figure PNG, read the claim and caption,
and recorded:
1. a rough 0–5 loading on **every** existing codebook dimension (categorical
   dimensions recorded as category strings) — these tallies are for coverage
   tracking, not final scores;
2. a `needs_context` judgment (no / helpful / required) with a short reason,
   capturing whether verification plausibly needs paper text beyond the
   figure and caption;
3. for each batch, a set of **new candidate dimensions** that the existing
   codebook fails to capture, each with a snake_case name, a one-line
   definition, draft 0/2/4 anchors, high- and low-loading exemplar item_ids,
   a `first_seen_item_id`, and a taxonomy tag. Over-generation was instructed;
   merging, pruning, and overlap were explicitly deferred to consolidation.
4. for batches containing a twin pair, an explicit comparison of the two
   members' profiles, reporting any dimension scored differently (a
   leakage-suspect signal).

### 6.3 Annotator configuration and the verdict-blind design
Each batch was annotated by an independent subagent (Claude Code Agent tool,
`general-purpose` type). The annotation prompt for each batch was generated
by `make_batch_prompt.py`, which embeds the *current* codebook (seeded plus
all emergent dimensions accumulated so far) and the 10 items. The prompt
**withholds the dataset label, the SciClaimEval modification-operation type,
and the SciVer perturbation explanation** from the annotator, so demand
profiles cannot be conditioned on the verdict. Twin-pair members were
presented as independent items sharing a claim, to be scored separately.

Emergent dimensions were fed forward into later batches' prompts (Section
6.4), which lets later annotators (a) record load counts on previously
proposed dimensions and (b) avoid re-proposing an existing dimension under a
new name, making the new-dimension count a more honest measure of marginal
novelty.

### 6.4 Execution: parallel waves and ordered merge
Batches were executed in five waves of six concurrent subagents. Within a
wave, all six annotators worked from the same codebook snapshot (so a
concept genuinely seen by two annotators in the same wave could be proposed
twice; this duplication is expected under the over-generation instruction and
is left for consolidation). After each wave, results were merged in batch
order by `merge_batch.py`, which: validates that every item was scored and
flags items missing any *seeded* tally; appends genuinely new dimensions
(exact-name deduplication only) to the codebook with `origin = emergent` and
`first_seen` index; appends one row per new dimension to
`saturation_log.csv` (`item_index`, `dimension_name`); records twin
divergences under a leakage-suspect section; updates the manifest's
`needs_context_judged`; and recomputes the stopping statistic. Prompts for
the next wave were regenerated after merging so they carried the updated
codebook.

### 6.5 Stopping criterion
The pre-registered rule: stop early if 50 consecutive items produce fewer
than 2 new emergent dimensions (codebook saturated); hard cap at all 300
items. If 300 is reached with the last 50 items still producing 3+ new
dimensions, declare non-saturation explicitly.

---

## 7. Aggregation and flagging

After all 300 items, `build_candidate_features.py` aggregated the 30 batch
result JSONs into per-dimension statistics: number of items that scored the
dimension (the denominator, which is below 300 for emergent dimensions
because each was only tallied by batches generated after it entered the
codebook), mean loading, population variance, count of nonzero loadings, and
count of loadings ≥ 2 (the "load"). Categorical dimensions report category
frequencies instead.

Two review flags were assigned. **LEAKAGE-SUSPECT** marks dimensions whose
scoring partly amounts to detecting a discrepancy between claim and figure,
which correlates with the verdict; most such dimensions were proposed from
refuted (modified-figure) items and must be re-tested for twin-invariance and
reformulated to score the demand site (presence of a checkable discrepancy)
rather than the outcome (whether claim and figure actually disagree) before
any regression use. **DEEP-REASONING** marks dimensions whose reliable
scoring needs claim/figure semantic judgment or multi-step inference, so a
cheap annotator model is expected to be noisy on them; these require tighter
operationalization or expensive-model annotation and validation against an
expert-scored subset in the scoring phase. The flag sets are curated lists
defined in the build script, not thresholds on the load statistics.

---

## 8. Results

### 8.1 Codebook size and saturation
**89 emergent dimensions** were proposed, for **112 total** (23 seeded + 89
emergent). The stopping criterion was **not** met; the run proceeded to the
300-item hard cap. The trailing 50-item new-dimension count, by window:

| Items | New dimensions first-seen |
|---|---|
| 1–50 | 27 |
| 51–100 | 17 |
| 101–150 | 15 |
| 151–200 | 8 |
| 201–250 | 12 |
| 251–300 | 10 |

The curve flattened from ~27 to a plateau near 8–12 per 50 items but did not
converge; the final window still produced 10 new dimensions, above the
threshold of 2. The codebook is therefore reported as **not saturated**. The
residual rate is dominated by (a) increasingly chart-type-specific readout
idioms (UpSet rows, Venn regions, radar spokes, contour extents) and (b) the
leakage-suspect claim-vs-figure discrepancy family, which keeps splintering
into new surface forms. The recommended next step is codebook consolidation
(many emergent dimensions are near-synonyms) before any decision to extend
the sample.

### 8.2 Load concentration
Seeded relational and integration dimensions carried the broadest load
(items scored ≥ 2 out of 300): caption_sufficiency 300, data_elements_to_
integrate 296, attention_scan_demand 295, claim_length_atomic_assertions 292,
readout_precision 279, lookup_vs_gestalt 274, domain_knowledge_demand 267.
numeric_specificity had the highest variance (4.44). The strongest emergent
additions by load were a VLM-handoff legibility cluster (small-text /
resolution / render legibility, ~118–131 of 210 items where present) and
subgroup_partition_selection (151/210).

### 8.3 Near-degenerate seeded dimension
Only **negation_present** was near-degenerate in this sample (28/300 items
scored ≥ 2, variance 0.42): explicit negation is rare in these claims. All
other seeded dimensions retained usable variance.

### 8.4 Flag counts
Of 112 dimensions, **53 are flagged DEEP-REASONING** and **23 are flagged
LEAKAGE-SUSPECT** (the leakage set is a subset of the deep-reasoning set).
Nine emergent dimensions recorded zero load on every item scored after the
one that prompted them, indicating single-figure-type idioms that are
candidates for dropping or folding during consolidation.

### 8.5 Twin-pair leakage analysis
Annotators scored every true twin pair's two members with identical demand
profiles; no seeded or emergent dimension diverged across any of the 10 true
twin pairs. The single recorded twin note was a false alarm: those two items
share a figure but pose different claims about different sub-panels, so the
divergence is claim-driven, not modification-driven.

Identity is the *designed* outcome rather than a defect: the SciClaimEval
modifications (Legend Swap, Graph Flip, Category Swap, Graph Swap) alter the
data semantics that determine the verdict while preserving the figure's
structural complexity and leaving the claim text unchanged, so every
structural/demand feature should be invariant and only the (unscored) verdict
should change. The informative question is therefore not whether profiles are
identical but whether the dimensions *most able* to leak — the relational
claim-vs-figure discrepancy family — held invariant on the members where they
were exercised.

They did, and this is positive evidence rather than a vacuous pass. On the
twin members where these dimensions engaged at nonzero levels, they scored
identically across the supported original and the refuted modification even
though the refuted member literally contains the discrepancy that makes the
claim false:

| Twin | Operation | claim_figure_label_mismatch | claim_figure_referent_mismatch | label_value_consistency_check | printed_value_claim_conflict_potential |
|---|---|---|---|---|---|
| twin_09 | Category Swap | 3 | 3 | 3 | 2 |
| twin_06 | Graph Flip | 2 | 2 | 1 | 0 |
| twin_00 | Legend Swap | 1 | 1 | 2 | – |

(values are identical on both members; "–" = dimension not yet in the codebook
when that batch ran). A leaking dimension would have scored the refuted
member higher, since the swap/flip introduces a real mismatch; it did not.
The annotator scored the *demand* (how much label/referent reconciliation
verification requires) symmetrically, not the *outcome* (whether claim and
figure agree). This is the strongest available test of the discrepancy family
and it passed.

The surviving caveat is statistical power, not design: only about 3–5 of the
10 pairs exercised the discrepancy dimensions at nonzero levels (the rest
show "–" because those dimensions had not yet been proposed when the early
batches ran, or scored zero). The positive twin-invariance evidence for the
LEAKAGE-SUSPECT discrepancy dimensions therefore rests on a handful of pairs.
Expanding the twin set (Section 11 / future work: SciVer origin-vs-perturbed
twins can be synthesized for essentially all 817 figures from existing
fields) would raise this from "passing on a small sample" to a demonstrated
result, and the members should be placed in different batches so identity is
an independent test rather than a within-batch co-assessment. The claim-
intrinsic narrative dimensions (mechanistic_causal_overreach,
mechanism_attribution_demand, claim_figure_scope_overreach) are twin-invariant
by construction because the claim text is identical across a twin, and are not
a leakage risk despite their flag (Section 11).

### 8.6 Context requirement
Across 300 items the `needs_context` judgments were 114 "no", 137 "helpful",
49 "required": approximately 16% of single-figure items plausibly cannot be
verified from figure and caption alone (claim references off-figure
mechanisms, missing panels, or external results). Per-item judgments are in
the manifest.

---

## 9. Compute provenance (model and effort)

This is recorded because it bears on reproducibility and on the
interpretation of the codebook as a model artifact.

- **Annotators:** all 30 batch-annotation subagents ran on **Claude Opus 4.8
  (`claude-opus-4-8`)**, verified from the per-subagent transcripts (609/609
  assistant messages; zero on any other model). No model override was passed
  to the Agent tool; subagents inherited the parent session model at spawn
  time.
- **Orchestrator:** the controlling session began on Claude Fable 5
  (`claude-fable-5`) and was downgraded to Opus 4.8 at 20:06:50Z, ~8 minutes
  in. Fable covered only initial schema inspection and the start of the
  split builder; every figure-viewing, tally, emergent-dimension proposal,
  merge, and the final codebook assembly occurred on Opus 4.8. The original
  design intended Fable as the annotator; the realized codebook is an Opus
  4.8 artifact. The VLM-failure-mode dimensions in particular reflect Opus's
  own inspection of the figures and may differ under a different annotator
  model.
- **Effort:** `~/.claude/settings.json` set `"effortLevel": "high"`. This is a
  global user setting read by the harness for every request, not a per-agent
  parameter, and no override was passed; main thread and all subagents ran
  under the same configured effort. The transcript does not record a
  per-request effort field, and extended-thinking blocks are absent in both
  the main thread (known high effort) and the subagent logs, so effort cannot
  be confirmed from response content alone — but no mechanism would give
  subagents a different effort than the global setting.

---

## 10. Reproducibility

Deterministic given a fixed environment:

1. Place the SciVer and SciClaimEval data under the project root as in
   Section 2.
2. Run `build_discovery_set.py` (`SEED = 20260612`) to regenerate the
   splits, the 300-item sample, twin selection, figure copies, manifest,
   discovery-exposure flags, and the 30-batch plan exactly.
3. Initialize `codebook.md` with the 23 seeded dimensions and an empty
   `saturation_log.csv`.
4. For each batch, generate the prompt with `make_batch_prompt.py`, annotate
   with a `general-purpose` subagent pinned to `claude-opus-4-8` (do not rely
   on the Fable→Opus downgrade recurring; pin the model explicitly), with
   `effortLevel: high` in settings and no per-agent overrides, then merge with
   `merge_batch.py`.
5. Aggregate and flag with `build_candidate_features.py`.

The annotation step itself is stochastic at the model level (sampling
temperature is not pinned through the Agent tool), so the exact set of
emergent dimensions is not bit-reproducible; the data construction, sampling,
splits, batch assignment, and aggregation are.

---

## 11. Limitations

- **Non-saturation.** The codebook is not saturated; the emergent set is
  almost certainly inflated by near-synonyms and should be consolidated
  before use.
- **Annotator-model confound.** The codebook reflects Opus 4.8's judgment of
  figure-verification demands, not the originally intended Fable annotator.
- **Load denominators differ.** Emergent-dimension load fractions are
  relative to the number of items scored after the dimension's introduction,
  not to 300; late-introduced dimensions have smaller, noisier denominators.
- **Single annotator per item.** Each item was scored once; no inter-annotator
  agreement was measured in this pass. Tallies are coverage signals, not
  calibrated scores.
- **Leakage not fully cleared.** LEAKAGE-SUSPECT dimensions passed the
  available twin checks only incidentally; each needs a dedicated
  twin-invariance test before entering a difficulty model.
- **Caption recovery.** SciVer captions were recovered from per-paper JSON by
  figure number, with a fallback to the base figure number for subfigure
  keys; a small number of subfigure captions may be the parent-figure caption.

---

## 12. File inventory

| File | Contents |
|---|---|
| `sciver_singlefig_split.json` | Authoritative SciVer paper-grouped 80/20 split; discovery-exposure flags |
| `sciclaimeval_split.json` | Authoritative SciClaimEval paper-grouped 80/20 split; discovery-exposure flags |
| `discovery_items.csv` | 300 discovery items (item_id, source, paper_id, figure_path) |
| `manifest.csv` † | Full per-item metadata incl. twin_id, caption, needs_context_judged |
| `figures/` † | 300 evidence figure PNGs |
| `codebook.md` † | Working codebook: definitions, anchors, per-batch log, twin notes |
| `candidate_features.md` | Final codebook with load/variance, origin tags, flags, and discovery summary |
| `saturation_log.csv` | 89 rows (item_index, dimension_name) for the accumulation curve |
| `batch_results/batch_00..29.json` † | Raw per-item tallies, needs_context, and proposed dimensions per batch |
| `batches.json` † | Deterministic 30×10 batch assignment |
| `build_discovery_set.py` | Builds splits, sample, manifest, figures, batch plan |
| `make_batch_prompt.py` | Generates the verdict-blind annotation prompt for a batch |
| `merge_batch.py` | Validates and merges a batch; updates codebook, saturation log, manifest |
| `build_candidate_features.py` | Aggregates statistics and assigns review flags |

† Not distributed in this repository. `figures/` and `manifest.csv` contain
figure images and claim/caption text redistributed from SciVer and
SciClaimEval; `codebook.md`, `batch_results/`, and `batches.json` are working
artifacts superseded by `candidate_features.md` and the `emergent_provenance/`
package. All are regenerable from the public datasets via
`build_discovery_set.py` (`SEED = 20260612`) and the pipeline in Section 10,
except the raw batch annotations, which are stochastic at the model level.

---

## References

- ADeLe / DeLeAn: arXiv 2503.06378.
- FUGU: arXiv 2510.21740.
- Verma & Fan, CogSci 2025 (chart-difficulty surface-feature analysis).
- Tolan et al., 2021 (capability taxonomy).
