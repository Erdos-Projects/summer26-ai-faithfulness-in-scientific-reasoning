# candidate_features.md — figure-verification demand codebook (discovery output)

Inductive discovery pass over 300 claim+figure items (200 SciVer
single-figure-evidence, 100 SciClaimEval figure-evidence). Extends
ADeLe/DeLeAn (arXiv 2503.06378) to multimodal claim verification, restoring
the visual-processing branch of Tolan et al. (2021). This file is the full
candidate codebook (seeded + emergent), each dimension annotated with its
load across the discovery sample, origin tag, and review flags. It is NOT a
consolidated rubric and NOT a set of final scores — consolidation, anchor
calibration, and scale annotation are later phases.

## How to read the load metrics
- "scored >=2 of N scored": items given a loading of at least 2 (a
  meaningful demand), out of the N items that included this dimension in
  their tally. N < 300 for emergent dimensions because each was only tallied
  by batches generated AFTER it entered the codebook (parallel annotation
  waves). Load fractions are therefore relative to N, not 300; a dimension
  introduced late (small N) with high load may still be common overall.
- Categorical dimensions (chart_type, vlat_task_type) report category
  frequencies instead of a 0-5 load.

## Review flags
- DEEP-REASONING: scoring this dimension reliably needs claim/figure
  semantic judgment or multi-step inference; a cheap annotator model will
  likely be noisy. These need either tighter operationalization or
  expensive-model annotation in the scoring phase.
- LEAKAGE-SUSPECT: scoring could in principle amount to detecting a
  discrepancy between claim and figure, which correlates with the verdict.
  These emergent "...mismatch / ...conflict / referent / ungroundable"
  dimensions were proposed from refuted (modified-figure) items, so the flag
  marks them for verification, not condemnation. Two sub-cases, which the flag
  currently conflates and consolidation should separate:
    (a) Claim-intrinsic narrative dimensions (mechanistic_causal_overreach,
        mechanism_attribution_demand, claim_figure_scope_overreach) are
        twin-invariant BY CONSTRUCTION — the claim text is identical across a
        twin, so they cannot encode the verdict. These are not a leakage risk
        and the flag is over-cautious for them.
    (b) Relational discrepancy dimensions (claim_figure_label_mismatch,
        claim_figure_referent_mismatch, label_value_consistency_check,
        printed_value_claim_conflict_potential, etc.) are the genuine risk.
        EMPIRICAL CHECK: on the twin pairs where these engaged at nonzero
        levels, they scored IDENTICALLY across the supported original and the
        refuted modification (e.g. Category-Swap twin: label_mismatch 3/3,
        referent_mismatch 3/3, label_value_consistency 3/3, printed_value
        conflict 2/2) even though the refuted member contains the actual
        mismatch. A leaking dimension would have scored the refuted member
        higher; none did. This is positive evidence they score the demand
        (reconciliation required), not the outcome. The caveat is sample size:
        only ~3-5 of 10 pairs exercised these dimensions. Before regression
        use, re-test twin-invariance at scale (synthesizable from SciVer
        origin-vs-perturbed pairs across ~800 figures), with twin members in
        separate batches so identity is an independent test. Any dimension
        that fails at scale must be reformulated to score the checkable
        discrepancy SITE (e.g. presence of a printed value to compare), not
        whether claim and figure actually agree, or discarded.

# Candidate demand-dimension codebook (discovery pass)

Inductive discovery over 300 claim+figure items (200 SciVer single-figure,
100 SciClaimEval figure-evidence). Extends ADeLe/DeLeAn (arXiv 2503.06378) to
multimodal items, restoring the visual-processing branch of Tolan et al.
(2021) that ADeLe discarded.

Governing principles:
1. DEMANDS, NOT VERDICTS — every dimension describes what verifying the item
   requires, never whether the claim is true. Twin pairs (same claim,
   original vs. modified figure) must receive near-identical profiles;
   dimensions that differ across twins are flagged leakage-suspect.
2. OPERATIONALIZED ANCHORS — 0–5 anchors concrete enough for a human or a
   weak model to apply from the text alone; observable criteria, not
   sophistication judgments.

Tally convention: per-batch loadings are rough 0–5 for coverage tracking
(categorical dimensions record the category), not final scores. Per-batch
tallies live in `batch_results/batch_NN.json`; the Batch log at the bottom
of this file summarizes them.

---

## Seeded dimensions

### chart_type  [figure-intrinsic, categorical]
- origin: seeded
- definition: Type of the evidence chart (bar, line, scatter, box, heatmap,
  pie, area, violin, composite/other). Control variable, not expected to
  carry main difficulty signal (Verma & Fan, CogSci 2025).
- anchors: categorical — record the type(s) present.
- **load**: categorical (300 items); top categories: line=106, bar=54, scatter=25, heatmap=23, composite=17, other=14

### data_elements_to_integrate  [figure-intrinsic]
- origin: seeded
- definition: How many distinct visual data elements (points, bars, line
  segments, cells) must be jointly considered to verify the claim.
  FUGU-calibrated.
- anchors: 0 = single point/element; 2 = about 4 elements; 4 = 16 or more
  elements; 5 = dense (element count not practically enumerable, e.g. whole
  scatter cloud or heatmap).
- **load**: 296 items scored >=2 of 300 scored (mean 3.59, variance 1.22, nonzero 300)

### panel_count_layout  [figure-intrinsic]
- origin: seeded
- definition: Number of panels/subplots in the evidence figure and layout
  complexity.
- anchors: 0 = single panel; 2 = 2–3 panels in a simple row/column;
  4 = grid of 4+ panels or nested/inset panels; 5 = 9+ panels or irregular
  composite layout.
- **load**: 146 items scored >=2 of 300 scored (mean 1.45, variance 2.79, nonzero 150)

### visual_density_clutter  [figure-intrinsic]
- origin: seeded
- definition: Degree of overplotting, overlapping marks, gridline/annotation
  clutter in the region(s) relevant to the claim.
- anchors: 0 = sparse, all relevant marks clearly separated; 2 = some
  overlap but every relevant mark individually resolvable; 4 = relevant
  marks overlap so that some must be disambiguated by inference (ordering,
  occlusion); 5 = relevant region is a dense mass where individual marks
  are not resolvable.
- **load**: 226 items scored >=2 of 300 scored (mean 2.64, variance 2.37, nonzero 267)

### legend_complexity  [figure-intrinsic]
- origin: seeded
- definition: Number of category-to-mark mappings the legend (or equivalent
  key) defines that are needed for the claim.
- anchors: 0 = no legend needed; 2 = 2–3 mappings; 4 = 6+ mappings or
  mappings split across multiple legends; 5 = 10+ mappings or legend
  symbols nearly indistinguishable.
- **load**: 232 items scored >=2 of 300 scored (mean 2.11, variance 1.89, nonzero 240)

### uncertainty_elements  [figure-intrinsic]
- origin: seeded
- definition: Presence and role of uncertainty encodings (error bars, CIs,
  bands) or distribution-encoding chart types relevant to verification.
- anchors: 0 = none present or none relevant; 2 = error bars/bands present
  and must be visually noted; 4 = claim verification requires reading
  uncertainty extents (e.g. overlap of CIs); 5 = requires comparing
  distributional shape/spread across groups.
- **load**: 75 items scored >=2 of 300 scored (mean 0.7, variance 1.67, nonzero 79)

### axis_complexity  [figure-intrinsic]
- origin: seeded
- definition: Non-default axis conventions that must be handled: log scale,
  dual axes, truncated/broken axes, inverted direction, nonlinear ticks.
- anchors: 0 = plain linear axes starting where expected; 2 = one
  non-default feature clearly labeled (e.g. log scale with labeled decades,
  truncated baseline); 4 = dual axes or combination of non-default
  features requiring active tracking of which applies; 5 = misleading or
  unlabeled non-default axes that must be inferred.
- **load**: 77 items scored >=2 of 300 scored (mean 0.7, variance 1.42, nonzero 90)

### label_indirection  [figure-intrinsic]
- origin: seeded
- definition: Lookup-path length from a claim entity to its mark.
- anchors: 0 = marks directly labeled; 2 = one mapping step (e.g.
  color→legend); 4 = chained mappings (e.g. abbreviation→caption→legend→
  mark) or caption-only mapping; 5 = mapping requires outside-figure
  knowledge.
- **load**: 265 items scored >=2 of 300 scored (mean 2.12, variance 0.78, nonzero 287)

### encoding_convention_literacy  [figure-intrinsic]
- origin: seeded
- definition: Chart-grammar knowledge required to decode the encoding.
- anchors: 0 = bar/line/scatter conventions; 2 = boxplot-level conventions
  (quartiles, whiskers); 4 = violin/Q-Q/ROC/kaplan-meier-level conventions;
  5 = bespoke or field-specific encoding explained only in caption/text.
- **load**: 130 items scored >=2 of 300 scored (mean 1.22, variance 1.63, nonzero 174)

### near_tie_discrimination  [figure-intrinsic]
- origin: seeded
- definition: Presence of visually close values among the elements the claim
  is about (potential for fine discrimination being required).
- anchors: 0 = relevant values clearly separated (>10% of axis range);
  2 = some relevant values within roughly 5–10% of axis range; 4 = claim
  hinges on values within ~2% of axis range or under one tick interval;
  5 = differences at or below plausible rendering/marker resolution.
- **load**: 242 items scored >=2 of 300 scored (mean 2.33, variance 1.57, nonzero 269)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### marks_clipped_occluded  [figure-intrinsic]
- origin: seeded
- definition: Relevant marks clipped at plot boundaries, occluded by other
  marks/annotations, or extending beyond axis limits.
- anchors: 0 = none; 2 = minor occlusion not affecting relevant marks;
  4 = a claim-relevant mark partially clipped/occluded; 5 = a claim-relevant
  mark substantially hidden so its value must be inferred.
- **load**: 105 items scored >=2 of 300 scored (mean 0.91, variance 1.26, nonzero 139)

### negation_present  [claim-intrinsic]
- origin: seeded
- definition: Negation in the claim (syntactic or lexical: "not", "no",
  "fails to", "absence of", "rather than").
- anchors: 0 = none; 2 = single negation; 4 = multiple or nested negations /
  negation interacting with a quantifier; 5 = double negation or negation
  scope ambiguity.
- **load**: 28 items scored >=2 of 300 scored (mean 0.22, variance 0.42, nonzero 34)

### quantifier_strength  [claim-intrinsic]
- origin: seeded
- definition: Strength/scope of quantifiers in the claim ("all", "every",
  "consistently", "always", "most", "some", "only").
- anchors: 0 = no quantifier (single specific fact); 2 = existential or
  hedged quantifier ("some", "tends to"); 4 = universal quantifier over an
  enumerable set ("all methods", "across every dataset"); 5 = universal
  plus exception handling ("all but one", "only X").
- **load**: 117 items scored >=2 of 300 scored (mean 1.03, variance 1.77, nonzero 131)

### numeric_specificity  [claim-intrinsic]
- origin: seeded
- definition: How numerically specific the claim is.
- anchors: 0 = purely qualitative trend ("increases"); 2 = ordinal or
  approximate magnitude ("roughly doubles", "about 0.8"); 4 = one exact
  value cited; 5 = multiple exact values or derived quantities (differences,
  percentages) cited.
- **load**: 216 items scored >=2 of 300 scored (mean 3.06, variance 4.44, nonzero 221)

### claim_length_atomic_assertions  [claim-intrinsic]
- origin: seeded
- definition: Number of independently checkable atomic assertions in the
  claim.
- anchors: 0 = one atomic assertion; 2 = two; 4 = four; 5 = five or more.
- **load**: 292 items scored >=2 of 300 scored (mean 3.46, variance 1.27, nonzero 297)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### readout_precision  [relational]
- origin: seeded
- definition: Precision of value readout the claim requires from the figure.
- anchors: 0 = none (existence/identity only); 2 = approximate position
  (which is higher/lower); 4 = exact value read against axis ticks;
  5 = multiple exact values that must be read and combined.
- **load**: 279 items scored >=2 of 300 scored (mean 3.05, variance 1.87, nonzero 288)

### lookup_vs_gestalt  [relational]
- origin: seeded
- definition: Whether verification is point lookup or statistical gestalt
  (correlation, cluster, distribution-shape judgment). FUGU: these fail
  differently in VLMs.
- anchors: 0 = single-point lookup; 2 = few-point comparison; 4 = gestalt
  over many marks (trend/correlation/spread); 5 = gestalt judgment that
  must be made separately in multiple regions/panels and then compared.
- **load**: 274 items scored >=2 of 300 scored (mean 2.71, variance 1.47, nonzero 294)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### vlat_task_type  [relational, categorical]
- origin: seeded
- definition: VLAT task type(s) the verification act instantiates: retrieve
  value / find extremum / compare / characterize trend / identify
  correlation / detect outlier.
- anchors: categorical — record all that apply.
- **load**: categorical (300 items); top categories: retrieve value=181, compare=180, characterize trend=86, find extremum=38, retrieve value; compare=20, identify correlation=12

### cross_panel_synthesis  [relational]
- origin: seeded
- definition: Degree to which information must be combined across panels of
  the single evidence figure.
- anchors: 0 = single panel suffices; 2 = locate the right panel among
  several, read within it; 4 = combine readings from 2+ panels; 5 = combine
  readings from 3+ panels or align values across panels with different
  axes.
- **load**: 131 items scored >=2 of 300 scored (mean 1.37, variance 2.77, nonzero 142)

### quantitative_reasoning  [relational]
- origin: seeded
- definition: Arithmetic/quantitative operations on extracted values (ADeLe
  QLq-style).
- anchors: 0 = none; 2 = single comparison or subtraction; 4 = multi-step
  arithmetic (percentage change, ratio of differences); 5 = chained
  computation or aggregation over many values (mean of 6+ readings).
- **load**: 209 items scored >=2 of 300 scored (mean 2.0, variance 1.73, nonzero 251)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### attention_scan_demand  [relational]
- origin: seeded
- definition: How much of the figure must be searched to locate
  claim-relevant elements.
- anchors: 0 = relevant element immediately localizable (titled, labeled,
  single mark); 2 = scan one axis or one series; 4 = scan most of the
  figure or check every series/category; 5 = exhaustive search of a dense
  figure with no localization cues.
- **load**: 295 items scored >=2 of 300 scored (mean 3.07, variance 1.11, nonzero 299)

### domain_knowledge_demand  [relational]
- origin: seeded
- definition: Field knowledge needed beyond the figure to interpret terms or
  conventions in the claim/figure.
- anchors: 0 = figure-sufficient, everyday vocabulary; 2 = technical terms
  resolvable from figure/caption labels; 4 = requires knowing field
  conventions not stated (e.g. that lower perplexity is better);
  5 = requires substantive domain facts external to the figure.
- **load**: 267 items scored >=2 of 300 scored (mean 2.68, variance 1.48, nonzero 293)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### caption_sufficiency  [relational]
- origin: seeded
- definition: Whether the claim could be verified from the caption text
  alone, without decoding the figure. (Demand on visual processing is
  highest when caption alone is useless.)
- anchors: 0 = caption alone settles the claim; 2 = caption substantially
  narrows it but figure reading still required; 4 = caption is generic,
  nearly all information must come from the figure; 5 = caption absent or
  actively unhelpful (mismatched labels, refers to other panels).
- **load**: 300 items scored >=2 of 300 scored (mean 3.7, variance 0.54, nonzero 300)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

---

## Emergent dimensions

(appended during discovery; origin=emergent, first_seen=<item index>)

### mechanistic_causal_overreach  [claim-intrinsic]
- origin: emergent
- first_seen: item 1 (sciver_test_0235)
- definition: Degree to which the claim asserts a causal/mechanistic explanation (why an effect occurs) that the figure can only corroborate at the level of an observed pattern, not establish.
- anchors: 0 = Claim is purely descriptive of figure content (a value, a comparison, a trend).; 2 = Claim attaches one causal/explanatory phrase ('due to', 'because', 'enabling') to a figure-readable pattern.; 4 = Claim's main assertion is a mechanism (latent encoding, contractivity, memory capacity, bias correction) the figure cannot directly show.
- exemplars: high = sciver_test_0235, sciver_val_0189, sciver_val_0012, sciver_val_0132; low = scev_val_fig_0262, scev_val_fig_1542, sciver_test_0724
- rationale: Several claims couple a checkable readout to an unfalsifiable mechanistic clause; this taxes the verifier to separate the figure-verifiable part from the explanatory rhetoric, which no current dimension captures.
- **load**: 91 items scored >=2 of 210 scored (mean 1.54, variance 3.06, nonzero 105)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### claim_entity_axis_coverage  [relational]
- origin: emergent
- first_seen: item 2 (sciver_val_0189)
- definition: Whether the value/region the claim references actually falls within the plotted axis range, or lies at or beyond the axis limits so it cannot be read.
- anchors: 0 = All claim-referenced values lie comfortably inside the plotted axis range.; 2 = A referenced value sits near an axis edge, readable only at the boundary tick.; 4 = A referenced value lies at or beyond the axis limit (e.g. claim cites spectral radius >2.0 but axis stops at 2.0).
- exemplars: high = sciver_val_0189; low = scev_val_fig_0262, sciver_test_0955, sciver_val_0132
- rationale: When a claim's region falls outside the plotted domain, verification demands recognizing absence of evidence rather than reading a value; this is a distinct demand from near_tie or clipping and is not currently captured.
- **load**: 39 items scored >=2 of 210 scored (mean 0.52, variance 0.67, nonzero 68)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### rotated_microlabel_decoding  [figure-intrinsic]
- origin: emergent
- first_seen: item 3 (scev_val_fig_0262)
- definition: Effort to read claim-relevant axis/tick labels that are rotated (often 45-90 degrees) and rendered at small point size, common in clustered heatmaps and stacked bar forests.
- anchors: 0 = All claim-relevant labels horizontal and clearly legible.; 2 = Some claim-relevant labels rotated but still legible at normal viewing.; 4 = Claim-relevant labels rotated and so small/dense that reading them requires zoom or near-pixel inspection.
- exemplars: high = scev_val_fig_0262, scev_val_fig_1542; low = sciver_test_0955, sciver_test_0235
- rationale: A VLM-specific failure mode: rotated tiny tick labels gate the entire row/column lookup in heatmaps; label_indirection covers mapping length but not the raw legibility cost of rotated microtext.
- **load**: 36 items scored >=2 of 210 scored (mean 0.53, variance 1.14, nonzero 51)

### panel_localization_among_many  [relational]
- origin: emergent
- first_seen: item 6 (sciver_test_0724)
- definition: Effort to find which titled sub-panel of a small-multiples grid the claim refers to before any reading can begin.
- anchors: 0 = Single panel or claim names no specific panel within a small set.; 2 = Claim names a panel among 3-5 that must be matched by title.; 4 = Claim names a specific panel among 9+ small-multiples that must be located by reading per-panel titles.
- exemplars: high = sciver_test_0724, sciver_val_0012; low = sciver_test_0235, scev_val_fig_0262
- rationale: cross_panel_synthesis measures combining across panels, and attention_scan measures within-panel search, but neither isolates the up-front cost of identifying the correct panel by title in a large grid before reading begins.
- **load**: 90 items scored >=2 of 210 scored (mean 1.12, variance 1.83, nonzero 99)

### in_figure_answer_printing  [relational]
- origin: emergent
- first_seen: item 9 (sciver_val_0132)
- definition: Whether the claim's key quantity is printed as text inside the figure (panel title, annotation) rather than requiring decode from marks against axes.
- anchors: 0 = No claim value is printed; all must be read from marks/axes.; 2 = Some supporting value is printed but the core comparison still needs mark reading.; 4 = The claim's headline number is printed verbatim as a panel title/annotation (e.g. 'FDR: 90%').
- exemplars: high = sciver_val_0132; low = sciver_test_0955, sciver_test_0235
- rationale: When a headline number is printed in-figure, the visual-decode demand collapses to OCR; this strongly modulates difficulty in the opposite direction from caption_sufficiency and is not otherwise captured.
- **load**: 46 items scored >=2 of 210 scored (mean 0.71, variance 1.98, nonzero 47)

### colormap_value_readout  [relational]
- origin: emergent
- first_seen: item 20 (sciver_test_0749)
- definition: Whether verification requires mapping a continuous color/intensity at a spatial location back to a numeric value via a colorbar.
- anchors: 0 = no color-encoded value needed (categorical hue or discrete marks only); 2 = must judge relative color (which region is hotter/higher) without a number; 4 = must read an approximate numeric value from a continuous colorbar at a named spatial location
- exemplars: high = sciver_test_0749; low = sciver_test_0562
- rationale: Existing readout dimensions assume axis-tick lookup; continuous-field colorbar inversion is a distinct VLM failure mode not covered by readout_precision or encoding_convention_literacy.
- **load**: 23 items scored >=2 of 210 scored (mean 0.4, variance 1.32, nonzero 24)

### spatial_roi_localization  [relational]
- origin: emergent
- first_seen: item 20 (sciver_test_0749)
- definition: Whether the claim references an unlabeled spatial region (e.g. 'jet-disk interface', 'left tail') that must be located from semantic cues before any value is read.
- anchors: 0 = target is at an axis-labeled tick or a directly labeled mark; 2 = target region named and roughly indicated by a gridline or panel quadrant; 4 = target is an unlabeled physical/semantic region the verifier must infer from domain knowledge of the field structure
- exemplars: high = sciver_test_0749; low = sciver_test_0562
- rationale: attention_scan_demand measures search breadth but not the need to semantically identify an unlabeled region; this is a separate grounding step before readout.
- **load**: 44 items scored >=2 of 210 scored (mean 0.62, variance 1.0, nonzero 71)

### printed_data_label_reliance  [figure-intrinsic]
- origin: emergent
- first_seen: item 15 (sciver_test_0562)
- definition: Degree to which claim-relevant values are printed as text annotations on the figure, shifting demand from visual estimation to text OCR plus arithmetic.
- anchors: 0 = no numeric labels printed; all values must be estimated against axes; 2 = some bars/points labeled but claim-relevant ones require axis estimation; 4 = all claim-relevant values printed as on-figure text, so verification reduces to reading text and computing
- exemplars: high = sciver_test_0562, sciver_test_0069; low = sciver_val_0321
- rationale: readout_precision scores precision required but conflates estimated-from-axis with read-from-printed-label; these load VLM capabilities very differently (FUGU vision-language handoff).
- **load**: 45 items scored >=2 of 210 scored (mean 0.71, variance 2.06, nonzero 45)

### claim_figure_label_mismatch  [relational]
- origin: emergent
- first_seen: item 14 (sciver_val_0422)
- definition: Whether claim entity names differ from the figure's literal labels, requiring a synonym/alias mapping before lookup.
- anchors: 0 = claim terms appear verbatim as figure labels; 2 = minor alias (abbreviation or expansion resolvable from caption); 4 = claim uses descriptive synonyms (e.g. 'geometric'->'geo', 'random'->'rand') with no in-figure key
- exemplars: high = sciver_val_0422, scev_val_fig_0057; low = sciver_test_0562
- rationale: label_indirection covers in-figure lookup path length but not the lexical gap between claim wording and figure tokens, a distinct disambiguation demand.
- **load**: 107 items scored >=2 of 210 scored (mean 1.41, variance 1.18, nonzero 157)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### subgroup_partition_selection  [relational]
- origin: emergent
- first_seen: item 11 (scev_val_fig_0181)
- definition: Whether the claim restricts attention to one sub-region/group of the figure that must be isolated before any reading (e.g. only the 'pu' column, only the 'Language Unseen / Script Seen' band).
- anchors: 0 = claim concerns the whole figure or a single panel; 2 = claim concerns one clearly demarcated panel/group among a few; 4 = claim concerns a sub-region defined by an internal divider or one column among a dense grid that must be selected before reading
- exemplars: high = scev_val_fig_0181, sciver_test_0069; low = sciver_val_0321
- rationale: cross_panel_synthesis assumes named subplots; this captures scoping to an internally-divided region within one panel, a precondition not captured by panel or scan dimensions.
- **load**: 151 items scored >=2 of 210 scored (mean 1.7, variance 1.35, nonzero 159)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### small_render_legibility  [figure-intrinsic]
- origin: emergent
- first_seen: item 13 (sciver_val_0009)
- definition: Whether the figure as supplied is small/low-resolution enough that claim-relevant marks, labels, or text approach illegibility.
- anchors: 0 = all claim-relevant elements crisply legible; 2 = some axis text small but claim-relevant marks legible; 4 = claim-relevant marks or labels near the resolution limit, requiring guesswork to resolve
- exemplars: high = sciver_val_0009, sciver_test_0082; low = sciver_test_0562
- rationale: A figure-intrinsic rendering property (per FUGU rasterization/small-text failure modes) orthogonal to clutter; two items here were supplied at sizes that degrade legibility independent of inherent density.
- **load**: 119 items scored >=2 of 210 scored (mean 1.81, variance 1.96, nonzero 163)

### stacked_segment_decomposition  [figure-intrinsic]
- origin: emergent
- first_seen: item 29 (sciver_val_0455)
- definition: Whether the claim requires reading the length of an interior segment of a stacked bar (not the whole bar), demanding subtraction of boundary positions rather than reading from a baseline.
- anchors: 0 = no stacking, or claim concerns total bar length from baseline; 2 = one interior segment to read, bounded by clearly contrasting colors; 4 = two or more interior segments to isolate, or segment boundaries low-contrast/abutting
- exemplars: high = sciver_val_0455; low = sciver_test_0757
- rationale: Reading an interior stacked segment requires endpoint differencing rather than baseline lookup, a VLM-specific failure mode not captured by readout_precision or data_elements_to_integrate.
- **load**: 10 items scored >=2 of 210 scored (mean 0.18, variance 0.62, nonzero 11)

### claim_figure_scale_mismatch  [relational]
- origin: emergent
- first_seen: item 26 (sciver_test_0757)
- definition: Whether numbers in the claim are stated in units or a scale that differs from the figure's displayed axis, requiring the verifier to reconcile or rescale before comparison.
- anchors: 0 = claim numbers in the same units/scale as the figure axis; 2 = claim uses an approximation or rounding of figure values in matching units; 4 = claim numbers expressed on a different scale/unit than any axis (must convert or the values cannot match directly)
- exemplars: high = sciver_test_0757; low = scev_val_fig_1552
- rationale: When claim magnitudes (e.g. ~55 GFLOPS) do not match the figure's labeled values (189.5), the verification act requires detecting a unit/scale mismatch, a demand orthogonal to readout precision.
- **load**: 28 items scored >=2 of 210 scored (mean 0.37, variance 0.8, nonzero 37)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### threshold_crossing_localization  [relational]
- origin: emergent
- first_seen: item 28 (sciver_test_0994)
- definition: Whether verification requires finding the coordinate at which a curve crosses a reference line or threshold and reading the orthogonal-axis value there.
- anchors: 0 = no threshold; direct value or trend read; 2 = single curve crossing a clearly drawn reference line, linear axis; 4 = multiple curves crossing a threshold, or crossing read on a log/nonlinear axis where the x-value must be interpolated
- exemplars: high = sciver_test_0994; low = sciver_test_0757
- rationale: Locating where a curve meets a p-value line and projecting to the x-axis is a distinct act from value retrieval or trend gestalt, and compounds with log-scale reading.
- **load**: 14 items scored >=2 of 210 scored (mean 0.19, variance 0.48, nonzero 18)

### matrix_axis_orientation_demand  [figure-intrinsic]
- origin: emergent
- first_seen: item 25 (scev_val_fig_1552)
- definition: Whether verifying the claim requires correctly resolving which axis of a matrix/grid encoding is predicted vs. actual (or row vs. column) to attribute a cell value to the right category.
- anchors: 0 = no matrix/grid, or cell directly labeled with its category pair; 2 = matrix present with conventionally oriented, clearly labeled axes; 4 = matrix where the claim's correctness depends on non-default or easily transposed axis orientation (e.g. predicted on rows)
- exemplars: high = scev_val_fig_1552; low = sciver_test_0757
- rationale: Confusion-matrix and heatmap claims can fail purely from transposing the predicted/actual axes, a VLM error mode not covered by label_indirection or encoding_convention_literacy.
- **load**: 22 items scored >=2 of 210 scored (mean 0.27, variance 0.65, nonzero 23)

### target_panel_localization  [relational]
- origin: emergent
- first_seen: item 23 (scev_val_fig_0241)
- definition: In a multi-panel composite, whether the claim names a specific sub-panel that the verifier must first locate among many lettered panels before any reading.
- anchors: 0 = single panel, or no panel reference needed; 2 = claim cites a panel letter and 2-3 panels present; 4 = claim cites a panel letter that must be found among 6+ visually similar panels
- exemplars: high = scev_val_fig_0241, scev_val_fig_1545; low = sciver_test_0757
- rationale: Finding 'Figure 4C' inside a 9-panel grid is a localization step preceding any value read; cross_panel_synthesis covers combining panels but not single-target sub-panel search.
- **load**: 99 items scored >=2 of 210 scored (mean 1.18, variance 1.75, nonzero 106)

### axis_claim_unit_mismatch  [relational]
- origin: emergent
- first_seen: item 39 (sciver_test_0382)
- definition: Degree to which the claim's quantity/units differ from what the figure's axis directly encodes, forcing a conversion or interpretation step before any value can be compared.
- anchors: 0 = claim quantity and units match an axis label verbatim (e.g. axis 'Accuracy (%)', claim '37.5%'); 2 = claim quantity matches axis concept but uses a transformed form (axis log-scaled, claim states linear value; axis ratio, claim states percent); 4 = claim names a quantity/unit not present on any axis (axis 'Efficiency', claim 'seconds'/'runtime'; axis unitless, claim '%^2') so the mapping must be assumed
- exemplars: high = sciver_test_0382, scev_val_fig_0202; low = sciver_val_0373, sciver_val_0285
- rationale: Existing axis_complexity covers non-default scales but not the case where the claim's named quantity has no axis to read it against; this is a distinct VLM failure where the model must invent a mapping.
- **load**: 50 items scored >=2 of 210 scored (mean 0.7, variance 1.15, nonzero 77)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### derived_aggregate_readout  [relational]
- origin: emergent
- first_seen: item 40 (scev_val_fig_0202)
- definition: Whether the claimed number is not a directly markable point but a statistic aggregated over multiple marks (variance, spread, mean, count), requiring computation over a set rather than a single lookup.
- anchors: 0 = claimed value corresponds to a single mark readable directly off an axis; 2 = claimed value is a simple difference/ratio of two readable marks; 4 = claimed value is an aggregate over 3+ marks (variance, spread across radial axes, mean) with no single mark representing it
- exemplars: high = scev_val_fig_0202, sciver_val_0106; low = sciver_val_0373, sciver_val_0285
- rationale: readout_precision and quantitative_reasoning each partly touch this, but neither isolates the demand of recognizing that a stated number is an emergent statistic with no point to look up, a common cause of false-grounding in VLMs.
- **load**: 70 items scored >=2 of 210 scored (mean 0.93, variance 1.82, nonzero 79)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### color_axis_decoding  [figure-intrinsic]
- origin: emergent
- first_seen: item 34 (sciver_test_0027)
- definition: Whether a continuous colorbar (color-mapped third variable) must be decoded to verify the claim, requiring reading hue back to a numeric scale.
- anchors: 0 = no colorbar, or color is categorical and covered by legend; 2 = continuous colorbar present and a coarse hue-to-value reading is needed (which region is higher/lower); 4 = claim hinges on mapping specific hues of individual marks back to colorbar values within a dense cloud
- exemplars: high = sciver_test_0027; low = sciver_val_0373, sciver_val_0285
- rationale: legend_complexity and label_indirection assume discrete category keys; continuous color-to-number decoding is a separate perceptual demand (color constancy, fine hue discrimination) where VLMs are known weak.
- **load**: 26 items scored >=2 of 210 scored (mean 0.42, variance 1.35, nonzero 27)

### radial_encoding_literacy  [figure-intrinsic]
- origin: emergent
- first_seen: item 40 (scev_val_fig_0202)
- definition: Demand from non-Cartesian positional encodings (radar/spider, polar) where value is read along radial spokes rather than a single shared axis.
- anchors: 0 = Cartesian chart, value read along horizontal/vertical axis; 2 = polar/circular axis but with a single clearly labeled radial scale; 4 = radar/spider with multiple spokes sharing one radial scale, so each value is read along a differently-oriented spoke
- exemplars: high = scev_val_fig_0202; low = sciver_val_0285, sciver_val_0373
- rationale: encoding_convention_literacy lists violin/ROC/KM but not radial geometry; reading values off rotated spokes against one centered scale is a distinct decoding act poorly handled by VLMs.
- **load**: 6 items scored >=2 of 210 scored (mean 0.11, variance 0.4, nonzero 8)

### small_figure_resolution_strain  [figure-intrinsic]
- origin: emergent
- first_seen: item 35 (sciver_test_0844)
- definition: Degree to which the rendered figure is physically small or low-resolution relative to the detail the claim requires, independent of intrinsic clutter.
- anchors: 0 = figure large/high-res; all claim-relevant marks and tick labels crisp; 2 = figure moderately small; values legible but fine gaps (sub-tick) hard to resolve; 4 = figure very small/thumbnail-scale so axis labels, marker shapes, or near-overlapping lines are at the edge of legibility
- exemplars: high = sciver_test_0844, sciver_val_0155; low = sciver_val_0285, sciver_test_0027
- rationale: near_tie_discrimination is about value proximity in data space; this captures pixel-space legibility (rasterization/thumbnail strain) which the BACKGROUND notes as a VLM-specific failure independent of the underlying data.
- **load**: 118 items scored >=2 of 210 scored (mean 1.82, variance 1.99, nonzero 164)

### marker_shape_keying  [figure-intrinsic]
- origin: emergent
- first_seen: item 33 (sciver_val_0155)
- definition: Whether the claim entity is identified by marker SHAPE (star/diamond/plus/pentagon/triangle) rather than color or position, requiring fine glyph discrimination.
- anchors: 0 = entity identified by position or direct label, no shape distinction needed; 2 = two clearly distinct marker shapes must be told apart; 4 = 3+ similar marker shapes (e.g. star vs pentagon vs plus) keyed only in caption must be distinguished, possibly at small size
- exemplars: high = sciver_val_0155; low = sciver_val_0285, sciver_val_0373
- rationale: legend_complexity counts mappings but not the perceptual cost of discriminating similar glyph shapes; caption-only shape keys (star/diamond/plus/pentagon) are a documented VLM weak point distinct from color legends.
- **load**: 37 items scored >=2 of 210 scored (mean 0.44, variance 0.97, nonzero 39)

### claim_figure_scope_overreach  [relational]
- origin: emergent
- first_seen: item 41 (sciver_test_0258)
- definition: How much of the claim asserts mechanism/causation/explanation that the figure cannot establish, requiring the verifier to separate figure-checkable content from non-checkable narrative.
- anchors: 0 = every clause maps to a readable figure feature; 2 = one explanatory clause beyond the figure (e.g. 'because X'); 4 = multiple causal/mechanistic clauses naming entities absent from figure and caption
- exemplars: high = sciver_test_0231, sciver_test_0258; low = sciver_test_0613, sciver_val_0269
- rationale: Distinct from caption_sufficiency and domain_knowledge: it measures the portion of the claim that is structurally unverifiable from any visual readout, a key handoff failure when a VLM treats narrative as confirmed.
- **load**: 93 items scored >=2 of 210 scored (mean 1.55, variance 2.9, nonzero 112)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### label_value_consistency_check  [relational]
- origin: emergent
- first_seen: item 47 (sciver_test_0076)
- definition: Whether verification requires cross-checking a descriptive label in the claim against a figure-stated label (axis/panel title/legend) that may not match, rather than just reading a value.
- anchors: 0 = claim labels match figure labels; only value lookup needed; 2 = claim uses a synonym/abbreviation that must be matched to a figure label; 4 = claim asserts a label-attribute pairing (e.g. band name with frequency range) that must be checked against the figure's own panel/axis labeling
- exemplars: high = sciver_test_0076; low = scev_val_fig_0014, sciver_test_0613
- rationale: Captures a VLM failure where the model reads a value but ignores that the claim's descriptive label conflicts with the figure's stated label; not covered by label_indirection (lookup path) or readout_precision (value precision).
- **load**: 124 items scored >=2 of 210 scored (mean 1.52, variance 0.99, nonzero 168)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### small_text_legibility_demand  [figure-intrinsic]
- origin: emergent
- first_seen: item 46 (sciver_test_0231)
- definition: Degree to which claim-relevant text (axis ticks, panel labels, row names, value annotations) is rendered at small size or low resolution requiring fine OCR-like reading.
- anchors: 0 = all claim-relevant text large and crisp; 2 = some relevant text small but legible at normal zoom; 4 = claim-relevant text is tiny/blurred (multi-panel composite labels, dense tick text) such that misreading is plausible
- exemplars: high = sciver_test_0231, scev_val_fig_0240, sciver_test_0525; low = sciver_test_0613, scev_val_fig_0014
- rationale: A figure-intrinsic VLM-specific failure mode (small/rasterized text) orthogonal to visual_density_clutter, which is about mark overlap rather than glyph legibility.
- **load**: 131 items scored >=2 of 210 scored (mean 1.98, variance 1.85, nonzero 176)

### panel_localization_demand  [relational]
- origin: emergent
- first_seen: item 45 (scev_val_fig_0240)
- definition: Difficulty of identifying which sub-panel of a multi-panel composite the claim refers to before any reading can begin.
- anchors: 0 = single panel or panel named unambiguously and easy to spot; 2 = panel cited by letter and must be located among a few labeled panels; 4 = panel must be located among many panels of varied type with small or ambiguous panel labels
- exemplars: high = scev_val_fig_0240; low = scev_val_fig_0014, sciver_test_0613
- rationale: cross_panel_synthesis covers combining across panels but not the prior step of finding the one correct panel in a dense composite; this is a distinct attention/search cost for VLMs.
- **load**: 99 items scored >=2 of 210 scored (mean 1.2, variance 1.78, nonzero 107)

### text_entity_matching_demand  [relational]
- origin: emergent
- first_seen: item 49 (sciver_val_0269)
- definition: Whether verification requires reading and comparing textual content (names, words, tokens) inside the figure rather than decoding graphical marks against axes.
- anchors: 0 = no in-figure text content to compare; purely graphical; 2 = match a few text labels between claim and figure; 4 = read and compare multiple text items across rows/groups and judge lexical or semantic overlap
- exemplars: high = sciver_val_0269; low = sciver_test_0258, scev_val_fig_0014
- rationale: Text-table figures pose a non-charting verification act (lexical/semantic comparison of listed items) that none of the chart-reading dimensions describe.
- **load**: 19 items scored >=2 of 210 scored (mean 0.23, variance 0.45, nonzero 25)

### mechanism_attribution_demand  [claim-intrinsic]
- origin: emergent
- first_seen: item 52 (sciver_test_0488)
- definition: Degree to which the claim asserts a causal/mechanistic explanation (why an effect occurs) rather than only an observable pattern the figure could show.
- anchors: 0 = claim is purely descriptive of a visible pattern (value, trend, comparison); 2 = claim names a pattern plus a single weakly-coupled attribution ('due to X'); 4 = claim's truth hinges on a multi-step causal mechanism (priors, gradient flow, exploration/exploitation) not depictable in any chart
- exemplars: high = sciver_test_0488, sciver_val_0033; low = sciver_test_0865, scev_val_fig_0292
- rationale: Several claims bundle a figure-checkable pattern with unfalsifiable mechanism prose; this taxes the verifier to separate the verifiable visual core from the non-visual rationale, a demand no existing dimension isolates.
- **load**: 85 items scored >=2 of 210 scored (mean 1.5, variance 3.13, nonzero 101)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### claim_figure_referent_mismatch  [relational]
- origin: emergent
- first_seen: item 55 (sciver_test_0322)
- definition: Degree to which entities/panels the claim cites are absent from or mislabeled in the rendered figure, forcing reconciliation before verification.
- anchors: 0 = every claim referent (panel, axis quantity, series) is present and identically named in the figure; 2 = referent present but named differently than claim (axis says 'score', claim says 'structural similarity'); 4 = a panel or entity the claim explicitly cites (e.g. 'Fig 4(c)', 'ResNet-32') is not present in the rendered image
- exemplars: high = scev_val_fig_0016, sciver_test_0322; low = sciver_test_0865, sciver_val_0173
- rationale: Distinct from label_indirection (lookup path within a coherent figure) and caption_sufficiency: here the cited referent is missing or renamed, a VLM-specific failure mode where the model must detect the gap rather than traverse a path.
- **load**: 64 items scored >=2 of 210 scored (mean 0.95, variance 1.14, nonzero 115)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### annotation_overlay_counting  [figure-intrinsic]
- origin: emergent
- first_seen: item 60 (sciver_val_0033)
- definition: Degree to which verification requires counting or comparing the frequency of overlaid annotation marks (circles, highlights, shaded regions) rather than the underlying data marks.
- anchors: 0 = no annotation overlays relevant to the claim; 2 = a few overlay marks must be noted but not counted (single shaded region); 4 = claim hinges on counting/comparing many overlay marks across regions (red circles per method row)
- exemplars: high = sciver_val_0033; low = sciver_test_0865, scev_val_fig_0292
- rationale: Captures the demand of enumerating overlay annotations (red circles marking errors) which is orthogonal to data-mark density and known to fail in VLMs that miss small overlaid marks.
- **load**: 9 items scored >=2 of 210 scored (mean 0.1, variance 0.23, nonzero 11)

### label_rotation_legibility  [figure-intrinsic]
- origin: emergent
- first_seen: item 57 (scev_val_fig_0292)
- definition: Degree to which claim-relevant category labels are rotated, small, or densely packed such that reading them is itself a bottleneck.
- anchors: 0 = relevant labels horizontal and well-spaced; 2 = relevant labels rotated <=45deg but individually legible; 4 = relevant labels rotated ~90deg and tightly packed so adjacent labels risk misreading
- exemplars: high = scev_val_fig_0292; low = sciver_test_0865, sciver_test_0961
- rationale: A VLM-specific text-decoding demand (rotated/dense tick labels) not covered by attention_scan_demand or visual_density_clutter, which concern marks not label text.
- **load**: 28 items scored >=2 of 210 scored (mean 0.41, variance 1.0, nonzero 38)

### exhaustive_set_extremum_demand  [relational]
- origin: emergent
- first_seen: item 57 (scev_val_fig_0292)
- definition: Degree to which the claim requires confirming that a named subset is the top/bottom-k of a larger enumerable set, requiring inspection of every candidate.
- anchors: 0 = claim about a single named element or no ranking; 2 = claim about which of 2-3 named elements is largest; 4 = claim that k named elements are the top-k of a set of many (must rule out every other element)
- exemplars: high = scev_val_fig_0292; low = sciver_test_0865, sciver_test_0961
- rationale: Combines find-extremum with universal-set verification: a top-4-of-19 claim demands ruling out 15 alternatives, a load distinct from generic attention_scan or single-extremum find tasks.
- **load**: 29 items scored >=2 of 210 scored (mean 0.4, variance 1.13, nonzero 29)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### venn_region_arithmetic  [figure-intrinsic]
- origin: emergent
- first_seen: item 70 (scev_val_fig_0290)
- definition: Whether verification requires reading and summing counts across the overlapping regions of a Venn/set diagram to obtain a per-set or per-subset total.
- anchors: 0 = No Venn/set diagram, or a single printed region count answers the claim directly.; 2 = Two or three Venn regions must be read and compared, counts legible.; 4 = Multiple small/overlapping Venn regions (4-set or more) must be summed per group to derive the compared totals.
- exemplars: high = scev_val_fig_0290; low = scev_val_fig_0288, scev_val_fig_0147
- rationale: derived_aggregate_readout covers aggregating marks against an axis, but set-diagram region summation has no axis and a unique containment geometry (which regions belong to which set) not captured by any existing dimension.
- **load**: 1 items scored >=2 of 150 scored (mean 0.01, variance 0.03, nonzero 1)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### significance_bracket_pairing  [relational]
- origin: emergent
- first_seen: item 61 (scev_val_fig_0288)
- definition: Whether verification requires matching a printed significance annotation (p-value bracket, asterisk) to the specific pair of groups it spans, rather than reading a data value.
- anchors: 0 = No significance annotation, or claim does not concern statistical significance.; 2 = A single p-value bracket present and the pair it connects is unambiguous.; 4 = Multiple significance brackets present and the claim's group pair must be matched to the correct bracket by tracing its span endpoints.
- exemplars: high = scev_val_fig_0288; low = sciver_test_0632, scev_val_fig_0147
- rationale: uncertainty_elements covers error bars/CIs but not the distinct act of attributing a printed p-value bracket to the correct group pair, the crux of significance claims in biology strip plots.
- **load**: 4 items scored >=2 of 150 scored (mean 0.07, variance 0.18, nonzero 4)

### encoded_variable_substitution  [relational]
- origin: emergent
- first_seen: item 69 (sciver_val_0247)
- definition: Whether the figure encodes a grouping variable different from the one the claim attributes the visual pattern to, requiring the verifier to detect that the claimed explanatory variable is not the one mapped to the visual channel.
- anchors: 0 = The variable the claim attributes the pattern to is the variable the figure's legend/axis encodes.; 2 = Claim variable maps to the encoded variable via a documented synonym or caption note.; 4 = Claim attributes the visual pattern to a variable (e.g. parcellation resolution) that the figure does not encode at all; the figure encodes a different variable (e.g. site).
- exemplars: high = sciver_val_0247; low = sciver_test_0632, scev_val_fig_0147
- rationale: claim_figure_referent_mismatch covers missing/renamed entities, but here all entities exist and are legible; the gap is that the claim's causal variable is simply not the channel-encoded one, a distinct grounding failure for VLMs that read clusters and assume the claimed cause.
- **load**: 9 items scored >=2 of 150 scored (mean 0.17, variance 0.33, nonzero 14)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### histogram_bin_localization_readout  [relational]
- origin: emergent
- first_seen: item 80 (sciver_test_0926)
- definition: Whether verification requires identifying a histogram bin by its x-position/center and reading its frequency height, where bins are unlabeled intervals not directly markable points.
- anchors: 0 = no histogram, or claim concerns overall shape only with no specific bin; 2 = one bin must be located by approximate center and its height read; 4 = two or more bins must each be located by center value and their heights read and compared, with bin centers not printed
- exemplars: high = sciver_test_0926; low = sciver_test_0737, sciver_test_0503
- rationale: Locating 'the bin centered at ~53 dB' requires mapping a claimed x-value to an unlabeled interval before height readout, a step neither readout_precision nor attention_scan isolates.
- **load**: 5 items scored >=2 of 150 scored (mean 0.1, variance 0.32, nonzero 5)

### cross_series_rate_comparison  [relational]
- origin: emergent
- first_seen: item 79 (sciver_val_0086)
- definition: Whether the claim compares the growth/change RATE of two series (slopes over an interval) rather than their levels, requiring estimating and comparing trends rather than point values.
- anchors: 0 = claim about a single value or a level comparison at one point; 2 = claim about the direction of one series' trend over an interval; 4 = claim compares the rates of change of two distinct series over a shared interval (e.g. 'grows at twice the pace'), requiring slope estimation for each
- exemplars: high = sciver_val_0086; low = sciver_test_0503, sciver_test_0737
- rationale: lookup_vs_gestalt and quantitative_reasoning touch trend and arithmetic separately, but neither captures the specific demand of estimating and ratioing two slopes, compounded here by dual axes.
- **load**: 8 items scored >=2 of 150 scored (mean 0.13, variance 0.29, nonzero 10)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### enumerated_list_membership_check  [relational]
- origin: emergent
- first_seen: item 82 (scev_val_fig_0256)
- definition: Whether the claim lists many named entities each requiring its own lookup, so verification is a checklist over a long enumeration rather than a single comparison.
- anchors: 0 = claim names one or two entities to check; 2 = claim lists 3-5 entities each needing an independent lookup; 4 = claim enumerates 8+ named entities (e.g. metabolites) each requiring a separate cell/mark lookup and sign judgement
- exemplars: high = scev_val_fig_0256, scev_val_fig_0257; low = sciver_test_0739
- rationale: claim_length_atomic_assertions caps at 5+ and does not capture the search cost of a long parallel enumeration where each item is a fresh figure lookup; this load scales with list length distinctly.
- **load**: 23 items scored >=2 of 150 scored (mean 0.41, variance 0.98, nonzero 26)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### shape_convexity_curvature_judgment  [relational]
- origin: emergent
- first_seen: item 81 (sciver_val_0219)
- definition: Whether verification requires judging the curvature/convexity of a curve or the trend of successive differences, not just monotonic direction or single values.
- anchors: 0 = no curvature judgment; direction or single value only; 2 = judge whether a curve is rising or falling overall (monotonicity); 4 = judge convexity/concavity or whether marginal differences grow vs shrink across consecutive points
- exemplars: high = sciver_val_0219; low = sciver_test_0739
- rationale: lookup_vs_gestalt and characterize-trend cover direction but not second-order shape (convexity, growing marginal drops), a distinct second-derivative reasoning demand.
- **load**: 13 items scored >=2 of 150 scored (mean 0.25, variance 0.63, nonzero 17)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### diagonal_reference_conformance  [relational]
- origin: emergent
- first_seen: item 85 (scev_val_fig_0129)
- definition: Whether verification requires judging how closely a plotted curve conforms to an implicit reference line such as the uniform/diagonal CDF or the identity line.
- anchors: 0 = no reference-line conformance needed; 2 = compare a curve to an explicitly drawn reference line; 4 = judge conformance to an implicit/unmarked reference (e.g. that a CDF is uniform i.e. follows the unplotted diagonal)
- exemplars: high = scev_val_fig_0129; low = sciver_test_0739
- rationale: threshold_crossing_localization concerns where a curve meets a line; this concerns conformance of a whole curve to an implicit reference shape (uniformity), a separate gestalt grounded in encoding convention.
- **load**: 6 items scored >=2 of 150 scored (mean 0.08, variance 0.15, nonzero 6)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### rank_position_verification  [relational]
- origin: emergent
- first_seen: item 92 (scev_val_fig_1557)
- definition: Whether the claim asserts a specific ordinal rank (second-lowest, third-highest, median) requiring the verifier to sort the full set and confirm a non-extremal position.
- anchors: 0 = claim names no rank, or only the single max/min (one extremum to find); 2 = claim asserts a rank among 2-3 elements, sortable at a glance; 4 = claim asserts a non-extremal rank (2nd/3rd lowest/highest) over 5+ elements, requiring a full ordering before the position can be confirmed
- exemplars: high = scev_val_fig_1557; low = sciver_val_0220, scev_val_fig_1521
- rationale: exhaustive_set_extremum_demand covers top-k membership and vlat 'find extremum' covers the single max; neither isolates confirming a specific interior ordinal rank, which forces a complete sort rather than an argmax.
- **load**: 9 items scored >=2 of 150 scored (mean 0.13, variance 0.3, nonzero 9)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### cluster_count_gestalt  [relational]
- origin: emergent
- first_seen: item 98 (sciver_val_0245)
- definition: Whether verification requires judging the NUMBER of distinct clusters/groups or whether groups are separated vs. merged in a point cloud or block-structured matrix, rather than reading any value.
- anchors: 0 = no grouping judgment; single-value or trend read; 2 = judge whether two named groups are separated or overlapping; 4 = count the number of distinct clusters/blocks in a dense cloud or matrix, or assert all of many groups are mutually separated
- exemplars: high = sciver_val_0245, sciver_val_0220; low = scev_val_fig_1521, scev_val_fig_1557
- rationale: lookup_vs_gestalt flags statistical-gestalt broadly, but counting clusters / judging mutual separation is a specific topological grouping act (perfectly distinct domain clusters; four clusters in a matrix) distinct from correlation or spread gestalt.
- **load**: 10 items scored >=2 of 150 scored (mean 0.17, variance 0.48, nonzero 10)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### external_reference_dependency  [claim-intrinsic]
- origin: emergent
- first_seen: item 100 (sciver_val_0220)
- definition: Degree to which the claim's truth hinges on a specific external result, citation, or numeric constant not present in the figure or caption (e.g. 'reproduces the four clusters of Yan et al.', 'under tau=0.05').
- anchors: 0 = claim is self-contained in figure/caption terms; 2 = claim cites a named threshold/constant resolvable from caption; 4 = claim cites an external paper's finding or an unstated numeric constant the figure alone cannot confirm
- exemplars: high = sciver_val_0220; low = sciver_val_0323, scev_val_fig_1557
- rationale: domain_knowledge_demand covers field conventions and claim_figure_scope_overreach covers mechanism prose, but neither captures dependency on a specific cited external result or numeric constant that makes part of the claim unverifiable from the figure regardless of reading skill.
- **load**: 12 items scored >=2 of 150 scored (mean 0.23, variance 0.47, nonzero 19)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### caption_axis_quantity_conflict  [relational]
- origin: emergent
- first_seen: item 97 (scev_val_fig_1521)
- definition: Whether the claim/caption describes the swept variable in terms that conflict with the figure's actual axis label, forcing the verifier to detect a description-vs-axis discrepancy.
- anchors: 0 = claim's swept variable matches the figure's axis label; 2 = claim uses a synonym for the axis variable resolvable from caption; 4 = claim/caption names a swept variable (e.g. 'different noise levels') that the figure's x-axis does not encode (axis says 'Length of a chain')
- exemplars: high = scev_val_fig_1521; low = sciver_test_0375, sciver_val_0323
- rationale: axis_claim_unit_mismatch covers unit/scale gaps in the claimed value; this captures the orthogonal case where the claim's described independent/swept variable does not match the plotted axis, a referent-not-value discrepancy.
- **load**: 8 items scored >=2 of 150 scored (mean 0.16, variance 0.37, nonzero 12)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### slope_estimation_from_line  [relational]
- origin: emergent
- first_seen: item 110 (sciver_test_0897)
- definition: Whether verification requires estimating the slope (rate of change) of a fitted/trend line from its geometry rather than reading any marked value.
- anchors: 0 = no slope needed; value or category lookup only; 2 = compare which of two lines is visibly steeper, same x-scale; 4 = compute a numeric slope (cm per cm) or slope ratio across panels with different x-axis spans, so visual steepness alone is misleading
- exemplars: high = sciver_test_0897; low = sciver_test_0858, scev_val_fig_0131
- rationale: quantitative_reasoning and readout_precision cover value arithmetic but not deriving a rate from line geometry; differing axis spans make apparent angle a trap, a distinct VLM grounding demand.
- **load**: 10 items scored >=2 of 150 scored (mean 0.15, variance 0.26, nonzero 12)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### referenced_panel_absent_from_render  [relational]
- origin: emergent
- first_seen: item 108 (scev_val_fig_1513)
- definition: Whether the caption/claim references a sub-panel (e.g. panel b) that is not present in the rendered image, forcing the verifier to work only from the available panel.
- anchors: 0 = all panels referenced by caption/claim are present in the image; 2 = caption names panels (a,b) but the claim only needs the present one; 4 = the claim's evidence depends on a panel the caption names but the image omits
- exemplars: high = scev_val_fig_1513, scev_val_fig_1556; low = sciver_test_0897, sciver_test_0585
- rationale: Distinct from claim_figure_referent_mismatch (renamed/missing entity within a coherent figure): here the caption advertises multi-panel content but the rendered crop supplies only a subset, so the verifier must detect the supplied scope before judging.
- **load**: 12 items scored >=2 of 150 scored (mean 0.2, variance 0.45, nonzero 14)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### gap_trend_comparison  [relational]
- origin: emergent
- first_seen: item 108 (scev_val_fig_1513)
- definition: Whether the claim is about how the vertical distance between two series changes across the x-axis (widening/narrowing disparity), requiring differencing two curves at 2+ x positions.
- anchors: 0 = no inter-series gap involved; single series or single comparison; 2 = judge whether two series gap is larger at one named x than another, qualitatively; 4 = quantify or rank the change in between-series gap across two or more x positions (disparity grows from depth 3 to depth 5)
- exemplars: high = scev_val_fig_1513; low = sciver_test_0858, sciver_test_0585
- rationale: lookup_vs_gestalt and quantitative_reasoning touch multi-mark integration, but neither isolates the act of tracking a between-series gap's evolution, a second-order visual differencing prone to VLM error.
- **load**: 22 items scored >=2 of 150 scored (mean 0.4, variance 0.88, nonzero 27)

### contour_overlay_extent_reading  [figure-intrinsic]
- origin: emergent
- first_seen: item 119 (sciver_test_0787)
- definition: Whether verification requires judging the spatial extent or boundary of irregular overlaid contour/iso-line regions (not point marks) against position axes.
- anchors: 0 = no contours; values read from discrete marks or bars; 2 = one contour region whose extent is roughly indicated and read against an axis; 4 = two or more irregular contour lobes whose outermost extents must each be located and compared against a position axis
- exemplars: high = sciver_test_0787; low = sciver_test_0938, sciver_val_0491
- rationale: Contour-map extent reading (common in astronomy/imaging overlays) is a perceptual act distinct from data_elements_to_integrate or readout_precision, which assume discrete marks; finding the farthest excursion of a jagged iso-contour against an axis is a separate grounding demand.
- **load**: 1 items scored >=2 of 150 scored (mean 0.01, variance 0.03, nonzero 1)

### significance_marker_reading  [figure-intrinsic]
- origin: emergent
- first_seen: item 120 (scev_val_fig_0254)
- definition: Whether the claim's truth depends on reading discrete significance/annotation glyphs (asterisks, daggers) inside cells or near marks rather than the underlying value.
- anchors: 0 = no significance glyphs relevant; only value/sign needed; 2 = a single significance glyph must be noted as present/absent at a named location; 4 = multiple tiny significance glyphs across cells must be located and distinguished by count (* vs ** vs ***)
- exemplars: high = scev_val_fig_0254; low = sciver_test_0938, sciver_val_0491
- rationale: Correlation/p-value heatmaps encode a second discrete layer (asterisk significance) that gates whether an association 'counts'; reading these tiny overlaid star glyphs is orthogonal to colormap_value_readout (continuous hue) and annotation_overlay_counting (which is about counting circles, not graded star tiers).
- **load**: 4 items scored >=2 of 150 scored (mean 0.07, variance 0.18, nonzero 4)

### claim_value_axis_unreadable_scale  [relational]
- origin: emergent
- first_seen: item 122 (sciver_test_0519)
- definition: Whether the claim's stated magnitude cannot be read off the figure because the axis encodes a different resolution (e.g. log decades) while the claim gives absolute large integers, forcing the verifier to reconstruct a value the axis cannot pin down.
- anchors: 0 = Claim magnitude maps to a linearly readable axis position at the claimed precision.; 2 = Claim magnitude is on the axis scale but near a coarse tick so exact value is approximate.; 4 = Claim gives a precise large value (e.g. '8,000 efficiency units') against a log axis whose decade ticks cannot resolve thousands-level precision.
- exemplars: high = sciver_test_0519; low = sciver_test_0972
- rationale: Distinct from claim_figure_scale_mismatch (unit conversion): here units match but the axis resolution (log) makes the claimed precision physically unreadable, a separate grounding failure.
- **load**: 12 items scored >=2 of 110 scored (mean 0.3, variance 0.72, nonzero 14)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### embedded_subplot_value_extraction  [figure-intrinsic]
- origin: emergent
- first_seen: item 127 (sciver_test_0733)
- definition: Whether claim-relevant values live inside tiny spark-plots nested within cells of a larger grid, so a per-cell mini-axis must be decoded before any value is read.
- anchors: 0 = Values read from one chart with a single shared axis.; 2 = A few small subplots each with its own legible mini-axis.; 4 = Many tiny embedded spark-panels (grid of mini line/heatmap plots) each with its own sub-axis at near-illegible size.
- exemplars: high = sciver_test_0733; low = scev_val_fig_1512
- rationale: panel_localization and small_text_legibility touch finding/reading panels, but neither captures that each grid cell is itself a complete chart with its own internal axis that must be decoded; a compound nested-axis demand.
- **load**: 6 items scored >=2 of 110 scored (mean 0.12, variance 0.21, nonzero 7)

### category_total_aggregation_demand  [relational]
- origin: emergent
- first_seen: item 129 (sciver_test_0972)
- definition: Whether the claim's headline figure is a grand total the verifier must obtain by summing printed per-category segment values across multiple bars/groups, not by reading any single bar.
- anchors: 0 = Claim value equals a single bar/segment readable directly.; 2 = Claim value is a sum of two clearly adjacent segments.; 4 = Claim total requires summing one color's printed values across 3+ separate categories before comparison.
- exemplars: high = sciver_test_0972; low = scev_val_fig_0046
- rationale: derived_aggregate_readout covers axis-estimated aggregates; this isolates the case where per-category labels are printed and the total is an additive roll-up across grouped bars, a distinct OCR-plus-sum step where the claimed grand total appears nowhere in the figure.
- **load**: 3 items scored >=2 of 110 scored (mean 0.09, variance 0.32, nonzero 3)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### ordinal_step_to_continuous_mapping  [relational]
- origin: emergent
- first_seen: item 140 (scev_val_fig_1516)
- definition: Whether the claim treats an ordered set of discrete labeled categories on an axis as if it were a continuous quantitative variable, requiring the verifier to decide if the categorical ordering legitimately stands in for the claimed numeric variable.
- anchors: 0 = Claim's independent variable matches a numeric axis with real tick values.; 2 = Axis ordinal labels are a monotone proxy the caption explicitly ties to the claimed quantity.; 4 = Claim names a continuous driver (e.g. 'data size') but the axis shows only ordered annotation labels (Baseline, +DA10, +DA20) whose mapping to that quantity is unstated.
- exemplars: high = scev_val_fig_1516; low = sciver_test_0181, scev_val_fig_0020
- rationale: Distinct from caption_axis_quantity_conflict (synonym mismatch) and axis_claim_unit_mismatch (unit gap): here the axis is ordinal-categorical and the claim asserts a monotone numeric trend over it, a mapping the verifier must judge as valid or not.
- **load**: 6 items scored >=2 of 110 scored (mean 0.12, variance 0.21, nonzero 7)

### legend_text_value_substitution  [relational]
- origin: emergent
- first_seen: item 138 (scev_val_fig_0159)
- definition: Whether the numeric quantity the claim depends on is printed inside the legend entries (per-series annotations) rather than read from plotted marks, shifting the comparison to legend OCR.
- anchors: 0 = No numbers in the legend; values come from marks against axes.; 2 = Legend lists one auxiliary number but the core comparison still needs mark reading.; 4 = The compared quantity (e.g. BiasFo per strategy) is stated only as text in legend entries, so verification is reading and comparing legend numbers.
- exemplars: high = scev_val_fig_0159; low = sciver_test_0091, scev_val_fig_0020
- rationale: in_figure_answer_printing and printed_data_label_reliance cover panel titles and on-mark labels, but not numbers embedded in legend rows that override the plotted points as the actual evidence.
- **load**: 1 items scored >=2 of 110 scored (mean 0.04, variance 0.14, nonzero 1)

### parametric_form_conformance  [relational]
- origin: emergent
- first_seen: item 131 (sciver_val_0121)
- definition: Whether verification requires judging that a plotted curve follows a specific named functional form (exponential, logistic, power-law) rather than just its direction or convexity.
- anchors: 0 = No functional-form claim; direction or value only.; 2 = Claim asserts a coarse shape family (concave/convex, saturating) checkable by eye.; 4 = Claim asserts a specific parametric law (exponential decay, logistic) the verifier must distinguish from nearby forms (e.g. linear vs exponential) by curve geometry.
- exemplars: high = sciver_val_0121, sciver_test_0136; low = scev_val_fig_0020, sciver_test_0091
- rationale: shape_convexity_curvature_judgment covers second-order curvature, but distinguishing a claimed exponential/logistic from a linear or other form is a stricter model-fitting judgment not isolated by existing trend dimensions.
- **load**: 3 items scored >=2 of 110 scored (mean 0.06, variance 0.11, nonzero 4)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### unspecified_panel_disambiguation  [relational]
- origin: emergent
- first_seen: item 135 (scev_val_fig_0018)
- definition: Whether a multi-panel figure offers several candidate panels/conditions but the claim does not name which one, forcing the verifier to decide which panel the claim should be checked against.
- anchors: 0 = Single panel, or claim explicitly names its panel.; 2 = Multiple panels but caption pins the claim's dataset/condition to one.; 4 = Multiple equally-plausible panels (e.g. Symmetric 40% vs Gaussian 30) and the claim cites neither, so the verifier must choose or check both.
- exemplars: high = scev_val_fig_0018; low = sciver_val_0121, sciver_test_0091
- rationale: panel_localization dimensions assume the claim names a target panel; here the gap is the absence of a panel reference when several apply, a distinct under-specification demand.
- **load**: 9 items scored >=2 of 110 scored (mean 0.23, variance 0.34, nonzero 16)

### unlabeled_axis_value_ungroundable  [figure-intrinsic]
- origin: emergent
- first_seen: item 150 (sciver_val_0209)
- definition: Whether the figure lacks axis ticks/scale labels entirely, so a claimed numeric value or named position cannot be grounded against any reference.
- anchors: 0 = Both axes have labeled ticks/scale; any claimed value can be read against them.; 2 = One axis labeled, the other lacks ticks so values on that axis are only ordinal.; 4 = Neither axis shows ticks, scale, or units, so a claimed value (e.g. '~20 h/day') or named position (e.g. 'summer') has no readable reference at all.
- exemplars: high = sciver_val_0209; low = sciver_test_0860, scev_val_fig_0218
- rationale: claim_entity_axis_coverage covers a value falling outside a plotted range; this captures the stronger case where no axis reference exists at all, making any numeric or positional readout impossible regardless of skill.
- **load**: 11 items scored >=2 of 110 scored (mean 0.26, variance 0.69, nonzero 11)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### category_state_attribution  [relational]
- origin: emergent
- first_seen: item 146 (scev_val_fig_0218)
- definition: Whether verifying the claim requires attributing a value to the correct categorical state/series among similarly-encoded sibling categories (e.g. Susceptible vs Intermediate vs Resistant), where picking the wrong state inverts the verdict.
- anchors: 0 = Only one state/series exists, or the claim's state is uniquely encoded and labeled.; 2 = Two or three sibling states present, distinguished by clear color/legend, and the claim names its state.; 4 = Multiple sibling states share one location/cluster and the claim's named state (e.g. 'susceptible') must be matched against the figure's legend before reading, with a different state's value being a plausible trap.
- exemplars: high = scev_val_fig_0218; low = sciver_test_0860, scev_val_fig_0032
- rationale: label_indirection covers lookup-path length and claim_figure_label_mismatch covers lexical aliasing, but neither isolates the demand of binding a value to the correct one of several co-located categorical states (S/I/R) where a sibling state's value is the wrong answer.
- **load**: 53 items scored >=2 of 110 scored (mean 1.31, variance 1.9, nonzero 60)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### multi_panel_directional_aggregation  [relational]
- origin: emergent
- first_seen: item 147 (scev_val_fig_0286)
- definition: Whether the claim asserts a direction-of-change (increase/decrease) for many distinct quantities each living in its own panel, requiring a separate same-group comparison in each panel and conjunctive aggregation.
- anchors: 0 = Single quantity / single panel direction judgment.; 2 = Two or three quantities each compared in its own panel for direction.; 4 = Six or more quantities, each in a separate panel, each needing an independent group-pair direction judgment that must all hold for the claim.
- exemplars: high = scev_val_fig_0286, scev_val_fig_0287; low = sciver_test_0860, scev_val_fig_0218
- rationale: cross_panel_synthesis measures combining readings and claim_length_atomic_assertions caps the assertion count, but neither isolates the conjunctive load of judging a direction-of-change per panel across 7-8 panels where every sub-direction must hold.
- **load**: 5 items scored >=2 of 110 scored (mean 0.13, variance 0.38, nonzero 5)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### interval_endpoint_growth_computation  [relational]
- origin: emergent
- first_seen: item 151 (sciver_test_0821)
- definition: Whether the claim states a percent/relative change between two endpoint values that the verifier must read at both ends and compute, not merely a level or a single difference.
- anchors: 0 = no relative-change figure; single value or level comparison; 2 = claim states a simple difference between two readable endpoints (e.g. 'rises by 0.2'); 4 = claim states a percentage increase/ratio across two endpoints, requiring reading both and computing (high-low)/low
- exemplars: high = sciver_test_0821; low = scev_val_fig_0276
- rationale: quantitative_reasoning is generic; this isolates the common endpoint-pair percent-growth pattern that requires both extreme readings plus a normalized ratio, a frequent VLM arithmetic slip.
- **load**: 6 items scored >=2 of 110 scored (mean 0.15, variance 0.46, nonzero 6)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### scatter_centroid_separation_estimation  [relational]
- origin: emergent
- first_seen: item 154 (sciver_val_0397)
- definition: Whether verification requires estimating the spatial distance between the centroids of two unlabeled point clusters and comparing it across panels or to plot dimensions.
- anchors: 0 = no cluster-distance judgment; clusters labeled or single value read; 2 = judge whether two clusters are separated or overlapping qualitatively; 4 = estimate centroid-to-centroid distance (as fraction of plot width) and ratio it against another panel's separation
- exemplars: high = sciver_val_0397; low = sciver_test_0184
- rationale: cluster_count_gestalt covers counting/separated-vs-merged judgments, but not estimating a metric distance between centroids and ratioing it across panels, a quantitative gestalt distinct from counting.
- **load**: 3 items scored >=2 of 110 scored (mean 0.09, variance 0.32, nonzero 3)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### size_encoded_magnitude_readout  [relational]
- origin: emergent
- first_seen: item 166 (sciver_test_0081)
- definition: Whether a claim-relevant quantity is encoded by mark SIZE/area (bubble radius) requiring inversion of a size-to-value legend rather than a position-to-axis reading.
- anchors: 0 = no size encoding; all values from position against an axis; 2 = size legend present and a coarse larger-vs-smaller judgment suffices; 4 = claim's number requires mapping a specific bubble area back to a discrete size-legend scale (e.g. 'eight publications' from bubble area)
- exemplars: high = sciver_test_0081; low = sciver_test_0546, sciver_val_0178
- rationale: readout_precision and colormap_value_readout assume position- or hue-to-value; area-encoded magnitude (bubble size keyed to a stepped size legend) is a perceptually distinct and VLM-weak inversion not covered by any existing dimension.
- **load**: 0 items scored >=2 of 110 scored (mean 0.0, variance 0.0, nonzero 0)

### printed_value_claim_conflict_potential  [relational]
- origin: emergent
- first_seen: item 170 (sciver_test_0360)
- definition: Whether the claim cites an exact number for a quantity that is also printed verbatim in the figure, so verification reduces to checking the claimed digits against the printed digits at the right location.
- anchors: 0 = no printed value for the claimed quantity; must estimate from marks; 2 = a printed value exists but the claim only references it approximately; 4 = claim states an exact number for a quantity printed in-figure, so the act is locating the printed token and comparing digit-for-digit
- exemplars: high = sciver_test_0233, sciver_test_0360; low = sciver_val_0151, scev_val_fig_0250
- rationale: in_figure_answer_printing and printed_data_label_reliance note that values are printed but not the relational act of matching a claimed exact figure against the printed one at the correct labeled location, where a single wrong digit (0.856 vs printed 0.946) decides verification.
- **load**: 17 items scored >=2 of 110 scored (mean 0.52, variance 1.56, nonzero 18)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### subset_minimum_then_global_contrast  [relational]
- origin: emergent
- first_seen: item 173 (sciver_test_0612)
- definition: Whether the claim requires finding an extremum within a restricted sub-region AND a separate extremum over the full set, then differencing or comparing the two.
- anchors: 0 = Claim needs a single extremum over one undivided set.; 2 = Claim needs two extrema but over the same set (max and min).; 4 = Claim needs one extremum within a demarcated sub-region and another over the whole figure, then their difference/relation.
- exemplars: high = sciver_test_0612; low = scev_val_fig_0188, sciver_val_0349
- rationale: exhaustive_set_extremum_demand and subgroup_partition_selection each cover one half; neither captures the compound act of computing a within-subset minimum and a global minimum and contrasting them (81 uJ difference between the 5-QW min and the all-fields min).
- **load**: 0 items scored >=2 of 110 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### filled_area_coverage_estimation  [relational]
- origin: emergent
- first_seen: item 178 (sciver_val_0073)
- definition: Whether verification requires estimating the spatial extent/fraction of a filled or colored region (how much of a grid/cell space is occupied) rather than reading a value at a point.
- anchors: 0 = No area/coverage judgment; values read from discrete marks.; 2 = Judge which of two regions is visibly more filled, coarse.; 4 = Compare the occupied-area fraction of two filled fields where the difference is slight ('reduced overall coverage').
- exemplars: high = sciver_val_0073; low = sciver_test_0612, scev_val_fig_0188
- rationale: colormap_value_readout and derived_aggregate_readout cover value/statistic readout, but estimating what proportion of a cell-grid is filled (archive coverage) is an area-extent gestalt distinct from reading hue or aggregating marks.
- **load**: 5 items scored >=2 of 110 scored (mean 0.13, variance 0.38, nonzero 5)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### claimed_extremum_value_offset  [relational]
- origin: emergent
- first_seen: item 190 (sciver_val_0384)
- definition: Whether the claim names the correct extremum/peak location but states a value offset from where the marks actually sit, so verification must read the value at the right position rather than trust the cited number.
- anchors: 0 = Claim cites no value, only a location or ordering.; 2 = Claim cites a value at a named point within one tick of the mark's apparent position.; 4 = Claim cites a value at the correct point that is visibly more than one tick off the mark (e.g. '83.0%' where the peak sits near 82.4%), so the verifier must read the actual height not the asserted one.
- exemplars: high = sciver_val_0384, sciver_test_0347; low = sciver_val_0485, sciver_test_0130
- rationale: near_tie_discrimination measures value proximity among marks and printed_value_claim_conflict_potential assumes an in-figure printed number; neither captures the act of reading an estimated mark height at a correctly-named point and comparing it to a claimed value with no printed token to check against.
- **load**: 13 items scored >=2 of 50 scored (mean 0.58, variance 0.96, nonzero 14)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### event_peak_counting_demand  [relational]
- origin: emergent
- first_seen: item 192 (sciver_test_0506)
- definition: Whether verification requires counting the number of discrete peaks/spikes in a signal or spike-train trace and comparing that count, rather than reading any value.
- anchors: 0 = No peak counting; value or trend read from smooth marks.; 2 = Count a small number (<10) of clearly separated peaks in one trace.; 4 = Count ~30+ spikes in a dense spike train and compare the count to another trace's spike count.
- exemplars: high = sciver_test_0506; low = sciver_test_0372
- rationale: Counting spikes in a dense impulse trace is an enumeration-over-time act distinct from data_elements_to_integrate (which counts marks to read, not events to tally) and from histogram_bin_localization (which targets a single named bin).
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### normalized_axis_ratio_recovery  [relational]
- origin: emergent
- first_seen: item 201 (scev_val_fig_1509)
- definition: Whether the figure's axis is normalized to an arbitrary maximum (e.g. all bars scaled so the largest = 1) so a claimed absolute ratio/factor must be recovered by dividing two normalized marks rather than read directly.
- anchors: 0 = Axis shows real units; claimed ratio reads off labeled values directly.; 2 = Axis normalized but the claimed quantity is itself a within-figure proportion, so normalization is harmless.; 4 = Axis normalized to an unlabeled max=1; the claim states an absolute multiplicative factor (1.86x, 2.96x) recoverable only by ratioing two bar heights, with no axis units to anchor it.
- exemplars: high = scev_val_fig_1509; low = sciver_val_0204, scev_val_fig_0060
- rationale: axis_claim_unit_mismatch and claim_figure_scale_mismatch cover unit gaps, but neither isolates the case where the axis is deliberately normalized to 1 so only ratios (not values) are recoverable, a distinct demand to compute factors from dimensionless marks.
- **load**: 2 items scored >=2 of 50 scored (mean 0.16, variance 0.61, nonzero 2)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### combined_total_extremum_search  [relational]
- origin: emergent
- first_seen: item 210 (sciver_val_0204)
- definition: Whether the claim asserts that a derived total (sum of components) attains its maximum/minimum at a particular category, requiring the verifier to compute the composite at every category and find the extremum, not just read one.
- anchors: 0 = No total-extremum claim; single value or single-category total.; 2 = Composite total compared at two named categories only.; 4 = Claim asserts a composite (sum of stacked/multiple components) peaks/bottoms at one category among 5+, so the verifier must aggregate per category and rank all composites.
- exemplars: high = sciver_val_0204; low = sciver_val_0128, scev_val_fig_1509
- rationale: category_total_aggregation_demand covers obtaining one grand total and exhaustive_set_extremum_demand covers ruling out alternatives, but neither isolates the compound act of computing a summed composite per category and then finding which category maximizes it.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### log_decade_gap_counting  [relational]
- origin: emergent
- first_seen: item 220 (sciver_val_0037)
- definition: Whether verification requires quantifying the separation between two series as a number of log-axis decades (orders of magnitude) by counting gridline decades, rather than reading a linear difference.
- anchors: 0 = No log axis, or only a same-scale linear difference is needed.; 2 = Log axis present and a coarse 'one series is higher' judgment suffices, no decade count.; 4 = Claim states an order-of-magnitude separation (e.g. '~10^9 larger') that must be confirmed by counting decade gridlines between two curves on a log axis.
- exemplars: high = sciver_val_0037; low = sciver_test_0260, sciver_val_0432
- rationale: axis_complexity flags log scales and claim_value_axis_unreadable_scale flags unreadable precision, but neither isolates the act of translating a vertical gap between two curves into a decade count, the crux of order-of-magnitude separation claims.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### total_overplot_coincidence_judgment  [relational]
- origin: emergent
- first_seen: item 215 (sciver_val_0432)
- definition: Whether multiple legend-distinct series are so coincident that only the last-drawn one is visible, so verifying a claim about them all requires inferring whether they truly overlap rather than reading separate marks.
- anchors: 0 = Each claim-relevant series is individually visible and separable.; 2 = Series partially overlap but each can be traced over part of its range.; 4 = Several legend-listed series are fully overplotted into a single visible line, so a claim about all of them ('all five identical/intersect') can only be judged as coincidence, not read per-series.
- exemplars: high = sciver_val_0432; low = sciver_test_0260, sciver_val_0037
- rationale: visual_density_clutter and marks_clipped_occluded cover overlap and occlusion of distinct marks, but not the specific case where N labeled series collapse to one visible curve, making per-series verification impossible and 'all identical' trivially plausible.
- **load**: 3 items scored >=2 of 50 scored (mean 0.18, variance 0.27, nonzero 6)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### inverted_axis_direction_tracking  [figure-intrinsic]
- origin: emergent
- first_seen: item 219 (sciver_val_0251)
- definition: Whether a claim-relevant axis runs in a non-default direction (values increasing downward or right-to-left) so that 'higher/lower/better' must be re-mapped against the inverted scale.
- anchors: 0 = All axes run in conventional direction (values increase up/right).; 2 = An axis is inverted but clearly tick-labeled so direction is unambiguous on inspection.; 4 = A claim about a maximum/minimum or 'lower is better' depends on an inverted axis (e.g. perplexity increasing downward) that, if read conventionally, flips the verdict.
- exemplars: high = sciver_val_0251; low = sciver_test_0260, sciver_val_0432
- rationale: axis_complexity lists inversion as one feature among many, but the specific perceptual trap of an inverted direction silently flipping a min/max or better/worse judgment is a distinct VLM failure worth isolating from general non-default-axis handling.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)

### exact_statistic_without_printed_value  [relational]
- origin: emergent
- first_seen: item 224 (sciver_val_0395)
- definition: Whether the claim cites a multi-decimal statistic (correlation, mean, percentage) that is neither printed in the figure nor readable to the stated precision from any axis, so the figure can at most corroborate the rough magnitude.
- anchors: 0 = Claimed precise statistic is printed in-figure or directly readable against fine axis ticks.; 2 = Claimed statistic is approximate and a coarse axis/gestalt reading can confirm its order of magnitude.; 4 = Claim gives a 3-4 significant-figure statistic (e.g. r=0.9100) with no printed value and only a point cloud or coarse axis to judge against, so the exact digits are physically unreadable.
- exemplars: high = sciver_val_0395; low = sciver_test_0529, sciver_test_0759
- rationale: readout_precision caps at 'multiple exact values read and combined' assuming the values are readable; this isolates the case where a stated four-digit statistic has no figure source at the claimed precision, a distinct false-grounding risk.
- **load**: 25 items scored >=2 of 50 scored (mean 1.6, variance 2.88, nonzero 26)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### claim_panel_number_vs_caption_letter  [relational]
- origin: emergent
- first_seen: item 229 (scev_val_fig_0029)
- definition: Whether the claim references a figure by a panel/figure number that differs from the caption's own labeling (e.g. claim says 'Fig 5(a)' but caption labels the supplied subplot '(a)'), forcing the verifier to reconcile the citation before locating evidence.
- anchors: 0 = Claim's panel/figure reference matches the caption and in-figure labels verbatim.; 2 = Claim adds or drops a figure number but the sub-panel letter still uniquely identifies the target.; 4 = Claim cites a figure/panel number absent from the caption (e.g. 'Fig 5(a)' vs caption '(a)') so the verifier must assume the intended panel.
- exemplars: high = scev_val_fig_0029; low = sciver_test_0529, sciver_val_0395
- rationale: claim_figure_referent_mismatch covers entities missing/renamed and referenced_panel_absent_from_render covers an omitted panel; neither isolates a numbering offset between claim citation and caption label that still points to a present panel.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### horizontal_reference_compliance_check  [relational]
- origin: emergent
- first_seen: item 240 (sciver_val_0266)
- definition: Whether verification requires confirming that every member of a series stays on one side of a fixed horizontal reference value (e.g. all lines below y=0) rather than reading any specific value.
- anchors: 0 = No reference-side test; a value or trend is read.; 2 = Confirm one curve stays above/below a drawn or implicit reference line over its range.; 4 = Confirm that EVERY member of a multi-line set stays on one side of a horizontal reference (e.g. PCHG<0) across all x, in an overplotted cloud where near-crossings must be scrutinized.
- exemplars: high = sciver_val_0266; low = sciver_val_0470, sciver_test_0576
- rationale: diagonal_reference_conformance covers whole-curve shape conformance and threshold_crossing_localization covers finding one crossing coordinate; neither isolates the universal sign/side test across many lines against a fixed horizontal value where a single excursion falsifies the claim.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### normalized_axis_absolute_claim_gap  [relational]
- origin: emergent
- first_seen: item 233 (scev_val_fig_0206)
- definition: Whether the figure's axis is normalized/relative (max scaled to 1) while the claim states an absolute multiplicative factor or count that cannot be confirmed from the normalized heights alone.
- anchors: 0 = Axis carries real units and the claim's quantity is directly readable against them.; 2 = Axis normalized but the claim only asserts a relative ordering the normalized bars already show.; 4 = Axis normalized to 1 and the claim asserts a specific absolute factor/count (e.g. '9.5x parameter reduction') that the normalized bar ratio cannot pin to that precision.
- exemplars: high = scev_val_fig_0206; low = sciver_test_0576, sciver_val_0470
- rationale: claim_value_axis_unreadable_scale covers log-axis precision loss and axis_claim_unit_mismatch covers named-unit gaps; this isolates the distinct case where a normalized (unitless, max=1) axis makes an absolute multiplicative claim unverifiable even though the bars are perfectly legible.
- **load**: 3 items scored >=2 of 50 scored (mean 0.2, variance 0.68, nonzero 3)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable); LEAKAGE-SUSPECT (scoring overlaps verdict; verify twin-invariance before regression use)

### upset_intersection_row_reading  [figure-intrinsic]
- origin: emergent
- first_seen: item 238 (sciver_test_0079)
- definition: Whether verification requires decoding an UpSet plot's dot-matrix rows (which sets are filled/connected) and matching them to the corresponding intersection-size bar to read a set-combination count.
- anchors: 0 = No UpSet/set-matrix encoding present.; 2 = An UpSet matrix present but the claim needs only the largest or a labeled bar.; 4 = Claim needs a specific intersection (e.g. the single-set-only row) found by reading the dot pattern across many rows and matching it to its size bar at small render size.
- exemplars: high = sciver_test_0079; low = sciver_test_0226, scev_val_fig_0206
- rationale: venn_region_arithmetic covers set-diagram containment but UpSet plots use a dot-matrix-plus-bar grammar (read which sets a row activates, then its count) that is a distinct encoding-literacy and row-matching demand not captured by any existing dimension.
- **load**: 0 items scored >=2 of 50 scored (mean 0.0, variance 0.0, nonzero 0)
- **flags**: DEEP-REASONING (cheap annotator may be unreliable)

### convergence_epoch_localization  [relational]
- origin: emergent
- first_seen: item 247 (sciver_test_0121)
- definition: Whether verification requires identifying the x-position at which a noisy decreasing curve first flattens to a stable plateau (its 'convergence' point) and reading that x-value, rather than reading a y-value or crossing of a drawn line.
- anchors: 0 = No convergence/plateau judgment; a value or monotone trend is read directly.; 2 = One curve's flattening point estimated against a clearly drawn plateau reference (e.g. a dashed asymptote line).; 4 = Two or more noisy curves' settling epochs must each be located by judging where oscillation stops, with no marked plateau line, then compared.
- exemplars: high = sciver_test_0121; low = sciver_test_0828, sciver_val_0369
- rationale: threshold_crossing_localization targets where a curve meets an explicit reference line; this isolates the noisier, looser act of deciding where a decaying training curve has 'converged', a settling-point judgment on oscillating data with no crisp crossing.
- **load**: not tallied

### area_normalized_distribution_reading  [figure-intrinsic]
- origin: emergent
- first_seen: item 242 (sciver_val_0475)
- definition: Whether the figure's y-axis is an area-normalized (density) distribution so absolute heights are arbitrary and only shape/relative differences are meaningful, which the verifier must recognize before interpreting any height or percent-deviation claim.
- anchors: 0 = Y-axis carries real counts/values; heights are directly interpretable.; 2 = Distribution normalized but the claim only concerns relative shape the normalization preserves.; 4 = Y-axis area-normalized (caption: 'area normalized') and the claim cites a percent excess/deficit that must be read from a separate ratio panel, since the main panel's heights encode density not rate.
- exemplars: high = sciver_val_0475; low = sciver_test_0828, sciver_val_0369
- rationale: normalized_axis_ratio_recovery and normalized_axis_absolute_claim_gap concern bars normalized to max=1; this captures area-normalized continuous distributions where the deviation claim must be read off an auxiliary ratio-to-baseline panel rather than the main density curve, a distinct encoding-recognition demand.
- **load**: not tallied

### convergence_plateau_judgment  [relational]
- origin: emergent
- first_seen: item 260 (scev_val_fig_0065)
- definition: Whether verification requires judging that a trajectory/learning curve settles to a stable plateau (or low-regret asymptote), distinct from reading its level or overall direction.
- anchors: 0 = No convergence judgment; a value, level, or monotone direction is read.; 2 = Judge whether one curve flattens to a stable value over its range.; 4 = Judge convergence-to-asymptote for multiple curves and/or in multiple regimes (e.g. before and after an event point) where transient spikes must be distinguished from failure to settle.
- exemplars: high = scev_val_fig_0065; low = scev_val_fig_0033, sciver_val_0376
- rationale: shape_convexity_curvature_judgment and diagonal_reference_conformance cover curve shape, but neither isolates the asymptotic-settling judgment central to 'converges to optimal' claims, where transient spikes vs a stable plateau decide the verdict.
- **load**: not tallied

### regime_split_event_partition  [relational]
- origin: emergent
- first_seen: item 260 (scev_val_fig_0065)
- definition: Whether the claim requires partitioning a single x-axis (often time/trajectory) at an unmarked event point and verifying a property separately in the before and after segments.
- anchors: 0 = No temporal/event partition; the whole x-range is judged as one.; 2 = Claim distinguishes early vs late behavior at a roughly indicated point.; 4 = Claim requires a before/after-event split where the change point is not drawn on the axis and the same property must hold in each segment.
- exemplars: high = scev_val_fig_0065; low = scev_val_fig_0033, sciver_val_0390
- rationale: threshold_crossing_localization finds where a curve meets a drawn line; this captures partitioning the domain at an unplotted event ('before and after the change') and verifying a property in each half, a distinct segmentation demand.
- **load**: not tallied

### categorical_proportion_summation  [relational]
- origin: emergent
- first_seen: item 257 (scev_val_fig_0249)
- definition: Whether the claimed quantity is the sum of two or more printed proportion/percentage labels (pie slices, variance-explained components) that must be located and added.
- anchors: 0 = Claim value equals a single printed proportion read directly.; 2 = Claim compares two printed proportions without summing them.; 4 = Claim's value is the additive total of 2+ printed percentage labels (e.g. PC1%+PC2%, or two pie slice shares) that must each be located and summed.
- exemplars: high = scev_val_fig_0249; low = sciver_val_0215, sciver_val_0376
- rationale: category_total_aggregation_demand covers summing bar segments and derived_aggregate_readout covers axis-estimated aggregates; neither isolates summing two printed percentage labels (28.61%+15.15%=43.76%) where each addend is OCR-read and the total appears nowhere.
- **load**: not tallied

### claim_named_entity_to_unlabeled_mark_binding  [relational]
- origin: emergent
- first_seen: item 261 (?)
- definition: Whether the claim identifies its target by a named model/method/condition that does not appear as a label on the figure, so the verifier must infer which unlabeled mark corresponds to that name via an external mapping (e.g. ordering, parameter value).
- anchors: 0 = Claim's named entity appears verbatim as an axis/legend/point label in the figure.; 2 = Claim's entity resolvable to a mark via a stated property already on an axis (e.g. claim names a GFLOP value that matches an x-tick).; 4 = Claim names an entity (e.g. 'B0', 'B2') absent from every figure label and the binding to a specific point requires external knowledge of which point that name occupies.
- exemplars: high = sciver_val_0042; low = sciver_val_0241, scev_val_fig_0144
- rationale: claim_figure_label_mismatch covers lexical aliasing between claim and present labels, but here the named entity has no figure label at all and must be located by inferred property; a distinct grounding step before any readout.
- **load**: not tallied

### claim_coordinate_granularity_mismatch  [relational]
- origin: emergent
- first_seen: item 276 (scev_val_fig_0279)
- definition: Whether the claim references a position/coordinate at a finer or different granularity than the figure indexes (e.g. claim cites a nucleotide base-pair position while the figure is indexed only by amino-acid residue), so the cited position cannot be located directly on the figure's axis/columns.
- anchors: 0 = Claim's position index matches the figure's own indexing units (same residue/column/coordinate system).; 2 = Claim's position is a coarser aggregate of the figure's index (claim cites a region, figure shows individual positions) but maps unambiguously.; 4 = Claim cites a position in a unit the figure does not index at all (nucleotide bp 280 vs amino-acid residue columns), so the verifier must convert coordinate systems (codon-to-residue) before locating it.
- exemplars: high = scev_val_fig_0279; low = scev_val_fig_0058, sciver_test_0721
- rationale: axis_claim_unit_mismatch and claim_figure_scale_mismatch concern the VALUE's units/scale; this isolates the orthogonal case where the claim's POSITION/index lives in a coordinate system the figure does not enumerate, forcing a coordinate-system conversion before any lookup is possible.
- **load**: not tallied

### compressed_high_value_band_discrimination  [figure-intrinsic]
- origin: emergent
- first_seen: item 290 (sciver_val_0255)
- definition: Whether all claim-relevant marks are crowded into a narrow band near one end of a full-range axis (e.g. 88-100% on a 0-100 axis) so small claimed differences occupy few pixels.
- anchors: 0 = Claim-relevant marks span a wide fraction of the axis range, differences clearly resolvable.; 2 = Relevant marks occupy roughly the top/bottom quarter of the axis but the claimed difference still spans several ticks.; 4 = All relevant marks lie within ~10% of a full-range axis so a claimed several-point difference is compressed to a few pixels of bar-height difference.
- exemplars: high = sciver_val_0255, sciver_val_0423; low = sciver_test_0421
- rationale: near_tie_discrimination measures data-space proximity of values, but this isolates the figure-intrinsic case where a full-range axis (0-100) compresses an inherently large claimed gap into near-illegible pixel height because the data lives only at the extreme.
- **load**: not tallied

### colorbar_sign_range_mismatch  [relational]
- origin: emergent
- first_seen: item 289 (sciver_test_0124)
- definition: Whether the claim's stated numeric values fall outside the polarity or range of the figure's colorbar/legend scale, so the cited values cannot be located on the displayed scale.
- anchors: 0 = Claimed values lie within the colorbar's displayed numeric range and sign.; 2 = Claimed values are at the colorbar's edge or require extrapolating one step beyond a labeled extreme.; 4 = Claimed values are of opposite sign or wholly outside the colorbar range (e.g. claim cites -0.5 to -1.3 while colorbar runs +1 to +3) so no displayed color maps to them.
- exemplars: high = sciver_test_0124; low = sciver_val_0255
- rationale: colormap_value_readout and claim_entity_axis_coverage assume the value is in range on a Cartesian axis; this captures a continuous colorbar whose displayed sign/range cannot represent the claimed values at all, a distinct false-grounding risk for color-mapped fields.
- **load**: not tallied

### unlabeled_radial_spoke_identification  [relational]
- origin: emergent
- first_seen: item 283 (sciver_val_0416)
- definition: Whether the claim names a specific spoke/category of a radar chart that carries no axis label, so the target spoke cannot be located without external information.
- anchors: 0 = Each radar spoke is directly labeled and the claim's category is found by reading the label.; 2 = Spokes labeled but the claim uses a synonym requiring a mapping step.; 4 = Claim names a specific spoke (e.g. 'the Netflix app') but no spoke on the radar is labeled, so the target axis cannot be identified from the figure.
- exemplars: high = sciver_val_0416; low = sciver_test_0005
- rationale: radial_encoding_literacy covers reading values along spokes and label_indirection covers lookup-path length, but neither captures a radar plot whose spokes bear no category labels at all, making a per-spoke claim ungroundable regardless of decoding skill.
- **load**: not tallied

### fitted_trend_vs_raw_point_selection  [relational]
- origin: emergent
- first_seen: item 300 (sciver_test_0127)
- definition: Whether the claim's endpoint/value should be read from an overlaid fitted regression line rather than the scattered raw data points, requiring the verifier to choose which graphical layer the claimed number refers to.
- anchors: 0 = Only one graphical layer present (raw marks or a single line); no ambiguity about source.; 2 = Both raw points and a fit line present, but they nearly coincide so the choice is immaterial.; 4 = Raw points and a fitted/smoothed line diverge, and the claimed value matches only one layer, so the verifier must decide whether to read the fit or the scatter.
- exemplars: high = sciver_test_0127; low = sciver_val_0336, scev_val_fig_0205
- rationale: When a figure overlays raw daily means and a regression line with confidence band, the claimed 3.2-to-3.6 endpoints can be read from either layer with different results; no existing dimension captures the demand of selecting the correct data layer before readout.
- **load**: not tallied

### axis_chrome_legibility_failure  [figure-intrinsic]
- origin: emergent
- first_seen: item 298 (sciver_test_0988)
- definition: Whether axis tick labels, axis titles, or scale numbers are rendered illegibly (low contrast against background, cropped, or absent) so the plotted marks cannot be grounded to numeric values at all.
- anchors: 0 = All axis ticks and titles crisp and high-contrast against the plot background.; 2 = Axis text small or faint but readable on close inspection.; 4 = Axis tick numbers and titles are unreadable (e.g. dark text on a black background, or no tick labels rendered) so the value scale is not recoverable from the image.
- exemplars: high = sciver_test_0988; low = sciver_val_0336, scev_val_fig_1519
- rationale: small_text_legibility_demand and small_figure_resolution_strain concern overall pixel strain; this isolates the specific case where the axis chrome (not data marks) is illegible because of contrast/cropping, making any value readout ungroundable independent of mark clarity. Distinct from unlabeled_axis_value_ungroundable, where ticks are simply absent by design rather than rendered-but-unreadable.
- **load**: not tallied

---

## Leakage-suspect flags

(dimensions that scored twin-pair members differently, with reasons)

- scev_val_fig_0288 / scev_val_fig_0290 (batch 06): readout_precision, quantitative_reasoning, derived_aggregate_readout, printed_data_label_reliance, in_figure_answer_printing, quantifier_strength, negation_present, subgroup_partition_selection, label_value_consistency_check — These are NOT leakage-suspect: the two items share the same figure but pose DIFFERENT claims targeting different sub-panels. 0288's claim concerns panel A (no-significant-difference, p-value brackets, contains a negation), while 0290's claim concerns panel C (Venn OTU count comparison, demanding region summation). The differing loadings reflect the different claims/panels, not the figure modification, so this is expected divergence rather than a verdict leak. Dimensions tied to the shared figure-intrinsic structure (panel_count_layout, visual_density_clutter, small_text_legibility_demand) were scored consistently.

---

## Batch log

(one line per completed batch: batch index, item index range, count of new
emergent dimensions, notes)
- batch 00: items 1-10, 5 new dims (mechanistic_causal_overreach, claim_entity_axis_coverage, rotated_microlabel_decoding, panel_localization_among_many, in_figure_answer_printing)
- batch 01: items 11-20, 6 new dims (colormap_value_readout, spatial_roi_localization, printed_data_label_reliance, claim_figure_label_mismatch, subgroup_partition_selection, small_render_legibility)
- batch 02: items 21-30, 5 new dims (stacked_segment_decomposition, claim_figure_scale_mismatch, threshold_crossing_localization, matrix_axis_orientation_demand, target_panel_localization)
- batch 03: items 31-40, 6 new dims (axis_claim_unit_mismatch, derived_aggregate_readout, color_axis_decoding, radial_encoding_literacy, small_figure_resolution_strain, marker_shape_keying)
- batch 04: items 41-50, 5 new dims (claim_figure_scope_overreach, label_value_consistency_check, small_text_legibility_demand, panel_localization_demand, text_entity_matching_demand)
- batch 05: items 51-60, 5 new dims (mechanism_attribution_demand, claim_figure_referent_mismatch, annotation_overlay_counting, label_rotation_legibility, exhaustive_set_extremum_demand)
- batch 06: items 61-70, 3 new dims (venn_region_arithmetic, significance_bracket_pairing, encoded_variable_substitution)
- batch 07: items 71-80, 2 new dims (histogram_bin_localization_readout, cross_series_rate_comparison)
- batch 08: items 81-90, 3 new dims (enumerated_list_membership_check, shape_convexity_curvature_judgment, diagonal_reference_conformance)
- batch 09: items 91-100, 4 new dims (rank_position_verification, cluster_count_gestalt, external_reference_dependency, caption_axis_quantity_conflict)
- batch 10: items 101-110, 3 new dims (slope_estimation_from_line, referenced_panel_absent_from_render, gap_trend_comparison)
- batch 11: items 111-120, 2 new dims (contour_overlay_extent_reading, significance_marker_reading)
- batch 12: items 121-130, 3 new dims (claim_value_axis_unreadable_scale, embedded_subplot_value_extraction, category_total_aggregation_demand)
- batch 13: items 131-140, 4 new dims (ordinal_step_to_continuous_mapping, legend_text_value_substitution, parametric_form_conformance, unspecified_panel_disambiguation)
- batch 14: items 141-150, 3 new dims (unlabeled_axis_value_ungroundable, category_state_attribution, multi_panel_directional_aggregation)
- batch 15: items 151-160, 2 new dims (interval_endpoint_growth_computation, scatter_centroid_separation_estimation)
- batch 16: items 161-170, 2 new dims (size_encoded_magnitude_readout, printed_value_claim_conflict_potential)
- batch 17: items 171-180, 2 new dims (subset_minimum_then_global_contrast, filled_area_coverage_estimation)
- batch 18: items 181-190, 1 new dims (claimed_extremum_value_offset)
- batch 19: items 191-200, 1 new dims (event_peak_counting_demand)
- batch 20: items 201-210, 2 new dims (normalized_axis_ratio_recovery, combined_total_extremum_search)
- batch 21: items 211-220, 3 new dims (log_decade_gap_counting, total_overplot_coincidence_judgment, inverted_axis_direction_tracking)
- batch 22: items 221-230, 2 new dims (exact_statistic_without_printed_value, claim_panel_number_vs_caption_letter)
- batch 23: items 231-240, 3 new dims (horizontal_reference_compliance_check, normalized_axis_absolute_claim_gap, upset_intersection_row_reading)
- batch 24: items 241-250, 2 new dims (convergence_epoch_localization, area_normalized_distribution_reading)
- batch 25: items 251-260, 3 new dims (convergence_plateau_judgment, regime_split_event_partition, categorical_proportion_summation)
- batch 26: items 261-270, 1 new dims (claim_named_entity_to_unlabeled_mark_binding)
- batch 27: items 271-280, 1 new dims (claim_coordinate_granularity_mismatch)
- batch 28: items 281-290, 3 new dims (compressed_high_value_band_discrimination, colorbar_sign_range_mismatch, unlabeled_radial_spoke_identification)
- batch 29: items 291-300, 2 new dims (fitted_trend_vs_raw_point_selection, axis_chrome_legibility_failure)

---

## Discovery summary

### Saturation: NOT reached
- 89 emergent dimensions were proposed across the full 300-item sample (the
  hard cap), on top of 23 seeded dimensions (112 total).
- The stopping criterion (fewer than 2 new emergent dimensions over 50
  consecutive items) was never met. The 50-item trailing window fell from 27
  new dimensions (items 1-50) to a plateau of 8-11 from item ~190 onward; the
  final window (items 251-300) still produced **10** new dimensions:
  convergence_plateau_judgment, regime_split_event_partition,
  categorical_proportion_summation, claim_named_entity_to_unlabeled_mark_binding,
  claim_coordinate_granularity_mismatch, compressed_high_value_band_discrimination,
  colorbar_sign_range_mismatch, unlabeled_radial_spoke_identification,
  fitted_trend_vs_raw_point_selection, axis_chrome_legibility_failure.
- Interpretation: the curve flattened but did not converge. The residual
  ~10-per-50 rate is dominated by (a) ever-finer chart-type-specific readout
  idioms (UpSet rows, Venn regions, radar spokes, contour extents) and (b)
  the leakage-suspect "claim-vs-figure discrepancy" family, which keeps
  splintering into new surface forms. **The sample should be extended, OR —
  more likely the right move — the codebook should first be consolidated**:
  many of the 89 emergent dimensions are near-synonyms (see redundancy note)
  and collapsing them will materially lower the apparent new-dimension rate.

### Where signal concentrated (high load, broad support)
- Seeded relational/integration dimensions dominate: caption_sufficiency
  (300/300 >=2), data_elements_to_integrate (296/300), claim_length_atomic_
  assertions (292/300), attention_scan_demand (295/300), readout_precision
  (279/300), numeric_specificity (216/300, highest variance 4.44),
  lookup_vs_gestalt (274/300). These are the backbone and behave as expected
  from FUGU (readout precision, integration load, lookup-vs-gestalt all carry
  spread).
- The strongest emergent additions by load are the VLM-handoff legibility
  cluster (small_text_legibility_demand, small_figure_resolution_strain,
  small_render_legibility — all ~120-130 of 210) and subgroup_partition_
  selection (151/210). These tap exactly the vision-encoder failure modes the
  design brief anticipated (small/clipped/low-contrast elements) and are
  worth carrying forward.

### Seeded dimensions with near-zero variance in this sample
- Only **negation_present** is near-degenerate (variance 0.42, just 28/300
  items >=2): scientific claims here rarely use explicit negation. Candidate
  for dropping or merging into quantifier handling unless the scoring sample
  is expected to differ.
- All other seeded dimensions showed usable variance. Lowest-load but still
  variable: uncertainty_elements (75/300), axis_complexity (77/300),
  marks_clipped_occluded (105/300) — present enough to retain.

### Redundancy (for the consolidation phase, NOT done here)
- Three panel-localization dimensions are effectively one construct:
  panel_localization_among_many, target_panel_localization,
  panel_localization_demand (loads 90/99/99 of 210). Collapse to one.
- Three small-render legibility dimensions overlap heavily (see above);
  collapse to one figure-intrinsic "render legibility" with sub-anchors for
  text vs marks vs axis chrome (axis_chrome_legibility_failure,
  rotated_microlabel_decoding, label_rotation_legibility also belong here).
- A large "claim-vs-figure discrepancy" family (claim_figure_label_mismatch,
  _scale_mismatch, _referent_mismatch, _scope_overreach,
  referenced_panel_absent_from_render, printed_value_claim_conflict_potential,
  exact_statistic_without_printed_value, axis_claim_unit_mismatch,
  caption_axis_quantity_conflict, encoded_variable_substitution, the two
  mechanism-overreach dims, and others) are all flagged LEAKAGE-SUSPECT.

### Leakage analysis (twin-pair purity controls)
- 10 supported/refuted twin pairs were embedded (same claim, original vs
  modified figure), spread across Legend Swap (3), Category Swap (3), Graph
  Flip (2), Graph Swap (2). Annotators scored figures verdict-blind (labels
  and modification types were withheld from the annotating agents).
- Annotators scored every twin pair's two members with **identical** demand
  profiles; no seeded or emergent dimension diverged across a true twin pair.
  The one recorded twin note (batch 06) was a false alarm: those two items
  share a figure but pose different claims about different sub-panels, so the
  divergence is claim-driven, not figure-modification-driven.
- This is reassuring for the seeded codebook but does NOT clear the
  LEAKAGE-SUSPECT emergent dimensions. Those were *proposed* from refuted
  items (where the claim contradicts the figure); twin-invariance was not
  separately verified for each of them at the modified-figure member, because
  most twins did not happen to load on most of those dimensions. Every
  LEAKAGE-SUSPECT dimension must be re-tested for twin-invariance on a
  purpose-built set before use, and reformulated to score the *demand*
  (presence of a checkable discrepancy site) rather than the *outcome*
  (whether claim and figure actually disagree). Legend-swap twins are the
  sharpest risk: any dimension reading legend-color binding can flip with the
  swap while the verification act is unchanged.

### Dimensions likely to be unreliable for a cheap annotator
- 53 of 112 dimensions are flagged DEEP-REASONING (full list inline above).
  These split into: (a) multi-step quantitative inference (quantitative_
  reasoning, derived_aggregate_readout, slope/rate/interval computations,
  aggregation-then-extremum), and (b) the leakage-suspect discrepancy family,
  which additionally requires semantic claim-figure alignment judgment. Both
  groups need either tighter anchors or expensive-model annotation in the
  scoring phase; the cheap-annotator pass should be validated against a
  small expert-scored subset on these specifically.

### Idiosyncratic / low-yield emergent dimensions
- 9 emergent dimensions recorded zero load on every subsequent item after the
  one that prompted them (size_encoded_magnitude_readout,
  subset_minimum_then_global_contrast, event_peak_counting_demand,
  combined_total_extremum_search, log_decade_gap_counting,
  inverted_axis_direction_tracking, claim_panel_number_vs_caption_letter,
  horizontal_reference_compliance_check, upset_intersection_row_reading).
  These are single-figure-type idioms; candidates to drop or fold into a
  generic "chart-type-specific readout idiom" tag during consolidation.

### needs_context (text beyond figure+caption)
- Across 300 items: 114 "no", 137 "helpful", 49 "required". So ~16% of
  single-figure items plausibly cannot be verified from figure+caption alone
  (claim references off-figure mechanisms, missing panels, or external
  results). The manifest records the per-item judgment (needs_context_judged).

### Deliverables written (provenance confirmed)
- feature_discovery/sciver_singlefig_split.json — authoritative paper-grouped
  80/20 split over the 817-item SciVer single-figure pool (654 train / 163
  test, 0 paper overlap); 200 discovery items flagged
  used_in_feature_discovery=true (all on the train side).
- feature_discovery/sciclaimeval_split.json — authoritative paper-grouped
  80/20 split over the 265-item SciClaimEval figure pool (212 train / 53
  test, 0 paper overlap); 100 discovery items flagged
  used_in_feature_discovery=true (all on the train side).
- feature_discovery/discovery_items.csv — 300 rows (item_id, source,
  paper_id, figure_path).
- feature_discovery/manifest.csv — full per-item metadata incl. twin_id and
  needs_context_judged.
- feature_discovery/figures/ — 300 PNGs.
- feature_discovery/saturation_log.csv — 89 rows (item_index, dimension_name).
- feature_discovery/codebook.md — working codebook with per-batch log and
  twin notes; feature_discovery/batch_results/batch_00..29.json — raw tallies.
