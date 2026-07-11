"""Assemble candidate_features.md from codebook.md + aggregated stats.

Adds per dimension: load count (items scored >=2 / present denominator),
mean, variance, origin tag (already in codebook), and two review flags:
  - deep_reasoning: a cheap annotator likely applies it unreliably
  - leakage_suspect: scoring partly detects claim-figure discrepancy, which
    correlates with the verdict (must be checked before use in any regression)
"""
import json
import re
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
codebook = open(os.path.join(ROOT, "codebook.md")).read()
stats = json.load(open(os.path.join(ROOT, "_dim_stats.json")))
cat = json.load(open(os.path.join(ROOT, "_cat_stats.json")))

# Dimensions whose scoring requires comparing claim content to figure content
# (detecting a discrepancy is partly detecting refutation) -> leakage-suspect.
LEAKAGE = {
    "claim_figure_label_mismatch", "claim_figure_scale_mismatch",
    "claim_figure_referent_mismatch", "claim_figure_scope_overreach",
    "referenced_panel_absent_from_render", "printed_value_claim_conflict_potential",
    "claimed_extremum_value_offset", "axis_claim_unit_mismatch",
    "caption_axis_quantity_conflict", "encoded_variable_substitution",
    "claim_value_axis_unreadable_scale", "unlabeled_axis_value_ungroundable",
    "exact_statistic_without_printed_value", "claim_entity_axis_coverage",
    "label_value_consistency_check", "claim_panel_number_vs_caption_letter",
    "normalized_axis_absolute_claim_gap", "colorbar_sign_range_mismatch",
    "unlabeled_radial_spoke_identification", "claim_coordinate_granularity_mismatch",
    "claim_named_entity_to_unlabeled_mark_binding", "diagonal_reference_conformance",
    "horizontal_reference_compliance_check", "parametric_form_conformance",
    "mechanistic_causal_overreach", "mechanism_attribution_demand",
    "regime_split_event_partition",
}
# Dimensions requiring judgment/semantics/multi-step inference a weak model
# may apply unreliably (superset includes most LEAKAGE plus reasoning-heavy).
DEEP = set(LEAKAGE) | {
    "quantitative_reasoning", "derived_aggregate_readout",
    "near_tie_discrimination", "lookup_vs_gestalt", "domain_knowledge_demand",
    "caption_sufficiency", "subgroup_partition_selection",
    "category_state_attribution", "shape_convexity_curvature_judgment",
    "slope_estimation_from_line", "cross_series_rate_comparison",
    "interval_endpoint_growth_computation", "rank_position_verification",
    "cluster_count_gestalt", "external_reference_dependency",
    "multi_panel_directional_aggregation", "scatter_centroid_separation_estimation",
    "subset_minimum_then_global_contrast", "combined_total_extremum_search",
    "convergence_plateau_judgment", "convergence_epoch_localization",
    "categorical_proportion_summation", "fitted_trend_vs_raw_point_selection",
    "total_overplot_coincidence_judgment", "filled_area_coverage_estimation",
    "category_total_aggregation_demand", "normalized_axis_ratio_recovery",
    "area_normalized_distribution_reading", "claim_length_atomic_assertions",
    "enumerated_list_membership_check", "event_peak_counting_demand",
    "exhaustive_set_extremum_demand", "log_decade_gap_counting",
    "upset_intersection_row_reading", "venn_region_arithmetic",
}

CAT_DIMS = {"chart_type", "vlat_task_type"}


def metrics_line(dim):
    if dim in CAT_DIMS:
        c = cat.get(dim, {})
        top = sorted(c.items(), key=lambda x: -x[1])[:6]
        dist = ", ".join(f"{k}={v}" for k, v in top)
        return f"- **load**: categorical (300 items); top categories: {dist}"
    s = stats.get(dim)
    if not s:
        return "- **load**: not tallied"
    flags = []
    if dim in DEEP:
        flags.append("DEEP-REASONING (cheap annotator may be unreliable)")
    if dim in LEAKAGE:
        flags.append("LEAKAGE-SUSPECT (scoring overlaps verdict; verify "
                     "twin-invariance before regression use)")
    fl = ("\n- **flags**: " + "; ".join(flags)) if flags else ""
    return (f"- **load**: {s['ld2']} items scored >=2 of {s['present']} "
            f"scored (mean {s['mean']}, variance {s['var']}, "
            f"nonzero {s['nz']})" + fl)


# Insert a metrics line after each "- origin:" block's anchors line.
# Strategy: for every "### name" header, append metrics before the next "###"
# or section delimiter.
lines = codebook.split("\n")
out = []
i = 0
cur_dim = None
header_re = re.compile(r"^### (\w+)")
while i < len(lines):
    line = lines[i]
    m = header_re.match(line)
    if m:
        # flush metrics for previous dim if pending handled below
        cur_dim = m.group(1)
        out.append(line)
        i += 1
        # copy the dimension block until blank line preceding next header /
        # section, then inject metrics at end of block
        block = []
        while i < len(lines) and not header_re.match(lines[i]) and not lines[i].startswith("## ") and not lines[i].startswith("---"):
            block.append(lines[i])
            i += 1
        # remove trailing blank lines in block
        while block and block[-1].strip() == "":
            block.pop()
        out.extend(block)
        out.append(metrics_line(cur_dim))
        out.append("")
    else:
        out.append(line)
        i += 1

body = "\n".join(out)

header = """# candidate_features.md — figure-verification demand codebook (discovery output)

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
- LEAKAGE-SUSPECT: scoring partly amounts to detecting a discrepancy between
  claim and figure, which correlates with the verdict. Most emergent
  "...mismatch / ...conflict / ...overreach / referent / ungroundable"
  dimensions are leakage-suspect BY CONSTRUCTION: they were proposed from
  refuted (modified-figure) items where the claim contradicts the figure.
  Before any of these enters a difficulty regression, confirm it scores
  twin-pair members (supported original vs refuted modification) IDENTICALLY;
  if it does not, it is measuring the verdict, not a demand, and must be
  reformulated (e.g. "potential for printed-value/claim digit conflict",
  scored on presence of a checkable printed value, not on whether it
  matches) or discarded.

"""

open(os.path.join(ROOT, "candidate_features.md"), "w").write(header + body)
print("wrote candidate_features.md")
print("dimensions:", body.count("\n### "))
