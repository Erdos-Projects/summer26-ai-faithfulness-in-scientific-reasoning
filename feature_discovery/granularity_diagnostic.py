"""Collapse the 112 discovery sub-dimensions into 9 parent dimensions and run a presence-count
granularity diagnostic. Writes 3 CSVs into feature_discovery/. NOT formal scoring: this measures
0-vs-1+ presence (a lower bound, because the discovery log scored sub-dims only from their
first-sighting batch onward, never retroactively).

Outputs:
  dimension_mapping.csv      sub_dimension, parent_dimension, origin
  dimension_counts.csv       item_id, source, <9 parent counts>, tally_depth
  dimension_granularity.csv  parent summary incl. coverage-aware columns
"""
import csv
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBOOK = os.path.join(HERE, "codebook.md")
BATCH_GLOB = os.path.join(HERE, "batch_results", "batch_*.json")

# ---- 112 sub-dimension -> parent mapping (single best fit; controls/typologies -> unassigned) ----
PARENTS = [
    "localization_grounding", "value_readout", "referent_mismatch", "multi_element_reasoning",
    "gestalt_shape", "encoding_literacy", "legibility_render", "claim_linguistic_load",
    "scope_overreach",
]
MAPPING = {
    # --- localization_grounding: find the right panel/region/referent/mark before reading ---
    "panel_count_layout": "localization_grounding",
    "legend_complexity": "localization_grounding",
    "label_indirection": "localization_grounding",
    "attention_scan_demand": "localization_grounding",
    "panel_localization_among_many": "localization_grounding",
    "spatial_roi_localization": "localization_grounding",
    "subgroup_partition_selection": "localization_grounding",
    "target_panel_localization": "localization_grounding",
    "panel_localization_demand": "localization_grounding",
    "marker_shape_keying": "localization_grounding",
    "text_entity_matching_demand": "localization_grounding",
    "significance_bracket_pairing": "localization_grounding",
    "unspecified_panel_disambiguation": "localization_grounding",
    "category_state_attribution": "localization_grounding",
    "claim_named_entity_to_unlabeled_mark_binding": "localization_grounding",
    "unlabeled_radial_spoke_identification": "localization_grounding",
    "fitted_trend_vs_raw_point_selection": "localization_grounding",
    # --- value_readout: extract a value and by what mechanism ---
    "readout_precision": "value_readout",
    "near_tie_discrimination": "value_readout",
    "in_figure_answer_printing": "value_readout",
    "colormap_value_readout": "value_readout",
    "printed_data_label_reliance": "value_readout",
    "color_axis_decoding": "value_readout",
    "histogram_bin_localization_readout": "value_readout",
    "significance_marker_reading": "value_readout",
    "embedded_subplot_value_extraction": "value_readout",
    "legend_text_value_substitution": "value_readout",
    "size_encoded_magnitude_readout": "value_readout",
    "printed_value_claim_conflict_potential": "value_readout",
    "claimed_extremum_value_offset": "value_readout",
    # --- referent_mismatch (ISOLATED, leakage-bearing): claim referent not cleanly in figure ---
    "claim_entity_axis_coverage": "referent_mismatch",
    "claim_figure_label_mismatch": "referent_mismatch",
    "claim_figure_scale_mismatch": "referent_mismatch",
    "axis_claim_unit_mismatch": "referent_mismatch",
    "label_value_consistency_check": "referent_mismatch",
    "claim_figure_referent_mismatch": "referent_mismatch",
    "encoded_variable_substitution": "referent_mismatch",
    "caption_axis_quantity_conflict": "referent_mismatch",
    "referenced_panel_absent_from_render": "referent_mismatch",
    "claim_value_axis_unreadable_scale": "referent_mismatch",
    "unlabeled_axis_value_ungroundable": "referent_mismatch",
    "normalized_axis_ratio_recovery": "referent_mismatch",
    "claim_panel_number_vs_caption_letter": "referent_mismatch",
    "normalized_axis_absolute_claim_gap": "referent_mismatch",
    "claim_coordinate_granularity_mismatch": "referent_mismatch",
    "colorbar_sign_range_mismatch": "referent_mismatch",
    # --- multi_element_reasoning: computation/aggregation/ranking over multiple readings ---
    "data_elements_to_integrate": "multi_element_reasoning",
    "cross_panel_synthesis": "multi_element_reasoning",
    "quantitative_reasoning": "multi_element_reasoning",
    "derived_aggregate_readout": "multi_element_reasoning",
    "annotation_overlay_counting": "multi_element_reasoning",
    "exhaustive_set_extremum_demand": "multi_element_reasoning",
    "venn_region_arithmetic": "multi_element_reasoning",
    "rank_position_verification": "multi_element_reasoning",
    "category_total_aggregation_demand": "multi_element_reasoning",
    "interval_endpoint_growth_computation": "multi_element_reasoning",
    "subset_minimum_then_global_contrast": "multi_element_reasoning",
    "event_peak_counting_demand": "multi_element_reasoning",
    "combined_total_extremum_search": "multi_element_reasoning",
    "log_decade_gap_counting": "multi_element_reasoning",
    "categorical_proportion_summation": "multi_element_reasoning",
    # --- gestalt_shape: perceptual trend/shape/structure judgments ---
    "lookup_vs_gestalt": "gestalt_shape",
    "uncertainty_elements": "gestalt_shape",
    "threshold_crossing_localization": "gestalt_shape",
    "cross_series_rate_comparison": "gestalt_shape",
    "shape_convexity_curvature_judgment": "gestalt_shape",
    "diagonal_reference_conformance": "gestalt_shape",
    "cluster_count_gestalt": "gestalt_shape",
    "slope_estimation_from_line": "gestalt_shape",
    "gap_trend_comparison": "gestalt_shape",
    "contour_overlay_extent_reading": "gestalt_shape",
    "parametric_form_conformance": "gestalt_shape",
    "scatter_centroid_separation_estimation": "gestalt_shape",
    "filled_area_coverage_estimation": "gestalt_shape",
    "horizontal_reference_compliance_check": "gestalt_shape",
    "convergence_epoch_localization": "gestalt_shape",
    "area_normalized_distribution_reading": "gestalt_shape",
    "convergence_plateau_judgment": "gestalt_shape",
    "regime_split_event_partition": "gestalt_shape",
    # --- encoding_literacy: non-standard chart grammar ---
    "axis_complexity": "encoding_literacy",
    "encoding_convention_literacy": "encoding_literacy",
    "stacked_segment_decomposition": "encoding_literacy",
    "matrix_axis_orientation_demand": "encoding_literacy",
    "radial_encoding_literacy": "encoding_literacy",
    "ordinal_step_to_continuous_mapping": "encoding_literacy",
    "inverted_axis_direction_tracking": "encoding_literacy",
    "upset_intersection_row_reading": "encoding_literacy",
    # --- legibility_render: pixel-space legibility independent of data ---
    "visual_density_clutter": "legibility_render",
    "marks_clipped_occluded": "legibility_render",
    "rotated_microlabel_decoding": "legibility_render",
    "small_render_legibility": "legibility_render",
    "small_figure_resolution_strain": "legibility_render",
    "small_text_legibility_demand": "legibility_render",
    "label_rotation_legibility": "legibility_render",
    "compressed_high_value_band_discrimination": "legibility_render",
    "total_overplot_coincidence_judgment": "legibility_render",
    "axis_chrome_legibility_failure": "legibility_render",
    # --- claim_linguistic_load: claim parsing difficulty ---
    "negation_present": "claim_linguistic_load",
    "quantifier_strength": "claim_linguistic_load",
    "numeric_specificity": "claim_linguistic_load",
    "claim_length_atomic_assertions": "claim_linguistic_load",
    "enumerated_list_membership_check": "claim_linguistic_load",
    "multi_panel_directional_aggregation": "claim_linguistic_load",
    # --- scope_overreach: portion of claim unverifiable from any figure ---
    "mechanistic_causal_overreach": "scope_overreach",
    "mechanism_attribution_demand": "scope_overreach",
    "claim_figure_scope_overreach": "scope_overreach",
    "external_reference_dependency": "scope_overreach",
    "exact_statistic_without_printed_value": "scope_overreach",
    # --- unassigned: controls / typologies / knowledge axis with no single visual-demand home ---
    "chart_type": "unassigned",                 # categorical control variable (chart type)
    "vlat_task_type": "unassigned",             # categorical task typology spanning many parents
    "domain_knowledge_demand": "unassigned",    # field-knowledge axis; no knowledge parent exists
    "caption_sufficiency": "unassigned",        # caption-vs-figure control, not a visual demand
}


def codebook_origins():
    txt = open(CODEBOOK).read()
    pairs = re.findall(r"^###\s+(\S+).*?\n(?:.*\n)*?\s*-\s*origin:\s*(\w+)", txt, re.M)
    return {n: o for n, o in pairs}


def source_of(item_id):
    if item_id.startswith("sciver_"):
        return "sciver"
    if item_id.startswith("scev_") or item_id.startswith("sciclaim"):
        return "sciclaimeval"
    return "unknown"


def fired(value):
    """A sub-dim 'fired' if its recorded loading is >=1 (numeric) or a category is recorded (str)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 1
    if isinstance(value, str):
        s = value.strip()
        return bool(s) and s not in {"0", "none", "n/a", "na"}
    return False


def main():
    origins = codebook_origins()
    # validate: every codebook dim has a mapping, and vice-versa
    missing = sorted(set(origins) - set(MAPPING))
    extra = sorted(set(MAPPING) - set(origins))
    if missing:
        raise SystemExit(f"codebook dims with no mapping: {missing}")
    if extra:
        raise SystemExit(f"mapping has dims not in codebook: {extra}")

    # ---- STEP 1: mapping CSV ----
    with open(os.path.join(HERE, "dimension_mapping.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sub_dimension", "parent_dimension", "origin"])
        for d in sorted(MAPPING):
            w.writerow([d, MAPPING[d], origins.get(d, "?")])
    members = {p: [d for d, par in MAPPING.items() if par == p] for p in PARENTS}
    n_unassigned = sum(1 for v in MAPPING.values() if v == "unassigned")
    print(f"STEP 1: mapped {len(MAPPING)} sub-dims -> {len(PARENTS)} parents "
          f"({n_unassigned} unassigned). dimension_mapping.csv written.")

    # ---- STEP 2: per-item parent presence counts (lower-bound fire-counts) ----
    batches = sorted(glob.glob(BATCH_GLOB), key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group()))
    rows = []
    for bp in batches:
        for it in json.load(open(bp))["items"]:
            tal = it.get("tallies", {})
            row = {"item_id": it["item_id"], "source": source_of(it["item_id"]),
                   "tally_depth": len(tal)}
            for p in PARENTS:
                row[p] = sum(1 for d in members[p] if d in tal and fired(tal[d]))
            # coverage: how many of the parent's members were even scoreable (present in tally)
            row["_cov"] = {p: sum(1 for d in members[p] if d in tal) for p in PARENTS}
            rows.append(row)
    cols = ["item_id", "source"] + PARENTS + ["tally_depth"]
    with open(os.path.join(HERE, "dimension_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"STEP 2: {len(rows)} item rows -> dimension_counts.csv "
          f"(tally_depth column exposes per-item coverage; counts are LOWER-BOUND fire-counts).")

    # ---- STEP 3: per-parent granularity summary ----
    gran = []
    for p in PARENTS:
        counts = [r[p] for r in rows]
        cov_any = [1 if r["_cov"][p] >= 1 else 0 for r in rows]              # parent scoreable at all
        counts_where_cov = [r[p] for r in rows if r["_cov"][p] >= 1]         # restrict to scoreable items
        n_sub = len(members[p])
        n_seed = sum(1 for d in members[p] if origins.get(d) == "seeded")
        mean_c = statistics.mean(counts)
        gran.append({
            "parent_dimension": p,
            "n_subdims_in_parent": n_sub,
            "n_seeded": n_seed,
            "n_emergent": n_sub - n_seed,
            "mean_count": round(mean_c, 3),
            "sd_count": round(statistics.stdev(counts), 3),
            "min": min(counts),
            "max": max(counts),
            "frac_items_present": round(sum(c >= 1 for c in counts) / len(counts), 3),
            "norm_mean": round(mean_c / n_sub, 3) if n_sub else 0.0,
            "frac_items_scoreable": round(sum(cov_any) / len(cov_any), 3),
            "mean_count_where_scoreable": round(statistics.mean(counts_where_cov), 3) if counts_where_cov else 0.0,
        })
    gcols = list(gran[0].keys())
    with open(os.path.join(HERE, "dimension_granularity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols)
        w.writeheader()
        w.writerows(gran)

    # pretty print
    print("\nSTEP 3: per-parent granularity (dimension_granularity.csv)\n")
    hdr = ["parent", "n_sub", "seed/emrg", "mean", "sd", "min", "max", "present", "norm", "scoreable", "mean|scor"]
    print("{:<24}{:>6}{:>11}{:>7}{:>6}{:>5}{:>5}{:>9}{:>7}{:>11}{:>11}".format(*hdr))
    for g in gran:
        print("{:<24}{:>6}{:>11}{:>7}{:>6}{:>5}{:>5}{:>9}{:>7}{:>11}{:>11}".format(
            g["parent_dimension"], g["n_subdims_in_parent"],
            f'{g["n_seeded"]}/{g["n_emergent"]}', g["mean_count"], g["sd_count"], g["min"],
            g["max"], g["frac_items_present"], g["norm_mean"], g["frac_items_scoreable"],
            g["mean_count_where_scoreable"]))

    readout(gran)


def readout(gran):
    by = {g["parent_dimension"]: g for g in gran}
    # 3-way split: bimodal/coverage-limited first (low present-rate), then near-binary (avg ~1),
    # then graded (present on most items with real depth + spread).
    bimodal = [g["parent_dimension"] for g in gran if g["frac_items_present"] < 0.7]
    near_binary = [g["parent_dimension"] for g in gran
                   if g["parent_dimension"] not in bimodal and g["mean_count"] <= 1.3]
    graded = [g["parent_dimension"] for g in gran
              if g["parent_dimension"] not in bimodal and g["parent_dimension"] not in near_binary]
    print("\n" + "=" * 80 + "\nPLAIN-LANGUAGE READOUT\n" + "=" * 80)
    print(
        f"GRADED signal worth formal 0-5 scoring later: {', '.join(graded)}. Each fires several "
        f"member sub-dims on nearly every item with real spread (e.g. localization_grounding "
        f"{by['localization_grounding']['mean_count']}±{by['localization_grounding']['sd_count']}, "
        f"legibility_render {by['legibility_render']['mean_count']}±{by['legibility_render']['sd_count']}), "
        f"so a graded rubric would add resolution rather than just re-detecting presence."
    )
    print(
        f"NEAR-BINARY / low resolution as-is: {', '.join(near_binary) or 'none'} "
        f"(encoding_literacy averages ~1 fired sub-dim — mostly a present/absent flag, little graded depth)."
    )
    print(
        f"RARELY present / bimodal AND coverage-limited: {', '.join(bimodal)}. referent_mismatch fires "
        f"on {by['referent_mismatch']['frac_items_present']*100:.0f}% and scope_overreach on "
        f"{by['scope_overreach']['frac_items_present']*100:.0f}% of items, but both are 100% emergent "
        f"so were only scoreable on ~{by['referent_mismatch']['frac_items_scoreable']*100:.0f}% of items "
        f"— their true presence is higher than these lower bounds, and when present they fire in clusters "
        f"(referent_mismatch max {by['referent_mismatch']['max']})."
    )
    print(
        "CRITICAL CAVEAT: this measures 0-vs-1+ PRESENCE, not difficulty. The discovery log scored "
        "emergent sub-dims only from their first-sighting batch onward (never retroactively), so the "
        "60-90 earliest items carry only the 23 seeded dims; parents that are mostly emergent "
        "(referent_mismatch, scope_overreach, gestalt_shape) are undercounted — compare frac_items_present "
        "against frac_items_scoreable / mean_count_where_scoreable. All counts are LOWER BOUNDS; formal "
        "0-5 scoring is the later step that adds difficulty resolution to whichever parents survive."
    )


if __name__ == "__main__":
    main()
