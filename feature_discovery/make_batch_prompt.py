"""Emit the discovery prompt for batch N (0-based). Usage:
   python3 feature_discovery/make_batch_prompt.py N > /tmp/batch_N.txt

Deliberately withholds labels, dataset operation types, and perturbation
explanations from the annotating agent so demand profiles stay verdict-blind.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ERDOS = os.path.dirname(ROOT)
N = int(sys.argv[1])

batches = json.load(open(os.path.join(ROOT, "batches.json")))["batches"]
batch = batches[N]
rows = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "manifest.csv")))}

codebook = open(os.path.join(ROOT, "codebook.md")).read()
# strip batch log / leakage sections; agent only needs dimension definitions
codebook = codebook.split("## Leakage-suspect flags")[0]

start_index = sum(len(b) for b in batches[:N]) + 1  # 1-based global index

items_txt = []
for k, iid in enumerate(batch):
    r = rows[iid]
    twin = f"\n  twin_group: {r['twin_id']} (this item is one of a pair sharing the same claim over different figure versions — score it independently on its own figure)" if r["twin_id"] else ""
    items_txt.append(
        f"ITEM {start_index + k} — {iid} (source: {r['source']})\n"
        f"  figure (view with Read): {os.path.join(ERDOS, r['figure_path'])}\n"
        f"  caption: {r['caption']}\n"
        f"  claim: {r['claim_text']}{twin}"
    )

out_path = os.path.join(ROOT, "batch_results", f"batch_{N:02d}.json")

print(f"""You are an annotator in an inductive rubric-discovery pass for figure-based
scientific claim verification (extending ADeLe's DeLeAn methodology, arXiv
2503.06378, to multimodal items). You will examine 10 claim+figure items and
(a) tally how they load on an existing codebook of demand dimensions,
(b) propose NEW candidate demand dimensions the codebook misses.

TWO GOVERNING PRINCIPLES:
1. DEMANDS, NOT VERDICTS. Every dimension describes what verifying the claim
   requires of a verifier (visual decoding, reasoning, knowledge), NEVER
   whether the claim is true. You are not told labels and must not try to
   infer or use them. If a candidate dimension would score differently
   depending on whether the claim is true, it leaks the verdict — rewrite it
   in terms of the demand (e.g. "near-tie discrimination potential", not
   "claim contradicts a close value").
2. OPERATIONALIZED ANCHORS. Anchors must cite observable criteria a human or
   weak model could apply ("must read exact values for 2+ points against
   axis ticks"), never sophistication judgments.

BACKGROUND for proposing dimensions: VLM failures on data visualizations
concentrate in the vision-language handoff (FUGU, arXiv 2510.21740):
exact-value readout, multi-point integration, and lookup-vs-gestalt fail
differently. Candidate dimensions tapping plausible VLM-specific failure
modes (small text, clipped marks, low-contrast overlays, rotated labels,
unusual aspect ratios, rasterization artifacts) are valuable even when
trivial for humans. Surface features like chart type carry little difficulty
signal (Verma & Fan, CogSci 2025) — favor dimensions about what the
verification ACT requires. New dimensions should fit the taxonomy: children
of visual processing (figure-intrinsic), claim-intrinsic, or relational
(claim-figure) demands.

CURRENT CODEBOOK (tally against ALL of these):
{codebook}

YOUR 10 ITEMS (view every figure with the Read tool before scoring):

{chr(10).join(items_txt)}

PROCEDURE per item: view the figure, read claim and caption, then for EVERY
codebook dimension assign a rough 0–5 loading (coverage tracking, not final
scores; for the two categorical dimensions record the category string(s)).
Also judge needs_context: could a verifier plausibly need surrounding paper
text (beyond figure+caption) to verify — "no" / "helpful" / "required", with
a 5–15 word reason.

Then ask explicitly: what does verifying these claims demand that NO existing
dimension captures? Propose new dimensions freely — over-generation is
correct; do NOT merge or prune. But do not re-propose an existing dimension
under a new name. For each: snake_case name, one-line human-applicable
definition, draft 0/2/4 anchors with observable criteria, exemplar item_ids
from this batch (high-loading and low-loading), and first_seen_item_id (the
item that prompted it).

If the batch contains a twin_group: after scoring both members independently,
compare their profiles and report any dimension you scored differently and
why (these become leakage-suspect flags).

OUTPUT: write EXACTLY one JSON file to {out_path}
with this schema (no other text output needed; your final message should be a
2–4 sentence summary):
{{
  "batch": {N},
  "items": [
    {{"item_id": "...",
      "needs_context": "no|helpful|required",
      "needs_context_reason": "...",
      "tallies": {{"<dimension_name>": <0-5 int, or string for categorical>}},
      "note": "optional, only if something unusual"}}
  ],
  "new_dimensions": [
    {{"name": "snake_case_name",
      "taxonomy": "figure-intrinsic|claim-intrinsic|relational",
      "definition": "...",
      "anchors": {{"0": "...", "2": "...", "4": "..."}},
      "exemplars_high": ["item_id"], "exemplars_low": ["item_id"],
      "first_seen_item_id": "...",
      "rationale": "1-2 sentences: what gap this fills"}}
  ],
  "twin_notes": [
    {{"twin_group": "...", "dimensions_scoring_differently": ["..."],
      "why": "..."}}
  ]
}}
The tallies object MUST contain every dimension name from the codebook,
using the exact section-header names (e.g. "chart_type",
"data_elements_to_integrate"). new_dimensions may be empty if nothing is
genuinely missing — do not invent filler.""")
