# Paired entailed-vs-refuted SciVer eval — design

**Date:** 2026-06-16
**Module:** `sciver_eval`
**Status:** approved (brainstorming → ready for implementation plan)

## Goal

Measure Haiku's claim-verification accuracy when **every** claim is entailed vs.
when **every** claim is refuted, over the *same* 817 SciVer chart items, to produce
**within-item paired accuracy data**. This isolates verdict behavior (the skeptic
bias seen in run-1: refuted recall 0.89 vs. entailed recall 0.46) from chart
difficulty, because the only thing that changes between the two conditions is the
truth value of the claim — same chart, caption, context, and prompt template.

## Key data fact

Every SciVer chart record carries **both** statements (verified: all 817 items,
zero empty, `claim` maps exactly to one of them by `label`):

- `origin_statement` — the true / **entailed** statement
- `perturbed_statement` — a targeted false / **refuted** perturbation of it
- `perturbed_explanation` — why the perturbation is wrong (NEVER shown to the model)

So both conditions are constructible over the full 817 items with no data loss.

## The three runs (no reuse, each independently complete)

| Run | Claim shown | Grade label | Verdicts | Status |
|-----|-------------|-------------|----------|--------|
| **1 — native** | official `claim` | official `label` | mixed | done (5 trials, 4085 cells) |
| **2 — entailed-all** | `origin_statement` (all 817) | `1` (yes) | all entailed | new |
| **3 — refuted-all** | `perturbed_statement` (all 817) | `0` (no) | all refuted | new |

Run-1 is the **native GitHub benchmark** baseline; runs 2 and 3 are the new
homogeneous conditions. There is no cell reuse across runs — three independent
sweeps. Any content coincidence (run-1's claim text equalling `origin_statement`
for entailed-labeled items) is irrelevant; we never copy predictions between runs.

The **paired analysis is run 2 ⋈ run 3** on `item_id`. Run-1 is compared against
both ("does forcing the truth value change accuracy vs. the natural mix?").

## Representation: approach A — run-as-condition

Chosen over (B) a `condition` column and (C) condition-suffixed `item_id`s.

- **Zero schema change.** `item_id`s stay byte-identical, so the rubric/DeLeAn
  demand join on `item_id` still works, and pairing is a self-join on `item_id`.
- Matches the existing README pattern ("new conditions = new run rows").
- Run→condition mapping is explicit in code (`CONDITIONS` dict) and recorded in
  `run.params` / `run.item_source`.

### Why it fits the existing machinery with near-zero churn

`scrape.collect` already grades `correct` against **`meta["label"]` from the
manifest** (`sciver_eval/scrape.py:88`), not from `item.label`. So the grading
label can be set per-condition without touching the schema or the shared `item`
table. Only two things change:

1. `prompts.build_prompt` takes which statement to show (instead of always
   `rec["claim"]`).
2. `prepare` writes the condition's statement + grade-label into the manifest.

`wf_dispatch.js` and `collect_cli` are unchanged — they read the manifest as-is.

## Design details

### 1. Conditions as a first-class concept

```python
CONDITIONS = {
    "entailed": ("origin_statement",    1),  # show true statement;  correct = yes
    "refuted":  ("perturbed_statement", 0),  # show false statement; correct = no
}
```

`prepare` gains `--condition {entailed,refuted}`; it selects `statement_field` and
`grade_label` from `CONDITIONS`, swaps `claim := rec[statement_field]`, and writes
`grade_label` into each manifest cell. The native run (run 1) keeps using the
official `claim`/`label` (a third implicit "native" condition; no code path needed
since run-1 is already complete, but the mapping is documented).

### 2. Verdict-clean preserved

Swapping *which statement is the claim* keeps the prompt free of `label` and
`perturbed_explanation` — the existing guarantee. Add an assertion/test that
neither `perturbed_explanation` nor the literal label leaks into a built prompt.

### 3. Run allocation

- run 2 = `claude-haiku-4-5`, condition `entailed`, full 817
- run 3 = `claude-haiku-4-5`, condition `refuted`, full 817

Leaves run 4/5 open for a later Sonnet/Opus pairing. `register_run` records the
condition string in `params` / `item_source`.

### 4. Manifest provenance

Each cell additionally stores `condition` and the **exact statement text shown**,
so a run is fully self-describing without re-deriving from source JSON.

### 5. Trial depth is a free parameter

The pipeline is already `--trials` + resume-aware. Workflow: pilot (small
`--limit` and/or `--trials 1`) → inspect → scale to `--trials 5`. Later trials
top up the same run; finished cells are never re-dispatched.

### 6. Dispatch / collect unchanged

`wf_dispatch.js` fans `(item, trial)` cells of a run's manifest to Haiku
subagents; `collect_cli` recovers verdicts + token provenance by `TASK_TAG`.
Both already read the manifest, so they work for the new runs unmodified.

## Deliverable: paired-analysis report

Join run 2 ⋈ run 3 on `item_id`:

- Per-item `p_correct` in each condition; the entailed-vs-refuted **accuracy gap**.
- Paired 2×2 + **McNemar** test (within-chart; controls for chart difficulty).
- Pooled **signal-detection** quantities: hit-rate = entailed accuracy,
  false-alarm = 1 − refuted accuracy → **d′** (discriminability) and
  **criterion c** (skeptic bias) — quantifies what run-1's recall split only hinted.
- Join to DeLeAn demand scores: does the entailed/refuted gap track AS / VO / etc.?
- Compare both conditions against the run-1 native baseline.

## Non-goals

- Reusing run-1 predictions for any new-run cell (run everything fresh).
- The prior-injection variant (telling the model "all claims here are entailed") —
  explicitly out of scope; this is the *homogeneous-subset* design, blind prompt.
- A `condition` schema column or suffixed `item_id`s (approach B/C, rejected).
- Multi-model sweep (Sonnet/Opus) — deferred; run-ids reserved.

## Testing

- Unit: `build_prompt` for a known item in each condition shows the right statement
  and never contains `perturbed_explanation` or a label token.
- Unit: `prepare` writes manifest cells with the correct `label` per condition.
- Smoke: a `--limit 3 --trials 1` pilot per condition dispatches, collects, and
  grades end-to-end; eyeball that entailed cells expect `yes` and refuted `no`.
