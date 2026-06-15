---
name: rubric-scoring
description: Use when running a DeLeAn demand-rubric annotation pass over SciVer chart items — scores each (item x dimension) with an isolated Sonnet subagent, credit-free, resumable, with per-call token provenance in SQLite.
---

# Rubric Scoring (DeLeAn demand annotation)

Score SciVer chart claim-verification items on DeLeAn demand rubrics. One isolated Sonnet
subagent per (item x dimension); verdict-blind; resumable across sessions; per-call tokens
captured from transcripts. Design: `docs/RUBRIC_SCORING_SKILL_DESIGN.md`. Run all commands
from the repo root with the venv active (`source .venv/bin/activate`).

## Parameters (confirm with the user before starting)
- **pass number** (integer; a new pass = a fresh full sweep for denoising).
- **active dims** (default `AS,QLq,QLl,MCr,AT,VO`; any subset of the 18).
- **limit** (optional; for a smoke test, e.g. 3 items).

## Procedure (repeat per session until the pass is complete)

1. **Prepare (resume-aware).** Run:
   `python -m rubric_scoring.prepare --pass <N> --dims <CSV> [--limit <k>]`
   It registers the pass, upserts items, and writes prompt files **only for cells not yet
   `parse_ok=1`**, plus `pass_<N>_manifest.json` (path is printed). Note the count of MISSING
   cells. If it prints `0 MISSING`, the pass is complete — skip to "When the pass is complete".

2. **Dispatch.** For each missing cell in the manifest, launch an isolated subagent with the
   **Agent tool** (`subagent_type: general-purpose`, `model: sonnet`). Send this prompt
   (substitute the cell's `prompt_file` and `dim_name`):
   > Read `<prompt_file>` and follow it EXACTLY. (1) Read that file. (2) Use the Read tool to
   > view the figure image it references. (3) Reason briefly. (4) Conclude with the EXACT line
   > "Thus, the level of *<dim_name>* demanded by the given TASK INSTANCE is: N" (N = 0-5).
   > (5) STOP. Use ONLY the Read tool (twice). Do NOT verify the claim.
   Dispatch in concurrent batches (the harness caps concurrency). Expect ~47-50k tokens and
   ~20s per agent. Stop when you hit rate limits; the run resumes next session.

3. **Collect.** After a batch/session, run:
   `python -m rubric_scoring.collect_cli --pass <N> --manifest <path to pass_<N>_manifest.json>`
   It scrapes THIS session's subagent transcripts, parses scores, captures exact tokens, and
   upserts to the DB. Re-running is safe (upsert). It prints how many cells the pass now has.

4. **Repeat** from step 1 across sessions. `prepare` always re-emits only what's still missing,
   so finished cells are never re-dispatched (each agent is expensive).

## When the pass is complete

QA report:
```
python -c "from rubric_scoring import db, config, qa; import json; print(json.dumps(qa.run_qa(db.connect(config.db_path()), N), indent=2))"
```
Confirm ~100% `parse_rate`, `thinking_clean: true`, and a spread of scores per dim (watch for
`flat: true` dims, which mirror the paper's 'lack of sensitivity' concern).

Aggregate across passes (per-cell mode/mean -> CSV):
```
python -c "from rubric_scoring import db, config, aggregate; aggregate.export_csv(db.connect(config.db_path()), ['AS','QLq','QLl','MCr','AT','VO'], 'rubric_scoring/aggregate.csv')"
```

## Invariants (do not violate)
- **Verdict-blind:** never put `label`/`origin_statement`/`perturbed_statement`/
  `perturbed_explanation` into a prompt. `prepare` already guarantees this (build_prompt reads
  only the claim + figure path).
- **Isolation:** one rubric per agent, fresh context. Never have one agent score multiple dims.
- **Reasoning off / no extra tools:** agents use only Read; they do not verify the claim.
  (Reasoning effort is not controllable on this path; QA's `thinking_clean` confirms it stayed off.)
- **Don't re-score finished cells:** always go through `prepare` (resume), never re-dispatch all.
