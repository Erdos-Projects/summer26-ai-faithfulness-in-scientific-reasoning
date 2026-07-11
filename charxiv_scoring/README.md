# charxiv_scoring

DeLeAn demand-rubric annotation over the **1,000 CharXiv validation reasoning items**, plus a
drafted DB of CharXiv's **released per-model correctness scores** (so the correctness side needs
no eval run). Cross-dataset companion to `sciver_eval` / `sciclaimeval_scoring`.

## Status (what is set up vs. not)

- ✅ **Data downloaded** to `../CharXiv` (sibling clone, like `../SciClaimEval`): `data/` QA JSONs,
  `images/` (2,323 jpgs; all 1,000 val charts covered), `existing_evaluations/` (78 val score files).
- ✅ **DB drafted** — `annotations_charxiv.db` has `item` (1,000) + `model_score` (195,000 rows:
  39 models × {reasoning 1/chart, descriptive 4/chart}). Released accuracies reproduce the paper
  (GPT-4o 0.471, Human 0.805, Claude-3.5-Sonnet 0.602).
- ✅ **Sonnet annotation infrastructure** ready (prepare → dispatch → collect), **NOT yet run** —
  the `annotation` table is empty.

## Item set

- **Source:** `data/reasoning_val.json` + `data/image_metadata_val.json` from the CharXiv release.
- **One item == one (chart, reasoning question)**, keyed by `figure_id`.
- **Item IDs:** prefixed `charxiv_val_` (e.g. `charxiv_val_0`).
- **Count:** 1,000 (verified by `items.load_items()`).
- **Task framing:** demand is scored for *answering the reasoning question using the chart* (vs the
  claim-verification framing in SciVer/SciClaimEval).

## Correctness side (no eval needed)

CharXiv releases pre-graded 0/1 scores for ~39 models on the val split. `build_db` parses them into
`model_score(model, task, item_id, sub_q, figure_id, extracted_answer, score)`:
- `task='reasoning'` → 1 row/chart (`sub_q=0`) — **the correctness label for the analysis**.
- `task='descriptive'` → 4 rows/chart (`sub_q=0..3`) — optional easy-vs-hard contrast.

```bash
python -m charxiv_scoring.build_db      # idempotent: load items + all released scores
```

## Demand pass — prepare → dispatch → collect (resume-aware)

Active dim set (same 12 as the SciVer cross-dataset analysis): `AS,QLq,QLl,MCr,AT,VO,VL,GS,MA,MCu,CL,KNf`
→ 12 dims × 1,000 items = **12,000 cells per pass**.

```bash
source .venv/bin/activate

# Step 1 — prepare (writes prompts for MISSING cells only, emits a manifest)
python -m charxiv_scoring.prepare --pass 1 --dims AS,QLq,QLl,MCr,AT,VO,VL,GS,MA,MCu,CL,KNf

# Step 2 — dispatch (one isolated Sonnet subagent per cell)
#   either manually via the Agent tool, or run the Workflow in wf_dispatch.js
#   (stage 1 self-refreshes the manifest, stage 2 fans out a batch).

# Step 3 — collect (scrape transcripts → scores + token provenance)
python -m charxiv_scoring.collect_cli --pass 1 --manifest charxiv_scoring/prompts/pass_01/pass_01_manifest.json
```

Repeat 1–3 until `prepare` prints `0 MISSING cells`. Smoke test: add `--limit 3`.

## Verdict-blind invariant

`build_prompt` interpolates ONLY the reasoning question and figure path — never the ground-truth
`answer` or `inst_category`. Those are stored in `item` for analysis but must not reach a prompt.
(Verified: the answer string appears in 0 generated prompts.)

## ID and tag schemes

| Namespace | Format | Example |
|-----------|--------|---------|
| Item ID | `charxiv_val_<figure_id>` | `charxiv_val_0` |
| Annotation tag | `cxr_p<N>__<item_id>__<dim_code>` | `cxr_p01__charxiv_val_0__AS` |

`cxr_p…` is disjoint from SciVer (`rs_p…`) and SciClaimEval (`rsc_p…`) tags, so all annotation DBs
can be scraped side by side without collision.

## Join demand ↔ correctness (once the pass has run)

```sql
-- annotations_charxiv.db (single DB holds both sides)
SELECT a.item_id, a.dim_code, a.score AS demand, s.score AS correct
FROM annotation a
JOIN model_score s
  ON s.item_id = a.item_id AND s.task='reasoning' AND s.sub_q=0
WHERE a.parse_ok=1 AND a.pass=1 AND s.model='GPT-4o';
```

## Module structure

| File | Role |
|------|------|
| `config.py` | Path resolution (CharXiv data dir, DB, rubrics); env-overridable |
| `items.py` | Load the 1,000 val reasoning items (figure + question + metadata) |
| `db.py` | SQLite schema (`item` / `pass` / `annotation` / `model_score`); upserts |
| `scores.py` | Parse released `existing_evaluations/scores-*_val.json` → `model_score` |
| `build_db.py` | Draft the DB: init + load items + load all released scores |
| `prompts.py` | `build_prompt` (verdict-blind), `tag`, `DIMS` registry |
| `extract_rubrics.py` | Dim registries (+ optional rubric regeneration from the supp. PDF) |
| `prepare.py` | Register pass, write prompt files for MISSING cells, emit manifest |
| `scrape.py` / `collect_cli.py` | Scrape agent transcripts → scores + token provenance → DB |
| `aggregate.py` / `qa.py` | Mode/mean across passes; per-pass quality checks |
| `wf_dispatch.js` | Self-fetching Workflow: refresh manifest + fan out a batch on Sonnet |
| `rubrics/` | 21 rubric `.txt` files (12 active + spares), shared library |
