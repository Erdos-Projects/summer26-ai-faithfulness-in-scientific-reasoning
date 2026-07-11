# sciclaimeval_eval

Haiku Supported/Refuted prediction over the 265 SciClaimEval dev figure items.

## Purpose

This module runs `claude-haiku-4-5` on each `(item × trial)` cell to predict
Supported or Refuted, producing a per-item `p(correct)` across N trials. The
final analysis joins these predictions to the demand scores in
`sciclaimeval_scoring/annotations_sciclaimeval.db` on `item_id` to compute
`r(demand, p_correct)`.

Deferred conditions (not implemented in the current build):
- `entailed` — claim is entailed by the figure without ambiguity
- `refuted` — claim is explicitly contradicted by the figure

These conditions are stored in the `operation` field of the `item` table
(verdict-blind at prompt time) and can be used as strata in post-hoc analysis.

## Item set

- **Source:** `dev_task1_release.json`, filtered to `evi_type == "figure"`.
- **Item IDs:** prefixed `scev_` (e.g. `scev_val_fig_0001`).
- **Count:** 265 items (verified by `items.load_items()` in `sciclaimeval_scoring`).

## Model default

`claude-haiku-4-5`

## ID and tag schemes

| Namespace | Format | Example |
|-----------|--------|---------|
| Item ID | `scev_<claim_id>` | `scev_val_fig_0001` |
| Prediction tag | `sce_r<N>_<item_id>_t<trial>` | `sce_r1_scev_val_fig_0001_t1` |

The `sce_` tag prefix is disjoint from the scoring tags (`rsc_p…`) and the
legacy SciVer tags (`rs_p…`), so all three annotation DBs can be referenced
side by side without collision.

## Build DB → Prepare → Dispatch → Collect loop

The run is split across three commands and is **resume-aware**: `prepare` only
emits cells that are not yet `parse_ok=1` in the DB, so re-running after a
partial session never re-dispatches finished work.

```bash
source .venv/bin/activate

# Step 1 — build the predictions DB (idempotent)
python -m sciclaimeval_eval.build_db

# Step 2 — prepare (resume-aware)
python -m sciclaimeval_eval.prepare --run 1 --trials 5

# Step 3 — dispatch (manual, via Claude Code Agent tool)
# For each MISSING cell in the manifest, launch one isolated Haiku subagent.

# Step 4 — collect (after each batch of agents finishes)
python -m sciclaimeval_eval.collect_cli --run 1 --manifest sciclaimeval_eval/prompts/run_01/run_01_manifest.json
```

Repeat steps 2–4 across sessions until `prepare` prints `0 MISSING cells`.

## Join to the scoring DB

Predictions join to demand scores in `sciclaimeval_scoring/annotations_sciclaimeval.db`
on `item_id` for `r(demand, p_correct)` analysis:

```sql
-- sciclaimeval_eval/predictions_sciclaimeval.db  ←→  sciclaimeval_scoring/annotations_sciclaimeval.db
SELECT p.item_id, a.dim_code, a.score, p.predicted, p.correct
FROM prediction p
JOIN annotation a USING(item_id)
WHERE p.parse_ok = 1 AND a.parse_ok = 1 AND p.run = 1 AND a.pass = 1;
```

`item_id` values are `scev_`-prefixed in both databases.

The two DB files are created at runtime and are not committed to the repo:
- `sciclaimeval_scoring/annotations_sciclaimeval.db`
- `sciclaimeval_eval/predictions_sciclaimeval.db`

## Module structure

| File | Role |
|------|------|
| `db.py` | SQLite schema, connect/init, upsert_item, upsert_prediction, completed_cells |
| `build_db.py` | One-time DB init (idempotent) |
| `prompts.py` | `build_prompt`, `tag`, `sce_` namespace |
| `prepare.py` | Register run, write prompt files for MISSING cells, emit manifest |
| `scrape.py` | Extract agent transcripts from `~/.claude/projects/` |
| `parse.py` | Parse Supported/Refuted answer from raw agent output |
| `collect_cli.py` | CLI: scrape + parse + upsert for a completed manifest |
