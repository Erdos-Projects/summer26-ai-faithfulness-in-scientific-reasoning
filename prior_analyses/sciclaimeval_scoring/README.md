# sciclaimeval_scoring

DeLeAn demand-rubric annotation over the 265 SciClaimEval dev figure items.

## Purpose

This module scores each `(item × dimension)` cell with an isolated Sonnet subagent,
producing a per-cell demand score (0–5) for up to 18 DeLeAn dimensions. The canonical
active set is `AS,QLq,QLl,MCr,AT,VO` (6 dims × 265 items = 1,590 cells per pass).
Scores are stored in `sciclaimeval_scoring/annotations_sciclaimeval.db` (SQLite).

## Item set

- **Source:** `dev_task1_release.json` from the SciClaimEval shared-task data release,
  filtered to `evi_type == "figure"`.
- **Item IDs:** prefixed `scev_` (e.g. `scev_paper42_claim3`).
- **Count:** 265 items (verified by `items.load_items()`).

## Prepare → Dispatch → Collect loop

The run is split across three commands and is **resume-aware**: `prepare` only emits
cells that are not yet `parse_ok=1` in the DB, so re-running after a partial session
never re-dispatches finished work.

```bash
source .venv/bin/activate

# Step 1 — prepare (resume-aware)
python -m sciclaimeval_scoring.prepare --pass 1 --dims AS,QLq,QLl,MCr,AT,VO

# Step 2 — dispatch (manual, via Claude Code Agent tool)
# For each MISSING cell in the manifest, launch one isolated subagent:
#   subagent_type: general-purpose, model: sonnet
# See .claude/skills/sciclaimeval-rubric-scoring/SKILL.md for the exact prompt template.

# Step 3 — collect (after each batch of agents finishes)
python -m sciclaimeval_scoring.collect_cli --pass 1 --manifest sciclaimeval_scoring/prompts/pass_01/pass_01_manifest.json
```

Repeat steps 1–3 across sessions until `prepare` prints `0 MISSING cells`.

### Optional: smoke test (limit)

```bash
python -m sciclaimeval_scoring.prepare --pass 1 --dims AS,QLq,QLl,MCr,AT,VO --limit 3
```

## Verdict-blind invariant and operation-leak note

**Verdict-blind fields** — never sent to any annotator prompt:
- `label` (ground-truth Supported/Refuted boolean)
- `operation` — **leaks the verdict**: `Supported_claim_only` implies Supported.
  This field is stored in the DB for analysis but must not appear in prompts.
- `caption`
- `context`
- `claim_id_pair`

`prepare` enforces this: `build_prompt` reads only the claim text and the figure image path.

## ID and tag schemes

| Namespace | Format | Example |
|-----------|--------|---------|
| Item ID | `scev_<claim_id>` | `scev_paper42_claim3` |
| Annotation tag | `rsc_p<N>_<item_id>_<dim_code>` | `rsc_p1_scev_paper42_claim3_AS` |

The `rsc_p…` tag prefix is disjoint from the SciVer rubric-scoring tags (`rs_p…`),
so the two annotation DBs can be referenced side by side without collision.

## Join to the eval DB

Demand scores join to the Haiku Supported/Refuted predictions on `item_id`:

```sql
-- annotations_sciclaimeval.db  ←→  sciclaimeval_eval/predictions_sciclaimeval.db
SELECT a.item_id, a.dim_code, a.score, p.prediction, p.label
FROM annotation a
JOIN predictions p USING(item_id)
WHERE a.parse_ok = 1 AND a.pass = 1;
```

`item_id` values are `scev_`-prefixed in both databases.

## QA and aggregation

```bash
# QA report for pass N
python -c "
from sciclaimeval_scoring import db, config, qa
import json
N = 1
print(json.dumps(qa.run_qa(db.connect(config.db_path()), N), indent=2))
"

# Aggregate across passes (per-cell mode/mean → CSV)
python -c "
from sciclaimeval_scoring import db, config, aggregate
aggregate.export_csv(
    db.connect(config.db_path()),
    ['AS','QLq','QLl','MCr','AT','VO'],
    'sciclaimeval_scoring/aggregate.csv'
)
"
```

## Module structure

| File | Role |
|------|------|
| `config.py` | Path resolution (DB, data dir, rubrics dir); env-overridable |
| `items.py` | Load and validate the 265 figure items from the dataset |
| `db.py` | SQLite schema, connect/init, upsert_item, upsert_annotation, completed_cells |
| `prompts.py` | `build_prompt`, `tag`, `DIMS` registry |
| `prepare.py` | Register pass, write prompt files for MISSING cells, emit manifest |
| `scrape.py` | Extract agent transcripts from `~/.claude/projects/` |
| `parse.py` | Parse score from raw agent output |
| `collect_cli.py` | CLI: scrape + parse + upsert for a completed manifest |
| `aggregate.py` | Mode/mean across passes; CSV export |
| `qa.py` | Per-pass quality checks (parse rate, thinking, score spread) |
