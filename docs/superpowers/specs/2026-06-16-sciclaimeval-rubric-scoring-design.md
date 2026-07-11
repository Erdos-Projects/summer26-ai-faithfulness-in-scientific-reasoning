# SciClaimEval Rubric-Scoring + Prediction — Design Spec

- **Date:** 2026-06-16
- **Status:** Draft for review
- **Author:** awndre (with Claude Code)
- **Reference:** Ports the SciVer pipeline (`rubric_scoring/` + `sciver_eval/`) to the
  SciClaimEval figure-evidence items. Method: Zhou et al. 2026, *General scales unlock AI
  evaluation with explanatory and predictive power*, Nature 652:58–67 (DeLeAn/ADeLe demand
  annotation). SciVer design: `docs/RUBRIC_SCORING_SKILL_DESIGN.md`.

---

## 1. Purpose

Reproduce, on the SciClaimEval dataset, the two coupled SciVer pipelines:

1. **Demand rubric scoring** — DeLeAn demand annotation: each figure item scored on a fixed
   set of demand dimensions, one scale per isolated subagent, verdict-blind, on the credit-free
   Claude Code subagent path, with per-call token provenance.
2. **Model prediction** — a model answers the actual Supported/Refuted verification task over
   the same items across repeated trials, yielding per-item `p(correct)`.

The payoff is the join on `item_id`: demand scores ⋈ predictions → the demand-vs-difficulty
correlation `r(demand, p_correct)` that is the point of the DeLeAn method.

## 2. Compartmentalization (hard requirement)

SciClaimEval work is fully isolated from the SciVer modules:

- **Full standalone copies.** No imports from `rubric_scoring` or `sciver_eval`. The proven
  code is physically duplicated and the dataset-specific parts rewritten. (Within the
  SciClaimEval pair, `sciclaimeval_eval` may import `sciclaimeval_scoring` for the shared item
  loader / path config — mirroring how `sciver_eval` imports `rubric_scoring`. Isolation is
  from the *SciVer* modules, not within the SciClaimEval pair.)
- **Own DBs.** `annotations_sciclaimeval.db` (demand scores), `predictions_sciclaimeval.db`
  (predictions). Never touches the SciVer DBs.
- **Own folders, own prompt namespaces, own skill.**

Accepted cost: ~380 lines of dataset-agnostic logic are duplicated; a future fix to shared
logic must be applied in both copies. This is the deliberate trade for isolation.

## 3. Layout

Two sibling modules in the `summer26-ai-science-reasoning` repo, mirroring SciVer's
`rubric_scoring` / `sciver_eval` split, both prefixed `sciclaimeval_`:

```
summer26-ai-science-reasoning/
  sciclaimeval_scoring/              # clone of rubric_scoring — DeLeAn demand scores
    __init__.py config.py items.py prompts.py db.py parse.py scrape.py
    collect_cli.py prepare.py aggregate.py qa.py
    rubrics/                         # copy of the 21 rubric .txt files
    prompts/                         # own prompt-file output namespace
    annotations_sciclaimeval.db      # own DB (created on first run)
    tests/                           # copied, repointed
    README.md
  sciclaimeval_eval/                 # clone of sciver_eval — model predictions
    __init__.py prompts.py db.py parse.py scrape.py
    prepare.py collect_cli.py build_db.py
    predictions_sciclaimeval.db      # own DB
    README.md
    # imports config + items from sciclaimeval_scoring (no own copy), mirroring
    # how sciver_eval imports rubric_scoring
  .claude/skills/
    sciclaimeval-rubric-scoring/SKILL.md   # new skill (clone of rubric-scoring)
```

`sciclaimeval_eval` imports `sciclaimeval_scoring.{config, items}` (mirroring
`sciver_eval`→`rubric_scoring`), so both pipelines key off the identical 265-item universe
without a third copy of the loader.

## 4. Target data

- **Source of truth:** `SciClaimEval/sciclaimeval-shared-task/data/dev_task1_release.json`,
  filtered to `evi_type == "figure"`. **Verified count: 265 items.** (NOT `feature_discovery`
  — the loader derives the set directly from the released dataset to stay self-contained.)
- **Labels (verified):** Supported 149 / Refuted 116.
- **Domains (verified):** ml 86, nlp 82, peerj 97.
- **Operations (verified):** Legend Swap 126, Category Swap 46, Graph Flip 40,
  Supported_claim_only 33, Graph Swap 20.
- **Images:** `figures/dev/<claim_id>.png`, resolved against the data root. **All 265 verified
  present on disk.**
- **`item_id` scheme:** `scev_` + `claim_id` → e.g. `scev_val_fig_0001`. Deterministic; matches
  the ids already used in `feature_discovery/sciclaimeval_split.json` so a later split join is
  free, without importing that file.

### Verdict-blindness (invariant)

The annotator prompt receives **only** `claim` (text) + figure path. Withheld to the DB only:
`label`, **`operation`** (leaks the verdict: `Supported_claim_only` is always Supported),
`caption`, `context`, `claim_id_pair`. The prediction pipeline likewise receives only
claim + figure (it must decide the verdict from evidence, not metadata).

## 5. The one real rewrite: `items.py`

`load_items()` rebuilt for SciClaimEval; the `Item` dataclass extended with two stratifier
fields (free, since the DB is new):

```
Item(item_id, source="sciclaimeval", paperid, claim_type, vtype="figure",
     image_path, claim, label, domain, operation)
```

- `paperid` = `paper_id`; `claim_type` = `""` (no SciVer-style claim_type here);
  `domain` and `operation` = the verified fields above.
- `label` = `Supported → True`, `Refuted → False`.
- `image_path` = absolute `data_root / evi_path`.

`db.py`'s `item` table gains `domain TEXT, operation TEXT`; `upsert_item` writes them. These
columns are stored, **never** read into a prompt.

## 6. Mechanical edits vs. a pure copy (everything else)

- **`config.py`:** new `sciclaimeval_dir()` resolver (env `SCICLAIMEVAL_DIR` override;
  candidates include `repo_root.parent / "SciClaimEval" / "sciclaimeval-shared-task" / "data"`),
  validated by presence of `dev_task1_release.json`. `db_path()` → `annotations_sciclaimeval.db`
  (env `SCICLAIMEVAL_SCORING_DB` override). `rubrics_dir()` → the module's own copy.
  `claude_projects_dir()` unchanged.
- **`prompts.py`:** `tag()` prefix `rs_p…` → `rsc_p…` (zero collision with the SciVer DB).
  Template wording "chart image" → "figure" (items are figures/plots, not only charts).
  Same DeLeAn template, same 18+3 `DIMS`, verdict-blind interpolation (claim + figure only).
- **`prepare.py`:** `item_source = "SciClaimEval dev figures (265)"`; same resume-aware,
  skip-completed, manifest-emitting logic.
- **Copied unchanged:** `parse.py`, `scrape.py`, `aggregate.py`, `qa.py`, `collect_cli.py`,
  and `db.py` (apart from the 2 new columns). `rubrics/` copied verbatim (21 files).
- **Defaults:** same 6 first-pass dims `AS,QLq,QLl,MCr,AT,VO`; full 18-dim library available
  by changing the active-dim list.

## 7. Prediction pipeline (`sciclaimeval_eval/`)

Clone of `sciver_eval`, swapped to the 265 figures, own `predictions_sciclaimeval.db`,
tag prefix `sce_…`.

- **Model:** `claude-haiku-4-5` (mirrors SciVer run 1 for a parallel comparison).
- **Task prompt:** single balanced CoT template (SciClaimEval has no `claim_type`, so the two
  SciVer templates collapse to one), built from `claim` + the inline `caption` + `context`
  fields (no separate paper-JSON parse needed — SciClaimEval carries them inline) + figure.
  Final line `Answer: yes` (Supported) / `Answer: no` (Refuted). Withholds `label`/`operation`.
- **Trials:** repeated stochastic trials → per-item `p(correct)`; schema keeps
  `UNIQUE(run, item_id, model, trial)` as in `sciver_eval`.
- **Native run only (first build):** the entailed/refuted paired-condition swapping
  (`sciver_eval/conditions.py`) is deferred — SciClaimEval items already ship as separate
  Supported/Refuted rows, so the native run covers both. See §12.
- **Join:** `predictions_sciclaimeval.db` ⋈ `annotations_sciclaimeval.db` on `item_id`.

## 8. Execution & scale

- **Path:** credit-free Claude Code subagents (Sonnet for rubric scoring, Haiku for
  prediction), one subagent per cell, resumable across sessions, skip-completed via `prepare`.
- **Scale (sourced):** demand pass = 265 × 6 = **1,590 agents/pass** (~3× smaller than SciVer's
  4,902). Per-agent cost/time is SciVer-measured (~47–50k tokens, ~20 s, ~$0.055–0.06
  API-equivalent, $0 actual on Max); **re-measured on the first SciClaimEval batch rather than
  assumed** (figure images may differ in token weight).
- **Deviations:** identical to the SciVer protocol — temperature ≠ 0 on the subagent path,
  output cap instructed not enforced, reasoning off, captions omitted, DeLeAn v1.0 textual-only
  anchors applied to figure items — logged in the `pass` row.

## 9. Skill packaging

New `.claude/skills/sciclaimeval-rubric-scoring/SKILL.md`, a clone of `rubric-scoring` with:
item count 265, module name `sciclaimeval_scoring`, tag prefix `rsc_`, item_source string,
and the verdict-blind invariant restated to include `operation`. Same prepare → dispatch →
collect procedure, same QA report. (The prediction pipeline ships as CLI + workflow like
`sciver_eval`, without its own skill, mirroring SciVer.)

## 10. Testing

Copy `rubric_scoring/tests/`, repoint to `sciclaimeval_scoring`:

1. `load_items()` returns exactly **265** items; all `image_path`s resolve on disk.
2. Label mapping: Supported→1, Refuted→0; verified counts 149/116.
3. **Verdict-blind audit:** no `label` / `operation` / `caption` / `context` string appears in
   any built prompt.
4. **Tag-collision check:** `rsc_p…` tags are disjoint from the SciVer `rs_p…` namespace.
5. DB round-trip: upsert item (incl. `domain`/`operation`), upsert annotation, resume skips
   completed cells.
6. Parse-rate smoke on 3 items × 6 dims.

## 11. Success criteria

- `sciclaimeval_scoring` runs a demand pass end-to-end: ~100% parse rate, verdict-blindness
  verified, per-cell scores + exact per-call tokens stored under `(pass, item_id, dim_code)`,
  resumable, deviations logged, cross-pass mode/mean exportable.
- `sciclaimeval_eval` produces per-item `p(correct)` over trials, joinable to demand scores on
  `item_id`.
- Zero coupling to the SciVer modules/DBs (no shared imports, no shared DB files).

## 12. Open / deferred

- **Gated test set:** only the 265 dev figures are available; the gated SciClaimEval test split
  is out of scope until released.
- **Table items (482):** out of scope (figures only, matching SciVer's charts-only decision).
- **Passes count (k):** chosen at run time (denoising vs campaign length).
- **Paired entailed/refuted conditions:** `sciver_eval/conditions.py` analog deferred; the
  native run already covers both members of each Supported/Refuted pair.
