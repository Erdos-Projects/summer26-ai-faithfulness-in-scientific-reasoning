# Reproducing the Haiku (and Sonnet) SciVer trials

End-to-end recipe for the verification eval that produced `../predictions.db` and
everything in this `trial_analysis/` folder. The whole eval runs **through the
Claude Code subagent harness** (credit-free) rather than the paid API: each
`(item, trial)` cell is answered by a subagent, and verdicts + exact token counts
are recovered from the subagents' own transcripts.

If you only want to **regenerate the figures/tables** from the existing DB, skip to
[§7 Analysis layer](#7-analysis-layer) — no model calls needed. Everything before
that re-runs the model and is stochastic.

---

## 1. What was produced

`predictions.db` holds **5 runs** over the same 817 chart items (val 279 + test 538):

| run | model | condition | claim shown | truth | trials |
|----:|-------|-----------|-------------|:-----:|:------:|
| 1 | `claude-haiku-4-5`  | native    | official `claim` (mixed) | mixed | 5 |
| 2 | `claude-haiku-4-5`  | entailed  | `origin_statement` (true)  | yes | 5 |
| 3 | `claude-haiku-4-5`  | refuted   | `perturbed_statement` (false) | no  | 5 |
| 4 | `claude-sonnet-4-6` | entailed  | `origin_statement` (true)  | yes | 1 |
| 5 | `claude-sonnet-4-6` | refuted   | `perturbed_statement` (false) | no  | 1 |

The **paired entailed/refuted design** (runs 2 vs 3, and 4 vs 5) is what lets us
separate *can the model read the chart* from *does it just default to "no"*: an
item shown as a true statement and as a false statement, scored independently.

All runs share `item_id` keys with `rubric_scoring/annotations_prod.db`, so
predictions join 1:1 to the DeLeAn demand scores.

---

## 2. Prerequisites

- **Python env:** `source /home/awndre/projects/Erdos/.venv/bin/activate`
  (repo installed editable; `matplotlib`, `matplotlib-venn` for the figures).
- **SciVer dataset** present at `../../SciVer` (val+test JSON, papers, figures).
  Path is resolved by `rubric_scoring.config.sciver_dir()`.
- **Shared item universe:** items come from `rubric_scoring.items.load_items()`,
  so `item_id`s are identical to the rubric DB. This is the join contract — do
  not regenerate item ids independently.
- **A Claude Code session** for the dispatch step (§5). The model is invoked via
  the harness `Workflow` / `agent()` path; there is no API key or temperature
  knob (see [§8 Caveats](#8-caveats--gotchas)).

All commands below are run from the repo root with the venv active and use the
`-m sciver_eval.<module>` form.

---

## 3. Data model & the TASK_TAG contract

`db.py` defines three tables (`item`, `run`, `prediction`). The keystone is:

```
UNIQUE(run, item_id, model, trial)   -- one prediction cell; enables resume + repeated trials
```

Every prompt embeds a **TASK_TAG** so the collector can match a subagent
transcript back to its cell (`prompts.tag`):

```
se_r{run:02d}__{item_id}__t{trial:02d}      e.g. se_r02__sciver_test_137__t04
```

The `item_id` is wrapped by `__t..` on the right so a short id (`sciver_val_1`) is
never a substring of a longer one (`sciver_val_10`) during transcript matching —
this collision was a real bug in the first tag scheme (782 would-be collisions).

The label the collector scores `correct` against comes from the **manifest**, never
from the prompt text the subagent saw (prompts are verdict-clean — they never
interpolate the label or the perturbation explanation).

---

## 4. Pipeline overview

```
build_db ──► prepare ──► shard ──► wf_dispatch.js (Workflow) ──► collect_cli ──► predictions.db
  (once)     (per run)  (per run)   (per run, in a CC session)   (per run)         │
                                                                                   ▼
                                                            viz_* / analyze_paired (§7)
```

`prepare`, `collect_cli` are **resume-aware**: re-running only emits / collects the
cells still missing (`parse_ok=1` marks a cell done). A crashed dispatch is
recovered by simply re-running `prepare` → dispatch → `collect_cli`.

---

## 5. Re-running a run from scratch

### 5.1 One-time: create the DB + item universe

```bash
python -m sciver_eval.build_db          # creates predictions.db, 817 items, registers run 1
```

### 5.2 `prepare` — register the run, write prompts, emit a manifest

`prepare` upserts items, registers the run row (model + provenance), then writes a
CoT prompt `.txt` for every **missing** `(item, trial)` cell and a manifest mapping
`tag -> {item_id, trial, prompt_file, images, label, condition, claim}`.

```bash
# run 1 — native claim (mixed truth), Haiku, 5 trials
python -m sciver_eval.prepare --run 1 --trials 5

# run 2 — entailed condition (shows origin_statement, truth=yes), Haiku, 5 trials
python -m sciver_eval.prepare --run 2 --trials 5 --condition entailed

# run 3 — refuted condition (shows perturbed_statement, truth=no), Haiku, 5 trials
python -m sciver_eval.prepare --run 3 --trials 5 --condition refuted

# run 4 / 5 — Sonnet, single trial. --model only sets the model id RECORDED on a NEW run row;
# the actual dispatch model is chosen separately in the Workflow args (§5.4).
python -m sciver_eval.prepare --run 4 --trials 1 --condition entailed --model claude-sonnet-4-6
python -m sciver_eval.prepare --run 5 --trials 1 --condition refuted  --model claude-sonnet-4-6
```

Prompts land in `sciver_eval/prompts/run_NN/`; the manifest is
`run_NN_manifest.json`. The CoT templates are ported verbatim from the SciVer
paper (`prompts.py`: `direct` → skeptical template, `analytical` → balanced
template — the only two claim types in the chart set) and include
claim + section context + caption, faithful to the benchmark task.

### 5.3 `shard` — split the manifest into cell-shards + workflow groups

```bash
python -m sciver_eval.shard --manifest sciver_eval/prompts/run_NN/run_NN_manifest.json \
    --shards 16 --groups 4
```

Writes `cells_shardKK.json` (`{"cells":[{id,trial}]}`) and `wf_groups.json` (16
shards round-robined into 4 parallel-workflow groups). Empty shards are dropped.
Sharding is deterministic (sorted by `(id, trial)`), so the layout is reproducible.

> **Historical note on the actual runs 2/3 layout.** Trials 2–5 of runs 2 and 3
> were dispatched from per-trial shard files named `cells_r{run}_t{trial}_s{NN}.json`
> (4 shards × 817 cells each), captured in `../trial_dispatch_specs.json`, rather
> than the `cells_shardKK.json` scheme above. Both are equivalent inputs to the
> dispatcher — a shard file just needs `{"cells":[{id,trial}, …]}`. Use the clean
> `shard.py` layout for fresh reproductions; `trial_dispatch_specs.json` documents
> exactly what was dispatched the first time. (`cells_canary.json` is a tiny smoke-
> test shard — dispatch it first to confirm the harness path before the full fan-out.)

### 5.4 Dispatch — `wf_dispatch.js` via the Workflow tool

This step runs **inside a Claude Code session**, not from the shell. The workflow
`wf_dispatch.js` reads each shard, then spawns **one verifier subagent per cell**
that (1) reads the figure image(s) with the Read tool, (2) reasons, (3) ends with
exactly `Answer: yes` / `Answer: no`. Loaders always use `haiku` (cheap JSON
echo); the **verifier** model comes from `args.model`.

Invoke the workflow once per group (the historical layout: once per dispatch
spec), passing the shard files, run number, prompt dir, and verifier model:

```jsonc
// Workflow args (parsed by wf_dispatch.js; arrives as a JSON string → it self-parses)
{
  "run": 2,
  "dir": "/abs/.../sciver_eval/prompts/run_02",
  "shardFiles": ["/abs/.../cells_r02_t02_s00.json", "…s01.json", "…s02.json", "…s03.json"],
  "model": "haiku"          // "haiku" for runs 1–3, "sonnet" for runs 4–5
}
```

`../trial_dispatch_specs.json` is the exact list of these payloads used for runs
2 & 3, trials 2–5 (trial 1 of each was the earlier `817×1` pilot dispatch). To
replay: iterate that array, calling `Workflow({scriptPath: ".../wf_dispatch.js",
args: spec})` for each entry. Concurrency is capped by the harness (~10–16
subagents live at once); 817-cell shards drain in waves.

The verifier model alias **must match** the model id recorded on the run row
(`haiku` ↔ `claude-haiku-4-5`, `sonnet` ↔ `claude-sonnet-4-6`).

### 5.5 `collect_cli` — scrape transcripts into `prediction`

After a dispatch finishes (or partly finishes), recover verdicts + exact tokens:

```bash
python -m sciver_eval.collect_cli --run NN \
    --manifest sciver_eval/prompts/run_NN/run_NN_manifest.json
```

`scrape.py` globs this session's `subagents/**/agent-*.jsonl`, matches each
manifest tag to the transcript(s) containing it, parses the **last** explicit
`Answer: yes|no` (`parse.py`), maps it to `predicted` (1=yes/entailed), and scores
`correct` against the manifest label. Token usage is summed from the transcript
`usage` blocks, so it is exact and credit-free. Re-running is safe (upsert).

> Collection is `O(tags × transcripts)` and slow at full scale — run it in the
> background. If a cell was retried, a transcript that actually parsed a verdict
> is preferred; otherwise the best available is stored with `parse_ok=0` for QA.

### 5.6 Verify completeness / resume

```bash
# how many cells landed, and parse health, per run
python -c "import sqlite3; from sciver_eval import db; c=sqlite3.connect(db.default_db_path()); \
print(c.execute('SELECT run,COUNT(DISTINCT trial),COUNT(DISTINCT item_id),AVG(parse_ok) \
FROM prediction GROUP BY run').fetchall())"
```

If a run is short of `817 × trials`, re-run `prepare` (emits only the missing
cells) → dispatch those shards → `collect_cli`. Target state: all 5 runs at
`parse_ok = 1.0`, runs 1–3 with 5 trials × 817, runs 4–5 with 1 × 817.

---

## 6. Per-run quick reference

| run | prepare | dispatch model | collect |
|----:|---------|:--------------:|---------|
| 1 | `--run 1 --trials 5` | haiku | `--run 1` |
| 2 | `--run 2 --trials 5 --condition entailed` | haiku | `--run 2` |
| 3 | `--run 3 --trials 5 --condition refuted`  | haiku | `--run 3` |
| 4 | `--run 4 --trials 1 --condition entailed --model claude-sonnet-4-6` | sonnet | `--run 4` |
| 5 | `--run 5 --trials 1 --condition refuted  --model claude-sonnet-4-6` | sonnet | `--run 5` |

(`shard` + dispatch between `prepare` and `collect` for every run.)

---

## 7. Analysis layer

All analysis scripts read `../predictions.db` and write into **this folder**
(`trial_analysis/`, resolved via `db.analysis_dir()`). These are pure post-hoc —
re-runnable any time, no model calls. Per-item correctness defaults to a
**majority vote across all trials** of a run; pass `--trial N` for a single trial.

```bash
# Paired entailed/refuted Venn (Haiku, maj.-vote of 5; default runs 2 vs 3)
python -m sciver_eval.viz_venn --by-claim-type
# …same, Sonnet single-trial (runs 4 vs 5)
python -m sciver_eval.viz_venn --entailed-run 4 --refuted-run 5 --by-claim-type \
    --out trial_analysis/venn_entailed_refuted_sonnet.png

# Per-item trial agreement: how many items were 5/5, 4/5, 3/5 unanimous (per run)
python -m sciver_eval.viz_consistency            # all runs present

# Entailed↔refuted agreement 2×2 + Cohen's κ
python -m sciver_eval.viz_agreement

# Run-1 per-item difficulty histogram p(correct) over 5 trials
python -m sciver_eval.viz_difficulty --run 1

# Paired stats: accuracy gap, McNemar, signal-detection d'/criterion
python -m sciver_eval.analyze_paired --ent 2 --ref 3                       # Haiku
python -m sciver_eval.analyze_paired --ent 4 --ref 5 --model claude-sonnet-4-6  # Sonnet
```

Outputs (PNG + CSV) are listed in `README.md` in this folder. Headline:
Haiku entailed 0.503 / refuted 0.882; Sonnet entailed 0.432 / refuted 0.956 —
same balanced accuracy (~0.69), Sonnet just a stronger "no"-skeptic.

---

## 8. Caveats & gotchas

- **Temperature is not controlled.** The harness-subagent `agent()` path exposes
  no temperature argument; the effective value is the Claude Code / Workflow
  default (assumed ~1.0, **not enforced**). Recorded as `NULL` with the caveat in
  `run.deviations` (`db.TEMPERATURE_UNCONTROLLED_NOTE`). Do not log a fake value.
- **Stochastic by design.** Re-running will not reproduce identical verdicts —
  that is the point of repeated trials. Reproduce *aggregate* numbers (accuracy,
  p(correct) buckets, Venn regions), not per-cell answers.
- **Trial-count asymmetry.** Haiku runs are 5 trials (denoised by majority vote);
  Sonnet runs are single-trial (noisier). For a strict Haiku-vs-Sonnet comparison,
  pass `--trial 1` to the Haiku analyses so both sides are single-shot.
- **Model alias must match the run row.** `wf_dispatch.js` `args.model`
  (`haiku`/`sonnet`) must correspond to the `model` recorded by `prepare`/
  `build_db`, or collection will attribute verdicts to the wrong model id.
- **Workflow `args` arrives as a JSON string** → `wf_dispatch.js` self-parses
  (`typeof args === 'string' ? JSON.parse(args) : args`). Keep that guard.
- **Tag substring safety** is load-bearing — keep the `__t{trial}` suffix on tags.
- **Session scope for collection:** `collect_cli` matches transcripts under the
  current session's `subagents/` tree (`CLAUDE_CODE_SESSION_ID`). Collect in the
  same session that dispatched, or point the scraper at the right projects dir.
```
