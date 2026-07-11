# sciver_eval — Haiku on the SciVer test task, via the Claude Code harness

Runs the **SciVer multimodal scientific claim-verification** test task with Claude
**through the Claude Code subagent harness** (credit-free), instead of the paid API.
Built as a verdict-aware sibling of `rubric_scoring`: same 817 chart-item universe,
identical `item_id` keys, so predictions join 1:1 to the DeLeAn demand scores.

## What this session built

- **Cloned the official harness** (`github.com/QDRhhhh/SciVer` → `../SciVer-repo`) for
  the paper's exact CoT prompts + `acc_evaluation` scorer; confirmed the local SciVer
  data (`../SciVer`, val+test).
- **Item set decision:** the rubric pipeline's 817 = exactly the **chart-type
  direct/analytical** items (val 279 + test 538). We run Haiku on that same set so it
  joins to the rubric demand scores (table-type items have no rubric scores).
- **`predictions.db`** — mirrors the rubric DB:
  - `item` (verbatim from `rubric_scoring`, same keys) + `run` (model-keyed) +
    `prediction` (`answer`/`predicted`/`correct` + per-call token provenance).
  - `UNIQUE(run, item_id, model, trial)` → supports **repeated stochastic trials**.
- **Pipeline** (`prepare → dispatch → collect`), resume-aware, mirroring the rubric
  machinery; verdicts + exact tokens recovered from subagent transcripts by `TASK_TAG`.
- **`wf_dispatch.js`** — fans a shard of `(item, trial)` cells out to Haiku subagents
  across background **Workflows** (4 in parallel).

### Fixes / gotchas found this session
- Tag collision: original tag put `item_id` last, so `…_val_1` ⊂ `…_val_10` (782 would-be
  collisions). Restructured to `se_r{run}__{item_id}__t{trial}` (0 collisions). The
  **rubric** scheme `rs_p{pass}__{item_id}__{dim}` was verified **unaffected**.
- Workflow `args` arrives as a JSON **string** → parse-guard in `wf_dispatch.js`.
- `collect` is `O(tags × transcripts)` — slow at scale; runs in the background.

## Results — run 1 = `claude-haiku-4-5`, trials 1–5 (4085 cells, parse_ok 1.0)

- **Accuracy 0.673 pooled**, stable per trial (0.656–0.683). By type: analytical 0.706,
  direct 0.642. Recall: refuted **0.89**, entailed **0.46** (strong skeptic bias).
- **Per-item `p(correct)` over 5 trials:** 50.7% always-right, 15.5% always-wrong,
  **33.8% flip at least once** (the multi-trial difficulty signal).
- Of the **414 always-right** items, **314 (76%) are refuted** — Haiku's reliable wins
  are overwhelmingly "no" answers.
- **Demand correlation** `r(demand, p_correct)` (negative = harder): **AS −0.131**
  (strongest), VO −0.076, MCr −0.066, AT −0.009, QLq +0.024, QLl +0.077.

## Run it

```bash
source .venv/bin/activate
python -m sciver_eval.build_db                     # one-time: 817 items + run 1
python -m sciver_eval.prepare --run 1 --trials 5   # resume-aware; writes prompts + manifest
# dispatch each (item,trial) cell to a Haiku subagent (Agent tool or wf_dispatch.js workflows)
python -m sciver_eval.collect_cli --run 1 --manifest sciver_eval/prompts/run_01/run_01_manifest.json
```

Predictions join to demand scores on `item_id` across `predictions.db` ⋈
`rubric_scoring/annotations_prod.db`.

## Next

More trials (tighter `p`), a **multi-model sweep** (Sonnet/Opus as new `run` rows →
item difficulty vs. model ability), or a formal IRT fit on the (item × model × trial)
correctness matrix.
