# CharXiv Demand-Scoring — Dev Log

- **Date:** 2026-06-18 (resume-to-completion session; package built earlier)
- **Branch:** `sciver-eval` (local commits only; nothing pushed)
- **Package:** `charxiv_scoring/` — see its `README.md` for module structure
- **DB:** `charxiv_scoring/annotations_charxiv.db` (committed; prompt `.txt` gitignored, manifests kept)
- **State note:** `memory/charxiv-scoring-pass-state.md`

## 1. Goal & final status

Run the DeLeAn demand-rubric annotation over the **1,000 CharXiv validation reasoning items × 12 dims = 12,000 cells**, one isolated Sonnet subagent per `(item × dimension)`, verdict-blind, resumable, with per-call token provenance. The correctness side needs no eval run — CharXiv ships pre-graded per-model 0/1 scores (`model_score`, 195k rows), so the demand pass is the only model work.

**Status: COMPLETE — 12,000 / 12,000 scored.** Closeout QA:
- parse rate **1.0** (every cell yielded a valid 0–5 score),
- **exactly 1,000 per dim** across AS, QLq, QLl, MCr, AT, VO, VL, GS, MA, MCu, CL, KNf (perfectly balanced),
- **thinking-clean 12,000/12,000** (`output_thinking_tokens=0` everywhere — extended reasoning stayed off, preserving the verdict-blind protocol),
- `prepare` confirms **0 MISSING**.

## 2. What this session did

Started at **6,432/12,000** (a prior session had stalled — see the old state note, blocked first on a server throttle, then on the Sonnet usage cap). This session drove it to **12,000 (+5,568)** and committed the result.

The work split into two phases: **fighting an active server-side throttle** (low yield, wasted tokens), then **a clean run** once it cleared. The operational lesson below is the durable takeaway.

## 3. The server-throttle saga (the key lesson)

**Symptom.** Parallel dispatch returned floods of `API Error: Server is temporarily limiting requests (not your usage limit) · Rate limited`. Measured yields under the throttle:
- **4×900 in parallel → ~22%** scored (+780 of 3,600 dispatched),
- **4×400 in parallel → ~29%** (+457 of 1,600).

**Two findings that shaped the strategy:**

1. **It is an Anthropic *server-side* throttle, independent of the usage-window reset.** When the user's usage limit reset, the throttle did **not** clear — the very next 4×400 wave was still ~29%. So waiting for a usage reset does nothing for it; you have to wait out the *server* limiter.

2. **Failed agents still cost tokens — a lot.** From the per-cell token provenance the collector records (even for `parse_ok=0` rows): the failed cells consumed **~2–3× the tokens of the successful ones** (failed: ~22M fresh input + 218M cache-write + 434M cache-read + 8.7M output; scored: ~0.05M + 80M + 261M + 2.4M). Why: each cell reads a **chart image** (token-heavy input), and the harness **retries** on a 429 — every retry re-bills that image as fresh/cache-write input. Only ~1,819 of the failed cells were "free" pure-rejects (429 at the gate); the rest burned tokens before dying. **Under an active throttle, parallel dispatch is doubly bad: low yield *and* you pay for the failures.**

**The protocol that worked (and is now saved):**
1. **Stop** hammering — more concurrency makes the throttle worse and wastes tokens.
2. **Wait ~15 min**, then fire **one 100-cell probe** (single workflow).
3. If the probe returns **zero rate-limit failures (~100% hit)** → throttle cleared, resume. If still ~20–30% → wait longer.
4. **Resume at low concurrency** — the user set a cap of **max 2 parallel workflows** to keep it manageable.

After one ~15-min wait, the 100-cell probe came back clean (first zero-failure run of the session). Every subsequent **2×400** wave then landed at **100% yield** (+800, +800, +800, +800), finishing with a single **2×512** covering the last 1,024.

## 4. Dispatcher change — `offset` + `prepare` flags

To run shards in parallel without duplicating work, `charxiv_scoring/wf_dispatch.js` was extended (commit `b6c4b6c`):
- **`offset`** — stage 1 returns the slice `[offset, offset+batch)` of the missing-cell manifest instead of always the first `batch`. Two workflows at offsets `0` and `400` therefore score **disjoint** cells.
- **`prepare:false`** — reuse an already-written manifest instead of re-running `prepare` inside the workflow. Required for parallel shards: if each workflow re-ran `prepare`, they would race on the manifest/prompt files. The controller runs `prepare` **once** before the wave, then fires the shards with `prepare:false`.

Without this, the original "first-N" dispatcher made parallel shards all grab the *same* first N cells.

## 5. Resume recipe (reusable)

```bash
# 1. refresh manifest to the current MISSING set
.venv/bin/python -m charxiv_scoring.prepare --pass 1 \
  --dims AS,QLq,QLl,MCr,AT,VO,VL,GS,MA,MCu,CL,KNf

# 2. fire <=2 parallel disjoint shards (Workflow tool, wf_dispatch.js)
#    {batch:400, offset:0,   prepare:false, model:'sonnet'}
#    {batch:400, offset:400, prepare:false, model:'sonnet'}

# 3. collect — --since 1800 skips the O(all-history) transcript scan (minutes -> seconds)
.venv/bin/python -m charxiv_scoring.collect_cli --pass 1 \
  --manifest charxiv_scoring/prompts/pass_01/pass_01_manifest.json --since 1800

# 4. repeat until `prepare` prints "0 MISSING cells"
```

Resume-safe at every step: `prepare` only emits still-missing cells, `collect` is idempotent, and the DB is the single source of truth. A throttle/crash mid-wave just leaves those cells missing for the next round.

## 6. Commits (this session, CharXiv)

- `b6c4b6c` `feat(charxiv_scoring)` — `offset` + `prepare` flags on `wf_dispatch.js`.
- `9384357` `data(charxiv_scoring)` — completed pass 1: 12,000/12,000, DB + now-empty manifest.

The demand scores are now joinable to the released correctness labels (`model_score`, `task='reasoning'`, `sub_q=0`) on `item_id` — the demand↔correctness analysis is unblocked.

## 7. Broader session context

This session also included two unrelated pieces of work (documented elsewhere, noted here for completeness):

- **Incremental transcript collection** for `rubric_scoring` — replaced the full-corpus re-screen in `collect()` with a stat-keyed ingestion ledger + inverted tag index; byte-identical output verified against the prod DB (10,013/10,013). Spec/plan: `docs/superpowers/specs/2026-06-18-incremental-transcript-collect-design.md`, `docs/superpowers/plans/2026-06-18-incremental-transcript-collect.md`. (Note: `charxiv_scoring` has its own older full-scan `collect`, mitigated here via `--since`; porting the ledger to CharXiv is a possible follow-up.)
- **Local organization** — committed all previously-untracked work (CharXiv package + DB, SciVer trial-analysis outputs, run manifests, notebook) and fast-forwarded `main`; added `docs/PROJECT_INVENTORY.md` mapping every committed area. Generated prompt `.txt` (~25k files, ~161 MB) were gitignored as regenerable; manifests and DBs kept. Nothing pushed to any remote.

## 8. Next steps

- Run the demand↔correctness join (per-dim demand vs per-model accuracy) — the DeLeAn/ADeLe payoff.
- Decide on pushing `sciver-eval`/`main` for off-machine backup (currently local-only by choice).
- Optional: port the incremental-collection ledger into `charxiv_scoring/collect` so it no longer needs `--since`.
