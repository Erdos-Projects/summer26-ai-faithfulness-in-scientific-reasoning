# SciClaimEval Demand-Scoring + Prediction — Dev Log

- **Date:** 2026-06-16 / 2026-06-17
- **Branch:** `sciver-eval` (feature built on `sciclaimeval-rubric-scoring`, merged via `459af56`; nothing pushed)
- **Spec:** `docs/superpowers/specs/2026-06-16-sciclaimeval-rubric-scoring-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-16-sciclaimeval-rubric-scoring.md`

## 1. Goal

Reproduce, on the **SciClaimEval** dataset, the two coupled pipelines already built for SciVer:
1. **Demand rubric scoring** (`rubric_scoring/` analog) — DeLeAn demand annotation, one isolated Sonnet subagent per (item × dimension), verdict-blind, resumable, per-call token provenance in SQLite.
2. **Model prediction** (`sciver_eval/` analog) — a model answers the Supported/Refuted task; per-item correctness joins to the demand scores on `item_id`.

The point of the join is the DeLeAn/ADeLe payoff: correlate how much each cognitive demand a task requires against how often the model gets it right.

**Requested constraint:** keep SciClaimEval fully compartmentalized from the SciVer work — its own folders and own DBs.

## 2. Why these design choices

- **Full standalone copy (not a shared engine).** The user chose compartmentalization over DRY. The SciVer modules were already path-portable and dataset-agnostic except the item loader, so the port is a physical copy + three mechanical edits (config paths, tag prefix, dataset loader) per module. Accepted cost: ~380 duplicated lines; a future shared-logic fix must be applied in both copies.
- **Two sibling modules** mirroring SciVer's `rubric_scoring` / `sciver_eval` split: `sciclaimeval_scoring/` (demand) + `sciclaimeval_eval/` (prediction). `sciclaimeval_eval` imports `config`/`items` from `sciclaimeval_scoring` (exactly as `sciver_eval` imports `rubric_scoring`); isolation is from the *SciVer* modules, not within the SciClaimEval pair.
- **Item universe = 265 dev figures.** Filter `dev_task1_release.json` to `evi_type=="figure"` (verified 265; labels Supported 149 / Refuted 116; domains ml 86 / nlp 82 / peerj 97). `item_id = "scev_" + claim_id` (e.g. `scev_val_fig_0001`), derived directly from the released dataset (not `feature_discovery`) so the module is self-contained while still matching the ids used elsewhere. Table items (482) and the gated test split are out of scope, matching SciVer's charts-only decision.
- **Verdict-blind / verdict-clean.** Scoring prompts get only claim + figure. Critically, the `operation` field leaks the verdict (`Supported_claim_only` ⇒ Supported), so it is stored in the DB but never put in a prompt — same treatment as `label`. Prediction prompts may show claim + caption + context + figure but never `label`/`operation`.
- **Tag namespaces** chosen disjoint from the SciVer DBs: scoring `rsc_p{pass:02d}__{item_id}__{code}`, prediction `sce_r{run:02d}__{item_id}__t{trial:02d}`.

## 3. How it was built (process)

brainstorming → written spec → written plan → **subagent-driven TDD** (12 tasks, fresh implementer + spec/quality reviewer per task) → final whole-branch review (Opus) → cleanup commit → merged to `sciver-eval`.

- 12 tasks, each failing-test → minimal code → green → commit. Pure-copy files (`db`, `scrape`, `parse`, `aggregate`, `qa`) copied + import-repointed; only `items.py` is a real rewrite.
- Final review verdict: *ready to merge with fixes*; no Critical/Important. The three load-bearing properties (dual verdict-blindness, SciVer isolation, `item_id` join + resume) were verified by inspection and empirical check. A one-commit cleanup folded the minors (stale docstring, dead imports, README id, `n_items`-under-limit).
- Result: full suite **31/31** green; zero imports from / writes to the SciVer modules/DBs.

## 4. Smoke test (pipeline sanity)

Ran a small live end-to-end check before any full run: 6 scoring agents (2 dims × 3 items) + 3 prediction agents. Confirmed prepare→dispatch→collect, `parse_rate 1.0`, `thinking_clean: true` (reasoning stayed off), exact token capture, correctness scored against the withheld label, and the cross-DB `item_id` join. One Refuted item the model got wrong by trusting the caption — the intended hard-negative behavior.

## 5. Full demand pass (pass 1)

- **Dims = the 12 used in the SciVer analysis, minus SNs.** Derived from `rubric_scoring/annotations_prod.db`: SciVer scored 13 dims, but SNs only reached 200/817 (dropped), leaving 12 at full coverage: **AS, QLq, QLl, MCr, AT, VO, VL, GS, MA, MCu, CL, KNf**. This is what "all 12, drop SNs" meant. Includes the 3 authored emergent visual dims (VL/GS/MA) and KNf/CL/MCu — not a clean DeLeAn group, which is why it was pinned from the SciVer DB rather than guessed.
- **Mechanism:** background **Workflow** dispatcher `sciclaimeval_scoring/wf_dispatch.js`, self-fetching — stage 1 refreshes the resume-aware manifest and returns the next N missing tags; stage 2 fans them to isolated Sonnet subagents. Chosen because 265×12 = **3,180** agents is a multi-session, rate-limited campaign that manual dispatch could not finish. Self-fetching keeps the (large) work-list inside the workflow instead of the controller's context.
- **Run:** 7 batches (24 → 300 → 500 → 500 → 800 → 800 → 240), collected between batches (resume-aware; `collect_cli` is `O(tags × transcripts)` so it slows as the transcript pool grows). ~40M tokens processed ($0 actual on Max).
- **Result: 3,180 / 3,180**, all 265 × 12, `parse_rate 1.0`, `thinking_clean: true`, no flat dims. Per-dim means (aggregate.csv): AS 2.44, QLq 1.74, QLl 1.53, MCr 1.93, AT 3.00, VO 1.43, VL 2.18, GS 1.09, MA 2.13, MCu 2.03, CL 0.50, KNf 1.82. **AT is near-constant at 3** (258/265) — the same low-sensitivity DeLeAn flagged and the SciVer pilot saw for AT.

## 6. Prediction prompt — why it changed

The user asked to confirm we run the prediction task the way the SciClaimEval authors did. Investigation of the paper (arXiv 2602.07621), its appendix, and the GitHub eval repo found:
- **No verbatim prompt template is published.** The paper says only: *"Following the SciVer dataset, we also employ zero-shot Chain-of-Thought prompting,"* with *"two setups: no-context for samples not requiring context and use-context for samples requiring either a short context or the full paper."*
- **No claim-type field** in SciClaimEval, and the repo's eval uses `prompt_type = "zeroshot"` (a single constant) → they applied **one uniform CoT prompt** to every sample, varying only context. So there is no SciVer template-variant to mirror; our single balanced CoT is structurally correct.
- The repo's `reproduce_all_models_task1.py` reports metrics split into **no-context / with-context / all** buckets keyed on `use_context`.

The one real gap: our prediction prompt **always** injected `context`, but 187/265 figures are `use_context=="no"`. Fix (`be6773b`): `build_prompt` drops the `Context:` block entirely when `use_context=="no"` (or context empty), includes it otherwise; `prompt_version → sciclaimeval-cot-v2-usecontext`. Wording stays our SciVer-style CoT since the authors' exact text is unreleased — only the *protocol* (single zero-shot CoT + use_context split) is matched. Eval suite **15/15**.

## 7. Prediction run 1 (Haiku, single trial, use_context-aware)

265/265, parse 1.0. **Overall accuracy 0.611.** Recall Supported 0.570 / Refuted 0.664 (mild skeptic bias; predicted-yes 0.468). By bucket: `use_context=no` 0.599 (n=187), `with` 0.641 (n=78).

**Demand ↔ correctness**, `r(demand, correct)` over 265 items (negative = higher demand → more errors): MCu −0.231, AS −0.202, MCr −0.185, AT −0.114, QLl −0.107, VO −0.098, VL −0.084, QLq −0.064, MA −0.057, CL −0.048, KNf +0.002, GS +0.028. Eleven of twelve point the expected way; MCu/AS/MCr are the strongest difficulty signals. Single trial → `correct` is binary, so treat magnitudes as provisional.

## 8. Artifacts (committed on `sciver-eval`)

- `sciclaimeval_scoring/` — module + `wf_dispatch.js`; `annotations_sciclaimeval.db` (3,180 cells) + `aggregate.csv` (committed, matching how SciVer tracks its DB/CSV).
- `sciclaimeval_eval/` — module + `wf_dispatch.js`; `predictions_sciclaimeval.db` (run 1, 265 cells).
- `.claude/skills/sciclaimeval-rubric-scoring/SKILL.md`.
- Prompt files under each module's `prompts/` are runtime artifacts and left **untracked** (parity with SciVer).

## 9. How to resume / extend

- **Scoring:** `python -m sciclaimeval_scoring.prepare --pass <N> --dims <12>` then `Workflow(sciclaimeval_scoring/wf_dispatch.js, {batch})` → `collect_cli`. Resume-aware; finished cells never re-dispatched. A new pass = a fresh sweep for denoising.
- **Prediction:** `Workflow(sciclaimeval_eval/wf_dispatch.js, {run, trials, batch, model})` → `collect_cli --run <N>`. `UNIQUE(run, item_id, model, trial)` supports repeated trials and new models.
- **Join:** `predictions_sciclaimeval.db` ⋈ `annotations_sciclaimeval.db` on `item_id`.

## 10. Caveats / open items

- Single prediction trial — more trials needed for tighter `p(correct)` and stable correlations.
- Prediction wording is our CoT, not the authors' (theirs is unpublished); protocol matches.
- AT carries little signal (near-constant) — consistent with prior DeLeAn/SciVer observations.
- Multi-pass denoising for the demand scores not yet run (pass 1 only).
- Gated SciClaimEval test split and the 482 table items remain out of scope.
- Nothing pushed; all local on `sciver-eval`.
