# Rubric-Scoring Skill — Development Log & Rationale

**Dates:** 2026-06-13 → 2026-06-14
**Outcome:** A working, tested Claude Code skill (`rubric-scoring`) + `rubric_scoring/` Python package for DeLeAn demand-rubric annotation of SciVer chart claims. On branch `rubric-scoring-skill` (kept unmerged), 45 tests passing, end-to-end validated.

This log records *what* was done and *why*. For the design contract see `RUBRIC_SCORING_SKILL_DESIGN.md`; for the build steps see `superpowers/plans/2026-06-14-rubric-scoring-pipeline.md`.

---

## 1. Objective

Build a reusable, versioned, citable instrument that applies the ADeLe/DeLeAn demand-annotation rubrics (Zhou et al. 2026, *General scales unlock AI evaluation*, Nature 652:58–67) to figure-based scientific claim-verification items — one cognitive/knowledge scale at a time — and records a per-item-per-scale score table plus academic-grade cost/provenance metadata. The skill pins the protocol so a run is reproducible and reportable ("annotations produced by skill vX").

The target corpus is the **SciVer single-figure chart** claims (817 items, val+test). The reference method scores each (instance, dimension) once with an LLM under chain-of-thought prompting.

## 2. Key decisions and why

| Decision | Why |
|---|---|
| **Credit-free Claude Code subagents** (Max plan), not the raw API | Avoid spending API credits; the project runs on the Max subscription. Trade-off accepted: temperature, a hard output cap, and reasoning effort are **not controllable** on the subagent path (verified — see §4), so we *document and measure* those instead of asserting them. |
| **One isolated subagent per (item × dimension)** | Guarantees cross-scale independence by construction (a fresh context can't leak one scale's reasoning into another) and gives **exact per-call token attribution** (one transcript = one cell). Mirrors the paper's one-call-per-(instance,dimension). |
| **6 dimensions on the first pass; all 18 bundled; UG excluded** | The 6 (below) target chart claim-verification; the library holds all 18 so any subset (or the full 18) is a one-arg change. UG (Unguessability) is constant on a binary task → carries no information. |
| **Verdict-blind prompts** | The annotator must score *demand*, not the answer. Prompts carry only the claim + figure; `label`/`origin_statement`/`perturbed_statement`/`perturbed_explanation` never enter a prompt. Enforced structurally (the prompt builder never reads `label`; proven by a label-independence test). |
| **Multimodal adaptation in the instance block only; rubric text byte-faithful** | DeLeAn v1.0 is textual-only. We frame the instance as "verify this claim against this figure (viewed via Read)" but keep the rubric anchors verbatim — the modality mismatch is a documented limitation, not silently papered over. Captions omitted (not part of the task; logged as a deviation). |
| **SQLite store with `pass` + `session_id`, `UNIQUE(pass,item,dim)` + upsert, resume-by-skip** | A *pass* = one full sweep (multiple passes denoise the run-to-run variance); `session_id` gives per-session provenance because one pass spans many sessions under rate limits. Upsert + the unique key make re-runs idempotent and let a chunked run resume by scoring only cells not already `parse_ok=1` — essential at ~50k tokens/agent. |
| **Charts only (817) for now** | "Annotate everything," but tables deferred. Charts-only keeps the first campaign to 817×6 = 4,902 agents. |

### The six first-pass dimensions

| Code | Name | Group | Why |
|---|---|---|---|
| AS | Attention and Scan | Elemental | locating the right panel/value among distractors |
| QLq | Quantitative Reasoning | Elemental | numeric comparison against chart values |
| QLl | Logical Reasoning | Elemental | deductive comparison/derivation |
| MCr | Identifying Relevant Information | Elemental | the paper's most discriminating dim; relevance filtering |
| AT | Atypicality | Extraneous | contamination / construction-artifact confounder |
| VO | Volume | Extraneous | amalgamation / length confounder |

## 3. Pilot — validating the approach before building

Before writing any pipeline, we ran a pilot on **3 SciVer chart items × 6 rubrics**, repeated **5 times** (90 annotations), dispatching one isolated Sonnet subagent per cell and scraping each subagent transcript for the score + exact token usage.

**Mechanics & cost (measured, highly reproducible across the 5 runs):**
- 100% score-parse rate.
- ~**47–50k tokens processed per agent** (of which only ~500 is output; the bulk is fixed Claude Code harness overhead — system prompt + tool defs — cache-created fresh per isolated agent).
- ~**20 s per agent**; **$0 actual** on the Max plan (~$0.06 API-equivalent).

**Reproducibility (the measured cost of having no temperature-0):**
- **82% mean modal agreement, 89% of cells within ±1, 68% mean pairwise exact-match** across the 5 passes.
- Most stable: AT, VO, QLq. Noisiest: QLl, then AS/MCr.
- Implication: a single pass gives ±1 reliability; **mode-of-k passes** denoises further at k× cost. This is why the schema is multi-pass from the start.

**Per-call token capture mechanism:** verified that subagent transcripts at `~/.claude/projects/**/<session>/subagents/agent-*.jsonl` carry a full `usage` block per message; summing across the agent's messages yields exact input/cache/output tokens. This is what makes the credit-free path academically honest (real tokens → API-equivalent cost).

## 4. The reasoning/effort investigation (and its conclusion)

A focused investigation (Claude Code 2.1.177) established what is and isn't controllable for Task-dispatched subagents:
- **Temperature 0:** not exposed → cannot be set.
- **Hard 1000-token output cap:** not enforceable; only instructable (observed output ~500 tok, well under).
- **Extended reasoning / effort:** **not enableable** on the subagent path. Across every probe — sonnet, opus, the `ultrathink` keyword, and an inherited `xhigh` session on a hard task — subagents produced **0 thinking tokens**. The Agent tool has no effort parameter, and the main session's effort does **not** propagate to subagents. So the agents run with no extended reasoning (only the visible chain-of-thought the prompt requests) — which matches the paper's CoT *prompting* and is verified per run by the QA `thinking_clean` check.
- **Controllable:** model (Sonnet) and the *orchestrator's* own effort (which can be lowered to speed the mechanical dispatch loop; it never reaches the subagents).

This is why the design treats temperature/cap/effort as documented deviations rather than parameters.

## 5. What was built

A path-portable Python package `rubric_scoring/` (no hardcoded absolutes; resolves repo/data/db paths from config or env), implemented TDD with a spec-compliance + code-quality review on every task:

- `config.py` — path resolution.
- `extract_rubrics.py` + `rubrics/` — all **18** DeLeAn rubrics extracted verbatim from the supplementary PDF (footer-stripped, UTF-8, first-line content-checked).
- `items.py` — the canonical **817-chart** item set (val+test, `type==chart`), verdict fields excluded from the prompt-facing object.
- `db.py` — `item` / `pass` / `annotation` tables; `UNIQUE(pass,item,dim)` upsert; `completed_cells` (resume); real UTC timestamps.
- `prompts.py` — verdict-blind prompt builder (proven label-independent) with unique cell tags.
- `parse.py` — lenient 0–5 score parser.
- `scrape.py` — session-scoped transcript scraper → per-cell score + exact tokens + thinking count (reads each transcript once).
- `prepare.py` / `collect_cli.py` — resume-aware CLIs (prepare writes prompts only for missing cells + a manifest; collect scrapes the session and upserts).
- `aggregate.py` — per-cell mode/mean across passes → CSV.
- `qa.py` — per-pass parse-rate, `thinking_clean` (==0), per-dim spread/flat flags.
- `.claude/skills/rubric-scoring/SKILL.md` — the orchestration protocol (prepare → dispatch one Sonnet subagent per cell → collect → repeat/resume → qa/aggregate) with the invariants.

**Verification:** 45 unit tests pass. A live **6-agent end-to-end smoke** (3 items × 2 dims) exercised the full loop and confirmed parse_rate 1.0, `thinking_clean: true`, correct token/session capture, and that re-running `prepare` reports **0 MISSING** (resume works). The pilot record (DB, scripts, 5 passes) is retained under `rubric_scoring/pilot/`. The production DB and generated prompts are gitignored.

## 6. Full-run projection (charts only)

817 items × 6 dims = **4,902 agents per pass**: ~235M tokens processed, **~$0 actual** ($266 API-equivalent), and a **multi-session, rate-limited campaign** (rate limits, not dollars, are the binding constraint; resume means finished cells are never re-scored). All 18 dims would be 3× that.

## 7. Deviations from the paper (documented, not hidden)

Temperature ≠ 0 (measured nondeterminism 82% modal / 89% within ±1); Sonnet 4.6 annotator vs GPT-4o; ~10–15k tokens/agent fixed harness overhead from the credit-free isolated-subagent path; visible-CoT only (no thinking mode); DeLeAn-v1.0 textual anchors applied to multimodal items; captions omitted. All recorded in each pass's `deviations` field.

## 8. How to run a real pass

From the repo root, `source .venv/bin/activate`:
1. `python -m rubric_scoring.prepare --pass N --dims AS,QLq,QLl,MCr,AT,VO` (use any subset, or all 18).
2. Dispatch one Sonnet subagent per missing-cell prompt via the Agent tool, per `SKILL.md`, in rate-limit-friendly batches.
3. `python -m rubric_scoring.collect_cli --pass N --manifest <manifest path>`.
4. Repeat across sessions until `prepare` reports 0 MISSING; then run the QA + aggregate snippets in `SKILL.md`.

**Operator tip:** the run session can set `/effort low` (or `medium`) — the dispatch loop is mechanical and doesn't need the orchestrator's deep reasoning; this won't affect scoring (effort never reaches subagents). The wall-clock is dominated by the subagents + rate limits, so this is a modest coordination speedup, not a transformation.

## 9. Open items / future

- **Batching** (one agent per rubric over N items) is the only lever that materially cuts the full-run token/time — it amortizes the per-agent overhead via cache-read — but it trades away the per-cell isolation and exact per-call token attribution that were made firm. Not implemented.
- **Tables** (683 items) deferred; charts only for now.
- **Number of passes (k)** chosen at run time (single pass = ±1 reliability; more passes denoise).
- The branch is **unmerged** — finish via merge or PR when ready.

## 10. Provenance

Branch `rubric-scoring-skill` (off `main`, ~20 commits). Key artifacts: this log, `RUBRIC_SCORING_SKILL_DESIGN.md`, `superpowers/plans/2026-06-14-rubric-scoring-pipeline.md`, `../rubric_scoring/` (package + `pilot/` record), `.claude/skills/rubric-scoring/SKILL.md`.
