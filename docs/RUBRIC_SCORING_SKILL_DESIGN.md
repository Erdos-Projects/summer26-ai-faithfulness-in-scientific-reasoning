# DeLeAn Rubric-Scoring Skill — Design Spec

- **Date:** 2026-06-14
- **Status:** Draft for review
- **Author:** awndre (with Claude Code)
- **Reference:** Zhou et al. 2026, *General scales unlock AI evaluation with explanatory and predictive power*, Nature 652:58–67. Method notes in `Annotations/Notes/`. Pilot evidence in `Annotations/results/` (runs `pilot_0001`–`pilot_0005`).

---

## 1. Purpose

A reusable, versioned Claude Code **skill** that performs routine **rubric-driven feature extraction** over figure-based scientific claim-verification items, following the ADeLe/DeLeAn demand-annotation method. For each item it applies a fixed set of demand rubrics — **one scale at a time, in isolation** — and records a tidy per-cell score table plus academic-grade provenance and per-call cost/usage metadata. The skill pins the protocol (template + rubric text + parameters + output schema + storage) so the procedure is reproducible and citable ("annotations produced by skill vX").

## 2. Scope

- **In scope (first production pass):** the **6 demand dimensions** below, applied to the **SciVer single-visual CHART items (817)** from the summer26 processed set.
- **Out of scope:** tables (deferred — charts only "for now"); the `feature_discovery/` codebook and any of its artifacts (explicitly excluded); captions (deliberately omitted, logged as a deviation); the raw-API execution path (project is credit-free); UG (Unguessability — constant on a binary task, carries no information).

## 3. Method background

Each DeLeAn rubric is a self-contained measurement instrument (construct description + six level definitions 0–5 + three anchor examples per level, ≈800–2,120 tokens). One rubric scores one task instance on one cognitive/knowledge dimension. The reference paper annotated with GPT-4o at temperature 0, max-output 1,000 tokens, chain-of-thought prompting, one call per (instance, dimension). This skill reproduces that **protocol** on a credit-free path, documenting every place the path forces a deviation.

## 4. Dimensions (first pass)

Source: `Annotations/Notes/ADeLesupplementary.pdf` §2 (PDF pp. 77–98), extracted verbatim into `Annotations/rubrics/`.

| Code | Name | Group | Why included |
|---|---|---|---|
| AS | Attention and Scan | Elemental | locating the right panel/value among distractors |
| QLq | Quantitative Reasoning | Elemental | numeric comparison against chart values |
| QLl | Logical Reasoning | Elemental | deductive comparison/derivation |
| MCr | Identifying Relevant Information | Elemental | the paper's most discriminating dim; relevance filtering |
| AT | Atypicality | Extraneous | contamination/construction-artifact confounder |
| VO | Volume | Extraneous | amalgamation/length confounder |

The skill bundles **all 18** rubrics as a library; each run declares its **active subset** (the 6 above for the first pass). Every agent's prompt contains exactly one rubric, so bundling all 18 has no leakage consequence. Running a different subset, or the full 18, requires only changing the active-dim list — no code change.

**Full library — all 18 demand dimensions** (extract verbatim from supplementary §2; currently only the 6 piloted are extracted):
- **Elemental (11):** AS p78 · CEc p79 · CEe p80 · CL p81 · MCt p82 · MCu p83 · MCr p84 · MS p85 · QLq p86 · QLl p87 · SNs p88
- **Knowledge (5):** KNn p90 · KNs p91 · KNf p92 · KNa p93 · KNc p94
- **Extraneous (2):** VO p96 · AT p97

(UG p98 is the 19th — answer-format only, not a demand scale — excluded; constant on a binary task.) Running all 18 over 817 charts = **14,706 agents/pass** (3× the 6-dim run); cost/time in §16 scales accordingly.

## 5. Target data

- **Item set:** the 817 single-visual **chart** claim-verification pairs from the summer26 processed SciVer set (val+test, single visual evidence, `type==chart`). The canonical id/claim/figure list is taken from the summer26 processed manifest (NOT `feature_discovery`); if read via parquet requires `pyarrow`/`pandas`, otherwise replicate summer26's documented filter over `SciVer/{valset,testset}.json`.
- **Per item, the prompt receives only:** `claim` (text) + the figure image (local PNG under `SciVer/images/`, viewed via the Read tool).
- **Withheld from the annotator (verdict-blind):** `label`, `origin_statement`, `perturbed_statement`, `perturbed_explanation`. These live only in the DB. Captions are omitted (logged deviation; extractable from paper JSONs if revisited).
- **Stable `item_id`:** deterministic, e.g. `sciver_{split}_{source_index}`.

## 6. Execution architecture

- **Path:** credit-free **Claude Code subagents** (Max plan). No raw API, no credits.
- **Unit of isolation:** **one subagent per (item × dimension)**. Fresh context → cross-scale independence is guaranteed by construction (no agent sees another's rubric, reasoning, or score) and per-call token attribution is exact (one transcript = one cell). For pass *p*: 817 × 6 = **4,902 agents**.
- **Per agent:** receives one filled prompt (rubric + claim + figure path), uses the Read tool to view the figure, writes chain-of-thought, emits the conclusion line, stops. Instructed to use only Read and to perform no verification.
- **Leakage-proofing:** agents are never given a path to the DB/results; the orchestrator (main loop) performs all DB I/O; agents receive only `rubric + claim + figure`. (Optional future hardening: a restricted subagent type lacking Bash/DB tools — see §15.)

## 7. The prompt

DeLeAn template (Supplementary Table 23), with the **instance block adapted for multimodal** and the **rubric text kept byte-faithful** (text-task anchors unchanged):

```
ROLE: demand-level annotator. Score ONE rubric on ONE task instance, then STOP.
Do not verify the claim. View the figure with Read. Use no tool other than Read.

QUERY: The following rubric describes six distinct levels of *{dimension}* ...
{rubric}                      # verbatim from Annotations/rubrics/{code}.txt

TASK INSTANCE: scientific claim-verification task — determine whether the claim is
supported or refuted by the figure (you only assess the rubric demand, not the answer).
Claim: "{claim}"
Figure: view with Read at {abs_path}

INSTRUCTION: Score 0–5 with chain-of-thoughts reasoning, then conclude exactly:
"Thus, the level of *{dimension}* demanded by the given TASK INSTANCE is: SCORE". Then STOP.
```

## 8. Subagent parameters & controllability (verified in v2.1.177)

| Control | State | Notes |
|---|---|---|
| Model | **sonnet** (set via Agent `model`) | controllable |
| Temperature 0 | **not controllable** | no subagent param; documented deviation |
| Hard 1000-tok output cap | **not enforceable** | instructed only ("stop after the score line"); pilot output 422–570 tok/agent, well under |
| Reasoning / extended thinking | **off — not pursued** | empirically 0 thinking tokens across all probe agents incl. inherited opus[1m] @ xhigh on a hard task; `ultrathink` keyword had no effect; Agent tool has no effort param. Visible CoT is faithful to the paper's CoT *prompting*, so extended thinking is out of scope. |

## 9. Score parsing

Lenient regex on the agent's returned text: match a score-bearing pattern such as `is:?\s*\*?\*?\s*([0-5])` (fallback `level .*? is:?\s*([0-5])`), take the last match. Pilot parse rate **100%** over 90 agents. Every non-matching call is logged (`parse_ok=0`, raw output retained), never silently dropped.

## 10. Data model (production schema)

SQLite system-of-record in a **fresh production DB** (`Annotations/results/annotations_prod.db`); the pilot DB (`annotations.db`, rows under `run_id=pilot_000X` / `skill_version=pilot-0.1`) is left untouched.

- **`item`**: `item_id` PK, source, paperid, claim_type, vtype, image_path, claim, **label** (stored, never sent to agents).
- **`pass`**: `pass` PK (int), created, skill_version, model, active_dims, item_source, n_items, params, deviations.
- **`annotation`**: `id` PK; **`pass`** (int); **`session_id`** (text, from `CLAUDE_CODE_SESSION_ID`); `item_id`; `dim_code`; `dim_name`; `agent_id`; `raw_output`; `score`; `parse_ok`; `input_tokens`; `cache_creation_tokens`; `cache_read_tokens`; `output_tokens`; `total_input_tokens`; `output_thinking_tokens` (QA — expect 0); `wall_clock_s`; `created`. **`UNIQUE(pass, item_id, dim_code)`**, written via **upsert** (`ON CONFLICT DO UPDATE`).
- **Token capture:** per-call usage scraped from subagent transcripts at `<transcript-dir>/<session-id>/subagents/agent-<agentId>.jsonl` (summing `input_tokens`/`output_tokens`/`cache_*` across the agent's messages; the Agent tool result also returns inline `subagent_tokens`/`duration_ms`).
- **Exports:** parquet/CSV for EDA.

## 11. Multi-pass & denoising

A **pass** = one full sweep of all (item × active-dim) cells. Multiple passes denoise the nondeterminism (no temperature 0). Aggregation = group by `(item_id, dim_code)` across passes → **mode** (primary) and **mean** (secondary) per cell. Measured stability (5 pilot passes, 18 cells): 82% mean modal agreement, 89% within ±1, 68% mean pairwise exact; most stable AT/VO/QLq, noisiest QLl/MCr. Number of passes is a run-time choice; the schema supports any k.

## 12. Orchestration & resumability

- A pass is **chunked across many sessions** (Max-plan rate limits, not dollars, are the binding constraint).
- **Resume = skip-completed:** at the start of each session/chunk, query `annotation` for cells with `parse_ok=1` in the current `pass` and dispatch only the **missing** cells. Never re-dispatch a finished cell (~47–50k tok each). Re-scoring a cell upserts cleanly.
- Dispatch via the Agent tool, in concurrent batches (harness caps concurrency); collect results, parse, scrape tokens, upsert.

## 13. Record-keeping & reproducibility

- The skill pins: prompt template, the 18 verbatim rubrics, active-dim set, parameters, parser, storage schema. Protocol is versioned (`skill_version`).
- Each `pass` row logs **deviations** explicitly: temperature ≠ 0 (with the measured stability figure), output cap instructed-not-enforced, captions omitted, DeLeAn-v1.0-textual-only modality mismatch, reasoning off.
- Per-cell provenance: `pass`, `session_id`, `agent_id`, raw output, exact tokens, wall-clock.

## 14. QA / verification (per pass)

1. **Parse rate** (target ~100%; investigate if low).
2. **Range/distribution** per dim — flag rubrics that collapse to one value (paper's "sensitivity" concern; AS looked flat in the pilot — watch at scale).
3. **Thinking-tokens == 0** check per agent (confirms the run stayed lightweight, since effort isn't a settable control).
4. **Verdict-blindness audit** — confirm no withheld field ever entered a prompt.
5. **Test–retest** on a subset across passes — report exact-match / within-±1.
6. **Rationale–score coherence** spot check on a hand sample.

## 15. Deviations from the paper (documented, not hidden)

- **Temperature:** paper used 0; subagent path can't — measured nondeterminism is 82% modal / 89% within ±1 over 5 passes; mitigated by multi-pass mode.
- **Annotator:** Sonnet 4.6 here vs GPT-4o in the paper (a model swap; the paper's bar for an alternative annotator was rWG≈0.83).
- **Harness overhead:** subagent path re-pays ~10–15k tokens of fixed Claude Code overhead per agent (no cross-agent cache sharing); ~47–50k tokens processed/agent total. This is the cost of credit-free isolation.
- **Reasoning:** off (visible CoT only) — matches the paper's CoT *prompting*, not a thinking model.
- **Modality:** DeLeAn v1.0 is textual-only; rubric anchors are text-task examples applied to figure items.
- **Captions:** omitted.

## 16. Cost & time projection (measured, charts-only, per pass)

From 90 pilot agents: ~47–50k tokens processed/agent, ~20 s/agent, ~$0.055–0.06 API-equivalent/agent (Max-plan **actual $0**), output ~500 tok.

| Per pass (817 × 6 = 4,902 agents) | value |
|---|---|
| Tokens processed | ~235M (~54M "new") |
| API-equivalent | ~$270–290 (actual $0 on Max) |
| Wall-clock | ~27 h sequential; compresses with concurrency but **rate-limited → multi-session campaign over days** |

Multiply by the number of passes.

## 17. Skill packaging

`SKILL.md` (protocol + step-by-step procedure) plus assets/scripts:
- `rubrics/` — all 18 verbatim rubric texts (library).
- prompt-builder — fills template per (item, dim), verdict-blind, writes tagged prompt files / inline prompts.
- dispatcher guidance — how the orchestrator launches isolated Sonnet subagents in resumable, concurrent batches, skipping completed cells.
- scraper/parser — extracts scores + per-call tokens from transcripts, upserts to the DB.
- aggregator — mode/mean per cell across passes; parquet/CSV export.
- QA checks — §14.

## 18. Open / to-verify items

1. **Canonical 817-chart list extraction** — confirm reading the summer26 processed manifest (parquet) vs replicating the filter over SciVer JSON; pin the `item_id` scheme.
2. **Passes count (k)** — chosen at run time based on desired denoising vs campaign length.
3. **Repo home / portability** — see §20; decide whether to build directly in summer26.

## 19. Success criteria

A pass completes with ~100% parse rate, verdict-blindness verified, per-cell scores + exact per-call tokens stored under `(pass, item_id, dim_code)`, resumable across sessions without re-work, deviations logged, and cross-pass mode/mean exportable for downstream analysis.

## 20. Portability & repo home

The skill's intended permanent home is the **summer26-ai-science-reasoning** repo (already git-tracked; contains the SciVer data). To avoid a fragile move, build it **path-portable from the start**:

- **No hardcoded absolute paths.** Resolve a repo root and data root from config/env (or relative to the skill's own location). The pilot scripts' `/home/awndre/projects/Erdos/...` absolutes become relative/configurable.
- **Skill location:** install under `summer26-ai-science-reasoning/.claude/skills/<name>/` so Claude Code registers it as a skill — a plain folder elsewhere is **not** a registered skill.
- **Transcript/token-scraping path is derived from the Claude Code project directory** (cwd). Running from a different repo changes that path, so the scraper must resolve the transcript dir from the live session (`CLAUDE_CODE_SESSION_ID` + project path) rather than a fixed string.
- **Data root:** point figure paths at summer26's `data/raw/SciVer` (or a shared SciVer), not the Erdos-relative `SciVer/`.
- **Memory** is keyed to the current project path and will not auto-follow a move.

**Drag-and-drop caveat:** copying the folder moves files but breaks (a) absolute paths in scripts, (b) the transcript-derived token-scraping path, (c) skill registration (must live in `.claude/skills/`), and (d) memory association. **Recommended:** build directly in summer26 with relative/config paths (no move needed), or do a deliberate path-fix migration — never a raw drag-and-drop.
