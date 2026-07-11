# Project Inventory — committed work map

_Last updated: 2026-06-18._

This note is a map of what lives in this repository and where, written so that
any committed work can be located and understood at a glance. It complements the
root [README.md](../README.md) (which covers the original SciVer vector-DB
baseline) by documenting the **demand-rubric scoring** efforts and analysis
outputs that were later added.

All paths below are committed to git (on `main` and `sciver-eval`, which are
currently identical). Generated per-cell prompt `.txt` files (~25k, ~161 MB) are
intentionally **gitignored** — they are regenerable via each package's
`prepare.py`. The run **manifests** (`*manifest*.json`) and the **databases**
ARE committed, so every run is reconstructable.

---

## Original baseline (pre-existing)

| Area | What it is |
|---|---|
| `sciver_vector_db/`, `scripts/`, `tests/` | GPU-free / LLM-free SciVer claim-verification baseline: parsing, embedding, Qdrant vector DB, CLI entry points. See root README and `docs/SCIVER_*`. |
| `data/raw/`, `data/processed/` | Local SciVer snapshot + generated vector DB / caches. **Gitignored** (downloaded/generated). |
| `notebooks/` | Usage, mapping, baseline-modeling, and (new) `05_future_directions.ipynb`. |
| `docs/FAITHFULNESS_PROBE_FINDINGS.md`, `docs/POSTHOC_AUDITS_AND_LIMITATIONS.md` | Statistical investigation showing the ~0.55 baseline accuracy is not robust signal, plus post-hoc audits and finalized limitations. |

## Demand-rubric scoring efforts

These four packages share a common shape (a DeLeAn-style demand-rubric pipeline:
`config / items / prepare / prompts / scrape / collect_cli / aggregate / db`,
a `rubrics/` library, and a SQLite annotations DB). Each scores
`(item × rubric-dimension)` cells with isolated subagents.

| Package | Target dataset | Status (2026-06-18) | Database |
|---|---|---|---|
| `charxiv_scoring/` | **CharXiv** charts (the original project focus) | **partial** — 6,432 / 11,989 cells scored | `annotations_charxiv.db` (40 MB) |
| `rubric_scoring/` | SciVer charts (val+test, 817 items) | **complete** — 10,013 / 10,013 cells scored (passes 1–3) | `annotations_prod.db` (25 MB) |
| `sciclaimeval_scoring/` | SciClaimEval dev figures (265 items × 12 dims) | **complete** — 3,180 annotation rows | `annotations_sciclaimeval.db` |
| `sciclaimeval_eval/` | SciClaimEval prediction task (265 figures) | run 1 complete — 265 predictions | `predictions_sciclaimeval.db` |

Notes:
- **`rubric_scoring/`** also received this session's feature: an incremental,
  ledger-based transcript collector (see
  `docs/superpowers/specs/2026-06-18-incremental-transcript-collect-design.md`
  and the matching plan). `tests/regression_collect.py` proves it reproduces the
  prod DB byte-for-byte. Run logs: `docs/RUBRIC_SCORING_DEV_LOG.md`.
- **`sciclaimeval_*`** dev log: `docs/SCICLAIMEVAL_DEV_LOG.md`.
- **`charxiv_scoring/`** is the in-progress effort; resume its scoring with that
  package's `prepare.py` → dispatcher → `collect_cli.py`. See its `README.md`.

## Analysis outputs

| Area | What it is |
|---|---|
| `sciver_eval/` | Paired entailed/refuted evaluation harness + `trial_analysis/` outputs: figures (PNG), CSVs (Kendall-τ, Spearman, RF quadrants), a trained model (`rf_5of5_frozen.joblib`), and `viz_*.py` scripts. Design: `docs/superpowers/specs/2026-06-16-paired-entailed-refuted-eval-design.md`. |

## Conventions

- **Gitignored & regenerable:** `*/prompts/**/*.txt` (per-cell prompts),
  `data/raw|processed/`, `.venv/`, vector DB / caches, teaching decks
  (`docs/*.pptx`). See `.gitignore`.
- **Committed & authoritative:** all `annotations_*.db` / `predictions_*.db`,
  all `*manifest*.json`, all `rubrics/*.txt`, analysis outputs, source code.
- **Backup status:** committed locally on `main` + `sciver-eval`; **not yet
  pushed to any remote.** Push when an off-machine backup is wanted.
