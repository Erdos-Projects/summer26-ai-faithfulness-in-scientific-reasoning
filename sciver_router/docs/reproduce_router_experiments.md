# Reproduce The Public SciVer Router Experiments

This document reproduces the public-review-safe router results from sanitized
derived features. It intentionally does not rerun private local VLM inference.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The public release ships a router-only dependency file. It intentionally omits
private inference, notebook, vector-database, and dataset-download dependencies
that are not needed to reproduce the public router modeling from
`router_dataset_public.csv`. For exact-version reproduction of the checked
artifacts, use `requirements-lock.txt` instead of `requirements.txt`.

## Verify The Bundled Public Dataset

```bash
.venv/bin/python scripts/verify_public_router_release.py --root .
```

Expected primary input:

```text
Data/derived/model_router/public_router/router_dataset_public.csv
```

The verifier should report 811 rows, 12 comparison rows per selection policy,
24 router `metrics.json` files, and no raw text, image paths, SQLite
annotations, raw JSONL files, or gold labels.

Maintainer note: `scripts/prepare_public_router_release.py` rebuilds the
sanitized release package from private derived inputs. Public reviewers do not
need that script to reproduce router modeling results.

## Run Router Experiments

For a true rerun, use a fresh copy of the public release bundle and omit
`--skip-existing`:

```bash
.venv/bin/python scripts/run_public_router_experiments.py
```

For a quick verification pass over the bundled generated artifacts, use:

```bash
.venv/bin/python scripts/run_public_router_experiments.py --skip-existing
```

This produces:

```text
Data/derived/model_router/public_router/runs/
Data/derived/model_router/public_router/comparisons/
```

The run grid is:

- VLM pool: `qwen3vl`, `pixtral12b`, `kimi_vl_a3b`
- Routers: Ridge/ElasticNet, XGBoost, PyTorch MLP
- Feature sets: `base`, `base_tda`, `base_nlp`, `base_tda_nlp`
- Selection policies: `regret_first`, `top1_first`
- Evaluation: grouped train/validation/test split by `paper_id`, grouped
  10-fold CV inside training, grouped bootstrap on held-out test groups.

The public `split` column stores the original SciVer source split label for
reporting. Router model evaluation uses the separate grouped split recorded as
`router_split` in each run's `splits.csv`.

## Build The HTML Report

```bash
.venv/bin/python scripts/build_sciver_router_presentation_report.py
.venv/bin/python scripts/verify_public_router_release.py --root .
```

Expected output:

```text
docs/sciver_router_presentation.html
```

Open this file in a browser. Figures can be clicked to zoom.

## Important Caveat

Metrics are internal grouped estimates over a router-development dataset. They
are not untouched official SciVer test-set benchmark results.

Generated `.joblib` and `.pt` model artifacts are included for inspection, but
loading serialized model files from untrusted sources can execute code. Prefer
rerunning the scripts from `router_dataset_public.csv` when validating results.
