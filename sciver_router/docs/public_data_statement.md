# Public Data Statement

The public router package is designed for reproducibility of router modeling,
not redistribution of raw benchmark content or private local model outputs.

## What The Public Dataset Contains

`Data/derived/model_router/public_router/router_dataset_public.csv` contains:

- Stable IDs and paper-group IDs.
- Source split indicator used only for reporting breakdowns.
- Claim type and evidence kind.
- Three observed five-run accuracy columns for Qwen3-VL, Pixtral12B, and
  Kimi-VL-A3B.
- Recomputed best-model target columns over those three VLMs.
- Twelve human-coded cognitive difficulty dimensions, included here as
  precomputed public features.
- Numeric base engineered features.
- Numeric chart TDA features.
- Numeric claim NLP features. TF-IDF/SVD and token-vector NLP features were fit
  once on the full feature-extraction corpus before the router split, so these
  columns are transductive exploratory features rather than a strict
  train-fold-only text pipeline.

The `split` column is the original source split label retained for reporting.
Router training/evaluation uses a separate grouped split by `paper_id`, recorded
as `router_split` in each run's `splits.csv`.

The public sidecar feature files can have more rows than
`router_dataset_public.csv` because they are exported before complete-case
filtering over the three public VLM accuracy columns. Router modeling uses the
811-row complete-case table.

## What The Public Dataset Excludes

The public dataset excludes:

- Raw claims and captions.
- Raw prompts, responses, and JSONL run logs.
- SciVer gold labels.
- Image paths and raw image files.
- SQLite annotation databases.
- LM Studio model metadata and local runtime logs.

## Why This Boundary

The router question is: given derived features in the public modeling table, can
a small model predict which local VLM is likely to be best on a chart-claim
pair? The twelve cognitive dimensions are human-coded annotations; a live
deployment would need those annotations or an automatic proxy for them.
The public package therefore preserves the feature matrix and observed VLM
accuracy targets needed for this modeling question while avoiding unnecessary
redistribution of raw benchmark text, private local inference traces, or local
machine-specific paths.

Generated `.joblib` and `.pt` model files may be included as artifacts. Loading
pickle-like or serialized model files from untrusted sources can execute code,
so public reviewers should prefer rerunning the scripts from the sanitized CSV
when validating results.
