# SciVer Public Router Release

This public package reproduces router modeling from sanitized derived features.
It does not rerun local VLM inference and does not include raw SciVer claims,
captions, prompts, responses, images, labels, SQLite annotation databases, or
local model logs.

## What Is Included

- `router_dataset_public.csv`: the 811-row public modeling table.
- `chart_tda_features_public.csv` and `claim_nlp_features_public.csv`: public
  sidecar feature tables.
- `Data/derived/model_router/public_router/runs/`: generated router artifacts.
- `Data/derived/model_router/public_router/comparisons/`: comparison tables and
  plots.
- `docs/sciver_router_presentation.html`: self-contained evidence report.
- `outputs/sciver_router_models_presentation.pptx`: revised twelve-slide
  presentation deck with acknowledgments and source references.
- `deliverables/Executive Summary.pdf`: paginated PDF edition of the evidence
  report, with every top-level section starting on a new page.
- `deliverables/SciVer Router Models Presentation.pdf`: PDF edition of the
  twelve-slide presentation.

## Reproduce The Router Modeling

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/verify_public_router_release.py --root .
python scripts/run_public_router_experiments.py
python scripts/build_sciver_router_presentation_report.py
python scripts/verify_public_router_release.py --root .
```

Use `python scripts/run_public_router_experiments.py --skip-existing` only as a
quick check of bundled generated artifacts. For a true rerun, start from a fresh
copy of this release and omit `--skip-existing`.

For exact-version reproduction of the checked public artifacts, install from
`requirements-lock.txt` instead of `requirements.txt`.

The expected public dataset has 811 rows, 12 model-comparison rows per
selection policy, and 24 router `metrics.json` files.

## Important Caveats

- Metrics are internal grouped estimates over this router-development dataset,
  not official untouched SciVer test-set benchmark results.
- The public source split labels (`val` and `test`) are retained for reporting.
  Router evaluation uses a separate grouped train/validation/test split by
  `paper_id`.
- Generated `.joblib` and `.pt` files are included as artifacts, but loading
  serialized model files from untrusted sources can execute code. Prefer
  rerunning the scripts from the public CSV when validating results.

## Acknowledgments

The 12 cognitive-difficulty dimensions were engineered by the
[AI Faithfulness in Scientific Reasoning](https://github.com/Erdos-Projects/summer26-ai-faithfulness-in-scientific-reasoning)
project. This public release acknowledges that contribution without claiming
to relicense upstream project code or documentation.

## References And Licensing

- SciVer paper: [SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification](https://arxiv.org/abs/2506.15569)
- SciVer dataset: [chengyewang/SciVer on Hugging Face](https://huggingface.co/datasets/chengyewang/SciVer), released under CC BY 4.0
- Project-authored code: MIT, as stated in `LICENSE`
- Dataset and third-party terms: see `DATA_LICENSE.md`
