# Public Router Release Manifest

This manifest lists the files intended for public review of the SciVer router
modeling package. The public package reproduces router training/evaluation from
sanitized derived features; it does not reproduce private local VLM inference.

## Included Public Artifacts

- `README.md`
- `PUBLIC_RELEASE_MANIFEST.md`
- `.gitignore`
- `.github/workflows/verify-public-release.yml`
- `LICENSE`
- `DATA_LICENSE.md`
- `requirements.txt` (router-only public dependencies)
- `requirements-lock.txt` (tested exact versions for metric reproduction)
- `router_dataset_public.csv`
- `router_dataset_public_audit.json`
- `chart_tda_features_public.csv`
- `claim_nlp_features_public.csv`
- `Data/derived/model_router/public_router/router_dataset_public.csv`
- `Data/derived/model_router/public_router/router_dataset_public_audit.json`
- `Data/derived/model_router/public_router/chart_tda_features_public.csv`
- `Data/derived/model_router/public_router/claim_nlp_features_public.csv`
- `Data/derived/model_router/public_router/public_router_checksums.json`
- `Data/derived/model_router/public_router/public_router_run_manifest.json`
- `Data/derived/model_router/public_router/runs/`
- `Data/derived/model_router/public_router/comparisons/`
- `scripts/run_public_router_experiments.py`
- `scripts/verify_public_router_release.py`
- `scripts/build_sciver_router_presentation_report.py`
- `scripts/train_model_router_ridge.py`
- `scripts/train_model_router_xgboost.py`
- `scripts/train_model_router_torch_mlp.py`
- `scripts/router_model_lib.py`
- `scripts/torch_mlp_router_lib.py`
- `scripts/compare_router_models.py`
- `docs/reproduce_router_experiments.md`
- `docs/public_data_statement.md`
- `docs/public_review_iterations.md`
- `docs/sciver_router_presentation.html`
- `outputs/sciver_router_models_presentation.pptx`
- `outputs/sciver_router_models_presentation_montage.png`

## Excluded Private/Raw Artifacts

- Raw local VLM JSONL outputs and raw prompts/responses.
- Raw SciVer claims, captions, images, paper JSONs, and labels.
- SQLite annotation databases.
- LM Studio caches, model weights, and local environment files.
- Internal exploratory outputs not needed to reproduce the public router results.

## Current Public Dataset

- Rows: 811
- Candidate VLMs: Qwen3-VL, Pixtral12B, Kimi-VL-A3B
- Feature sets: base, chart TDA, claim NLP, chart TDA + claim NLP
- Router families: Ridge/ElasticNet, XGBoost, PyTorch MLP
- Selection policies: mean-regret-first and top-1-hit-first

## Reproduction Boundary

Public reviewers should reproduce router modeling from
`router_dataset_public.csv`. The maintainer-only release preparation script
rebuilds the sanitized package from private derived inputs and is not part of
the public rerun path.

## Attribution

- SciVer paper: <https://arxiv.org/abs/2506.15569>
- SciVer dataset: <https://huggingface.co/datasets/chengyewang/SciVer>
  (CC BY 4.0)
- Cognitive-difficulty dimensions:
  <https://github.com/Erdos-Projects/summer26-ai-faithfulness-in-scientific-reasoning>
