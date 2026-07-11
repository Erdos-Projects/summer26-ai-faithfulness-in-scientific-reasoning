# SciVer Qdrant Vector DB

This repo fetches SciVer from Hugging Face or converts an existing local SciVer `valset.json`/`testset.json` copy into a Qdrant feature store for lightweight supervised chart/table-claim classification. It does not call external LLMs or generate evidence descriptions.

## Commands

Inspect local data:

```bash
python scripts/fetch_sciver.py --raw-dir data/raw
python scripts/inspect_sciver.py
```

Build local Qdrant collections:

```bash
python scripts/build_sciver_qdrant.py \
  --data-root data/raw/SciVer \
  --qdrant-path data/processed/qdrant_sciver \
  --processed-dir data/processed \
  --collection-prefix sciver \
  --modalities chart table \
  --pair-mode single_visual_only \
  --reset
```

Smoke-query the result:

```bash
python scripts/query_sciver_qdrant_smoke.py --qdrant-path data/processed/qdrant_sciver --collection sciver_pairs --limit 5
```

Export classifier features:

```bash
python scripts/export_pair_features.py \
  --qdrant-path data/processed/qdrant_sciver \
  --collection sciver_pairs \
  --out data/processed/sciver_pair_features.parquet
```

See `docs/SCIVER_FETCH_AND_CONVERT.md` for the full fetch-first workflow.

## Collections

The default collection prefix is `sciver`, producing:

| Collection | Unit | Named vectors |
| --- | --- | --- |
| `sciver_claims` | one claim instance | `claim_vec` |
| `sciver_evidence_items` | one visual chart/table item | `evidence_text_vec`, `image_vec` when enabled |
| `sciver_pairs` | one supervised chart/table-claim pair | `claim_vec`, `evidence_text_vec`, `pair_text_vec`, `image_vec` when enabled |

Qdrant point IDs are deterministic UUIDv5 values derived from human-readable IDs. The human-readable IDs are stored in payload fields such as `claim_id`, `evidence_item_id`, and `pair_id`. `pair_id` includes both a claim hash and an evidence hash because the local SciVer snapshot contains repeated `request_id`/evidence combinations for different claims.

## Vector Dimensions

Defaults:

| Vector | Default model | Dimension |
| --- | --- | --- |
| `claim_vec` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `evidence_text_vec` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `pair_text_vec` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `image_vec` | `openai/clip-vit-base-patch32` | 512 |

Model names are configurable with `--text-model` and `--image-model`. Disable image embeddings with `--no-image-embeddings`.

## Payload Fields

`sciver_claims` payload:

`claim_id`, `paperid`, `request_id`, `split`, `claim`, `claim_type`, `label`, `label_id`, `evidence_item_ids_json`, `feature_version`, `embedding_model_text`.

`sciver_evidence_items` payload:

`evidence_item_id`, `paperid`, `split`, `modality`, `item_key`, `image_path`, `caption`, `ocr_text`, `context_text`, `section_json`, `paper_path`, `feature_version`, `embedding_model_text`, `embedding_model_image`.

`sciver_pairs` payload:

`pair_id`, `claim_id`, `evidence_item_id`, `paperid`, `request_id`, `split`, `claim`, `label`, `label_id`, `claim_type`, `modality`, `image_path`, `section_json`, `n_evidence_items`, `gold_pair`, `pair_mode`, `feature_version`, `embedding_model_text`, `embedding_model_image`.

Payload indexes are created for `split`, `label`, `label_id`, `paperid`, `claim_type`, `modality`, `pair_mode`, `gold_pair`, and `n_evidence_items`.

## Embedded Text

`claim_vec` embeds only:

```text
{claim}
```

`evidence_text_vec` embeds non-answer evidence text only: caption, OCR text if already present, local context text if present, and section title if present.

`pair_text_vec` embeds:

```text
CLAIM:
{claim}

EVIDENCE:
{caption}
{ocr_text}
{context_text}
```

## Leakage Rules

Gold labels are payload targets only; they are never embedded. Rationale, explanation, answer, perturbed fields, `origin_statement`, and `perturbed_statement` are never embedded. The converter does not use LLM-generated descriptions.

For downstream train-neighbor features, do not query validation/test records when constructing training features. Build split-aware retrieval filters before computing any supervised retrieval-derived features.

## Manifests

The build writes:

| File | Purpose |
| --- | --- |
| `data/processed/sciver_claims_manifest.parquet` | claim payload manifest |
| `data/processed/sciver_evidence_items_manifest.parquet` | visual evidence payload manifest |
| `data/processed/sciver_pairs_manifest.parquet` | pair payload manifest |
| `data/processed/sciver_build_report.json` | build counts, dimensions, skips, models, CLI args |

Embedding caches live under `data/processed/embedding_cache/`.

## Classifier Feature Export

`scripts/export_pair_features.py` exports one parquet row per pair with metadata, `label_id`, and vector columns. A downstream scikit-learn loader can construct:

```python
X = concat(
    claim_vec,
    evidence_text_vec,
    pair_text_vec,
    image_vec,
    abs(claim_vec - evidence_text_vec),
    claim_vec * evidence_text_vec,
    one_hot(claim_type),
    one_hot(modality),
)
y = label_id
```

## Known Limitations

Only `single_visual_only` is fully implemented. `explode_visuals` and `aggregate_visuals` intentionally raise `NotImplementedError` in this first version. Optional OCR is not implemented; the converter only uses OCR text already present in local SciVer/context records. Image embeddings require local model availability or network/model cache access through Hugging Face tooling.
