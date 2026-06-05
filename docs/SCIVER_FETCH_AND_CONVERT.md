# Fetch and Convert SciVer

This guide starts from the public Hugging Face dataset `chengyewang/SciVer`, stores the raw snapshot under `data/raw`, and converts it into a local Qdrant evidence-claim vector database under `data/processed`.

The workflow does not use any LLM or answer-generating API. Labels are stored as metadata/targets only and are never embedded.

## 1. Environment

Create or reuse the repo-local virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name sciver-vector-db --display-name "SciVer Vector DB"
```

If the dataset or model downloads require authentication, log in with the Hugging Face CLI or set `HF_TOKEN`. Keep tokens out of git.

## 2. Fetch Raw SciVer

### 2.1 Set Up Hugging Face Authentication

SciVer is fetched from the Hugging Face dataset repo `chengyewang/SciVer`. Anonymous downloads can work, but this dataset contains many small JSON/image files, so authenticated downloads are more reliable and less likely to hit strict anonymous rate limits.

Create a read-only token in Hugging Face:

1. Sign in at <https://huggingface.co>.
2. Open **Settings** -> **Access Tokens**.
3. Create a token with read access.
4. Copy the token once. Treat it like a password.

Preferred persistent setup for this machine:

```bash
.venv/bin/hf auth login
```

Paste the read-only token when prompted. This stores the token in the Hugging Face CLI cache on your machine, outside this repo. Verify it:

```bash
.venv/bin/hf auth whoami
```

Alternative repo-local setup:

```bash
printf 'export HF_TOKEN=hf_your_read_only_token_here\n' > .env.local
chmod 600 .env.local
```

Before running fetch commands in a new terminal, load it:

```bash
set -a
source .env.local
set +a
```

`.env` and `.env.local` are ignored by git in this repo. Do not commit Hugging Face tokens.

### 2.2 Download the Raw Snapshot

Download the dataset snapshot into `data/raw/SciVer`:

```bash
.venv/bin/python scripts/fetch_sciver.py --raw-dir data/raw
```

On this machine, the Hugging Face Xet/CAS transfer path stalled during the SciVer snapshot download. The regular Hub download path completed successfully with Xet disabled:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/fetch_sciver.py --raw-dir data/raw
```

Use `--force` only when you intentionally want to refresh the local snapshot:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/fetch_sciver.py --raw-dir data/raw --force
```

Useful options:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/fetch_sciver.py \
  --raw-dir data/raw \
  --dataset-id chengyewang/SciVer \
  --revision main \
  --force
```

If the command appears paused, check the terminal output before stopping it. During the successful authenticated re-download, Hugging Face returned temporary HTTP `429` rate-limit responses after thousands of file requests. The SDK printed messages such as `Rate limited. Waiting 249.0s before retry`, slept for the requested retry window, then resumed and completed. This is not a token failure if `hf auth whoami` works; it means the downloader is obeying Hugging Face's server-side rate-limit window.

If a download is interrupted, rerun the command without `--force` to resume from the files already present under `data/raw/SciVer`.

The fetch step writes:

```text
data/raw/SciVer/
data/raw/SciVer/valset.json
data/raw/SciVer/testset.json
data/raw/SciVer/sciver_download_manifest.json
```

The exact image/context layout is resolved by the parser rather than hard-coded.

## 3. Inspect Raw Data

Check labels, claim types, modalities, usable examples, and skip reasons:

```bash
.venv/bin/python scripts/inspect_sciver.py
```

The default `--data-root` is `data/raw/SciVer`. To inspect another local copy:

```bash
.venv/bin/python scripts/inspect_sciver.py --data-root /path/to/SciVer
```

## 4. Convert to Qdrant

Build local Qdrant collections under `data/processed/qdrant_sciver` using both chart and table modalities:

```bash
.venv/bin/python scripts/build_sciver_qdrant.py --reset
```

Equivalent explicit command:

```bash
.venv/bin/python scripts/build_sciver_qdrant.py \
  --data-root data/raw/SciVer \
  --qdrant-path data/processed/qdrant_sciver \
  --processed-dir data/processed \
  --collection-prefix sciver \
  --modalities chart table \
  --pair-mode single_visual_only \
  --reset
```

Default text embeddings use `sentence-transformers/all-MiniLM-L6-v2`. Default image embeddings use `openai/clip-vit-base-patch32`. Use `--no-image-embeddings` for a text-only conversion.

The build writes:

```text
data/processed/qdrant_sciver/
data/processed/sciver_claims_manifest.parquet
data/processed/sciver_evidence_items_manifest.parquet
data/processed/sciver_pairs_manifest.parquet
data/processed/sciver_build_report.json
```

## 5. Smoke Query and Export Features

Run a retrieval smoke test:

```bash
.venv/bin/python scripts/query_sciver_qdrant_smoke.py \
  --qdrant-path data/processed/qdrant_sciver \
  --collection sciver_pairs \
  --limit 5
```

Export one parquet row per pair with vectors and metadata:

```bash
.venv/bin/python scripts/export_pair_features.py \
  --qdrant-path data/processed/qdrant_sciver \
  --collection sciver_pairs \
  --out data/processed/sciver_pair_features.parquet
```

Downstream scikit-learn baselines can load the export and construct:

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

Keep split-aware experiments strict: do not query validation/test records when constructing train-neighbor features.

## 6. One-Command Preparation

For convenience, run the full fetch/build/smoke/export sequence:

```bash
.venv/bin/python scripts/prepare_sciver.py --reset
```

This wrapper uses the same defaults as the individual commands.

## 7. Raw-to-Processed Mapping Notebook

After fetch and conversion, execute:

```bash
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/03_raw_to_processed_sciver_mapping.ipynb
```

The notebook shows how raw `valset.json`/`testset.json` examples map to processed `pair_id`, `claim_id`, `evidence_item_id`, labels, modalities, and image paths.

## Notes

- `data/raw/` and `data/processed/` are ignored by git.
- Qdrant local payload indexes are accepted but only meaningful in server mode.
- `single_visual_only` remains the fully implemented pair mode.
- Rationale, explanation, answer, perturbed, and label fields are never embedded.
