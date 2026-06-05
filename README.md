# SciVer Benchmarks

This repo builds a lightweight, GPU-free, and LLM-free baseline workflow for scientific claim verification with the SciVer dataset. The main artifact is a vectorized SciVer evidence-claim database that can support simple downstream classifiers such as logistic regression, linear SVMs, ridge classifiers, SGD classifiers, and nearest-neighbor baselines.

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Local SciVer snapshot downloaded from Hugging Face
│   └── processed/           # Qdrant database, manifests, exports, and caches
├── docs/                    # Project documentation and research roadmap
├── notebooks/               # Usage, mapping, and baseline modeling notebooks
├── sciver_vector_db/        # Core parsing, fetching, embedding, and Qdrant helpers
├── scripts/                 # Command-line entry points for fetch/build/query/export
├── tests/                   # Parser and fetching tests with synthetic fixtures
├── requirements.txt         # Runtime and notebook dependencies
└── pyproject.toml           # Project metadata and tooling configuration
```

`data/raw/` and `data/processed/` are ignored by git because they contain downloaded data, generated vector databases, embedding caches, and model-ready exports.

## Main Documentation

- Read [docs/SCIVER_FETCH_AND_CONVERT.md](docs/SCIVER_FETCH_AND_CONVERT.md) for one-shot setup of the project, including Hugging Face authentication, fetching SciVer, converting to Qdrant, smoke querying, and feature export.
- Read [docs/SCIVER_VECTOR_DB.md](docs/SCIVER_VECTOR_DB.md) for the embedding details, collection schema, vector names, payload fields, leakage rules, and classifier feature export format.
- Read [docs/CHECKPOINT_0605_AND_ROADMAP_FOR_EDA.md](docs/CHECKPOINT_0605_AND_ROADMAP_FOR_EDA.md) for the current project checkpoint and the roadmap for the next EDA and baseline-modeling stage.
