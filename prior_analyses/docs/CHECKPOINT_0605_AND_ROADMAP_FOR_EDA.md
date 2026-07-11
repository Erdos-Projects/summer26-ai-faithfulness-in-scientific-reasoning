# Checkpoint 0605 and Roadmap for EDA

## Project Purpose

This project is designed to build a lightweight, GPU-free, and LLM-free baseline system for scientific claim verification using the SciVer database ([arXiv](https://arxiv.org/abs/2506.15569), [Hugging Face](https://huggingface.co/datasets/chengyewang/SciVer)).

The supervised task is binary claim verification:

```text
input:  one scientific claim + one chart/table evidence item
target: entailed or refuted
```

The goal is not to reproduce SciVer with a large language model or a retrieval-augmented generation system. Instead, the goal is to create a reusable evidence-claim feature store that supports simple downstream classifiers such as:

- `LogisticRegression`
- `LinearSVC`
- `RidgeClassifier`
- `SGDClassifier`
- nearest-neighbor baselines

The core research question is whether carefully constructed vector features from claims, visual evidence text, pair text, and images can support useful scientific claim verification baselines without expensive multimodal model training.

## Challenge and Novelty

SciVer is multi-modal by nature. Each observation is not a single row of ordinary tabular features. Instead, each example is a bundle containing:

- claim text
- chart or table image evidence
- caption, OCR, section, and local context text when available
- metadata such as paper ID, split, modality, and claim type
- a binary entailment/refutation label

Traditional tabular machine learning usually starts from a matrix-like dataset:

```text
rows = examples
columns = scalar or categorical features
```

SciVer does not naturally start in this form. The raw dataset has nested records, image files, paper/context files, and visual evidence references that need to be resolved before a classifier can use them.

The key technical step in this project is therefore to convert SciVer into a vector database. The converted database stores one supervised chart/table-claim pair per classifier example and attaches multiple named vectors to each pair:

- `claim_vec`
- `evidence_text_vec`
- `pair_text_vec`
- `image_vec`

This is different from a traditional tabular dataset because EDA and modeling must operate over both metadata and embedding spaces. We inspect not only label counts and claim types, but also nearest neighbors, vector distances, modality-specific clusters, and possible overlap between train/test retrieval neighborhoods.

The vector database is a research substrate rather than a final model. It makes downstream lightweight experiments possible while preserving the multi-modal structure of the original SciVer task.

## Work Completed So Far

1. Built the pipeline to download and convert the SciVer database.

   The current codebase can fetch the public Hugging Face dataset `chengyewang/SciVer`, save it under `data/raw/SciVer`, inspect the raw records, and convert usable chart/table evidence-claim pairs into processed Qdrant collections under `data/processed`.

2. Built [notebooks/01_sciver_qdrant_basic_usage.ipynb](../notebooks/01_sciver_qdrant_basic_usage.ipynb).

   This notebook demonstrates basic operations on the converted vector database, including how to connect to Qdrant, inspect collection information, retrieve sample records, print metadata, and run simple nearest-neighbor searches.

3. Built [notebooks/03_raw_to_processed_sciver_mapping.ipynb](../notebooks/03_raw_to_processed_sciver_mapping.ipynb).

   This notebook demonstrates the relationship between the raw SciVer dataset in `data/raw` and the converted evidence-claim database in `data/processed`. It helps verify how raw examples map to processed `pair_id`, `claim_id`, `evidence_item_id`, labels, modalities, and image paths.

4. Built [notebooks/02_direct_chart_logistic_regression.ipynb](../notebooks/02_direct_chart_logistic_regression.ipynb).

   This notebook trains a very basic baseline logistic regression model using converted chart-claim pair features with `claim_type = direct`. It is intended as a first sanity-check classifier, not as a tuned final baseline.

## Roadmap for EDA on Vectorized SciVer Database

EDA on this project should combine metadata summaries with vector-space diagnostics. A normal tabular EDA workflow is not enough because many important properties of the data only appear after claims, evidence text, pair text, and images are embedded.

At the initial stage, the EDA should stay focused on the checks that most directly affect whether downstream lightweight classifiers can be trusted.

### Metadata EDA

1. Dataset and schema checks.

   Verify collection sizes, vector names, vector dimensions, payload fields, missing values, and consistency between `sciver_claims`, `sciver_evidence_items`, and `sciver_pairs`.

2. Label, split, and modality distribution analysis.

   Summarize `entailed` versus `refuted` counts overall and by split, chart/table modality, claim type, paper ID, and evidence count. This establishes the supervised population before any model training.

3. Evidence text and image availability analysis.

   Measure caption length, OCR length, context length, section availability, empty evidence-text rates, missing images, and failed image embeddings. These checks identify whether weak evidence representations are caused by the dataset, parser, or embedding step.

### Embedding EDA

1. Nearest-neighbor inspection by vector type.

   For `claim_vec`, `evidence_text_vec`, `pair_text_vec`, and `image_vec`, inspect nearest neighbors manually and summarize whether retrieved examples share label, modality, claim type, paper, or visual style.

2. Similarity and distance distribution analysis.

   Compare cosine similarity distributions for same-label versus different-label pairs, same-paper versus different-paper pairs, and same-modality versus cross-modality pairs. This tests whether the embedding space contains useful geometric separation for lightweight classifiers.

3. Duplicate, near-duplicate, and split-neighborhood checks.

   Use vector similarity and metadata to identify repeated claims, repeated images, repeated captions, and near-duplicate evidence items across splits. This is especially important before using nearest-neighbor features or retrieval-based baselines.

The main difference from traditional tabular EDA is that EDA on a vector database is more geometric and topological than purely statistical. The vector database still has metadata columns, but the central objects are high-dimensional embeddings and their neighborhood structure.

- metadata EDA: counts, labels, splits, modalities, claim types, missing values
- embedding EDA: nearest neighbors, vector distances, local neighborhoods, clusters, manifolds, duplicate detection, and retrieval behavior

Some classical tabular EDA tasks are either unavailable or much less meaningful on raw vector database records:

- Direct interpretation of individual columns as human-readable variables is not available for embedding dimensions. A dimension of `claim_vec` or `image_vec` usually has no stable semantic meaning by itself.
- Per-column summary statistics such as mean, median, quantiles, skewness, and kurtosis are less informative for embedding dimensions than for ordinary scalar features.
- Pairwise feature correlation tables are hard to interpret because embedding coordinates are dense learned features rather than named variables.
- Standard missing-value analysis applies to metadata, but not cleanly to dense vectors once an embedding exists. A present vector can still be low-quality even when it is not missing.
- Outlier detection is less about extreme scalar values and more about isolated points, unusual neighborhoods, or cross-modal mismatches in embedding space.

Both metadata and embedding EDA are necessary. Metadata-only EDA can miss embedding-space problems, while vector-only EDA can hide label imbalance, split leakage, parser failures, or evidence availability issues.

## Roadmap for Improving the Logistic Regression Baseline

The current notebook [notebooks/02_direct_chart_logistic_regression.ipynb](../notebooks/02_direct_chart_logistic_regression.ipynb) is a minimal baseline. The next step is to tune it carefully while keeping the experiment lightweight, GPU-free, and LLM-free.

Recommended improvements:

1. Confirm the exact supervised population.

   Start with `claim_type = direct` and `modality = chart`, then separately evaluate whether adding `table` examples improves or weakens the baseline.

2. Enforce strict split handling.

   Use the original SciVer split labels. Do not use validation/test records to construct training features, tune preprocessing, or compute train-neighbor features.

3. Compare feature blocks.

   Evaluate logistic regression with different feature sets:

   ```text
   claim_vec only
   evidence_text_vec only
   pair_text_vec only
   image_vec only
   claim_vec + evidence_text_vec
   claim_vec + evidence_text_vec + pair_text_vec
   all vectors
   all vectors + abs(claim_vec - evidence_text_vec)
   all vectors + claim_vec * evidence_text_vec
   ```

4. Add categorical metadata features.

   Add one-hot features for `claim_type` and `modality` when the experiment includes more than one claim type or modality.

5. Tune regularization.

   Run a small grid over `C`, penalty type, class weighting, and solver options. Keep the grid modest so the baseline remains easy to reproduce.

6. Evaluate with more than accuracy.

   Report accuracy, F1, precision, recall, confusion matrix, and class-specific error rates. Since this is an entailment/refutation task, both false entailment and false refutation errors should be inspected.

7. Add calibration checks.

   Inspect predicted probabilities, reliability curves, and threshold sensitivity. Logistic regression can be useful as a calibrated baseline if the probabilities are well behaved.

8. Run modality-specific and claim-type-specific error analysis.

   Break down errors by chart/table, direct/non-direct claim type, paper ID, evidence text length, and nearest-neighbor similarity.

9. Save experiment artifacts.

    Store feature configuration, model hyperparameters, metrics, and predictions under `data/processed` or a future `reports/` directory so runs can be compared without rerunning the full pipeline.

10. Keep leakage rules explicit.

    Continue to ensure that labels, rationales, explanations, answer fields, and perturbed statements are never embedded or used as classifier input features.

11. Convert the notebook into a script when stable.

    Once the logistic regression workflow is stable, create a script version for reproducible command-line experiments while keeping the notebook as an explanatory walkthrough.
