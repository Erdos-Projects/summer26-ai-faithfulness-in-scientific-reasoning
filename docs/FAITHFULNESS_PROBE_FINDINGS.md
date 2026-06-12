# Faithfulness Probe — Findings

Consolidated record of an investigation into whether the lightweight SciVer
chart/table baseline carries real, leakage-free signal for entailed-vs-refuted
classification, or whether the reported ~0.55 accuracy is consistent with random
chance. All numbers below were produced during the analysis session; this is a
report, not new work.

## TL;DR

- The text features carry **no** leakage-free signal on the notebook's subset (claim text alone = chance).
- The image edge seen on a single split (0.565) **does not survive cross-validation** (0.514 ± 0.047, p = 0.29) — it was a lucky split.
- A single train/test accuracy here has a ~5-point standard deviation, so **0.55 is <1.2 SD from chance**. It is not distinguishable from random.
- Root cause: the leakage-free features do not contain enough faithful information to do the task as currently built.

## Dataset / data shape

- Source: SciVer (`chengyewang/SciVer`), local snapshot; binary task `entailed` (1) vs `refuted` (0).
- Parsed examples (val + test): 3,000. Supervised chart/table pairs (`single_visual_only`): **1,500** — val 504, test 996.
- Notebook 02's population (`modality == chart` & `claim_type == direct`): **416 pairs** — val 140, test 276.
- Label balance (≈ 50/50, so chance ≈ 0.500): full val 505/495; direct+chart test 137 entailed / 139 refuted, val 76/64.
- Image corpus: 3,756 figures, 1,273.9 MiB raw (~1,698 MiB base64); largest 9.2 MiB; **3 images exceed the 10 MB/image base64 API cap**.

## Features (what is actually embedded)

- `claim_vec` — MiniLM (`sentence-transformers/all-MiniLM-L6-v2`), **384-d**, of the `claim` string only.
- `evidence_text_vec` — MiniLM 384-d of caption + OCR + context + section title. **Empty for all 1,500 pairs**: those fields don't exist in the val/test records; captions/context live in the `paper_path` JSON, which the parser never reads.
- `pair_text_vec` — MiniLM 384-d of `"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}"` → effectively the claim plus a constant suffix (evidence empty).
- `image_vec` — CLIP (`openai/clip-vit-base-patch32`) image encoder, **512-d**, `get_image_features` (L2-normalized).
- Derived blocks in notebook 02: `abs(claim_vec − evidence_text_vec)`, `claim_vec * evidence_text_vec` — degenerate, since `evidence_text_vec` is constant.
- Leakage handling: of the three claim strings per example (`origin_statement`, `perturbed_statement`, and the `claim` actually used), **only `claim` is embedded**; `origin_statement` / `perturbed_statement` / `perturbed_explanation` are excluded by the leakage filter (they are the answer key).
- Note: CLIP image (512-d) and MiniLM text (384-d) are **not** in a shared space and are not even the same dimensionality. This is fine for a concatenate-then-classify model (no cross-modal cosine similarity is used); alignment would only matter for retrieval.

## Methods / parameters

- Classifier (matches notebook 02): `StandardScaler` → `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`.
- Text-shortcut probe: `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → balanced `LogisticRegression`, on the `claim` string only.
- Label-permutation null: shuffle **training** labels, refit, score the real test set; 300 permutations (single-split), 200 (CV). p = (#null ≥ observed + 1)/(N + 1). The tables in Analyses 1–2 record the upper tail only (the original recording); full two-sided null distributions and two-sided p-values were added later — see "Two-sided null recording (2026-06-10)" below.
- Bootstrap CI: 2,000 resamples of the test predictions (95% percentile interval).
- Cross-validation: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)`, scoring = balanced accuracy.
- Seed 0 throughout. Analytic chance bands (normal approx): n = 276 → 0.50 ± 0.059; n = 996 → 0.50 ± 0.031.

## Analysis 1 — claim-text-only TF-IDF probe (does the claim string leak the answer?)

The claim cannot honestly reveal entailment, so any above-chance text-only accuracy = a perturbation artifact.

| Subset | n_train / n_test | Accuracy | Majority baseline | Null mean / 95th / max | Perm p |
|---|---|---|---|---|---|
| direct + chart | 140 / 276 | 0.507 | 0.504 | 0.495 / 0.547 / 0.565 | **0.379** |
| all chart + table | 504 / 996 | 0.537 | 0.504 | 0.500 / 0.530 / 0.549 | **0.030** |

- Notebook subset: **no signal, no shortcut** (p = 0.38). Even a zero-skill (label-shuffled) model reaches 0.547 at the 95th percentile and 0.565 at max — i.e. **0.55 is inside the chance distribution** for this n.
- Full set: a **small but significant text-only shortcut** (0.537, p = 0.03) — a perturbation-artifact fingerprint, not verification. Significant only because n = 996.

## Analysis 2 — embedding ablation (original val→test split, notebook 02 classifier)

n_train = 140, n_test = 276, chance = 0.500.

| Feature set | dims | Accuracy | Balanced acc | 95% bootstrap CI | Null 95th / max | Perm p |
|---|---|---|---|---|---|---|
| `claim_vec` only | 384 | 0.493 | 0.493 | [0.435, 0.551] | 0.543 / 0.580 | 0.591 |
| `image_vec` only | 512 | 0.565 | 0.565 | [0.507, 0.623] | 0.544 / 0.580 | **0.023** |
| `claim + image` | 896 | 0.576 | 0.576 | [0.518, 0.634] | 0.543 / 0.565 | **0.003** |
| full notebook stack | ~2.4k | 0.565 | 0.565 | [0.507, 0.623] | 0.551 / 0.620 | **0.023** |

- Text is dead: `claim_vec` alone = 0.493 (p = 0.59), exactly chance.
- The only signal is image-driven and marginal (`image_vec` 0.565, p = 0.02; CI lower bound barely above 0.50).
- The full stack equals image-only — the empty/claim-derived blocks add nothing.

## Analysis 3 — repeated cross-validation (the robustness check)

Repeated 5-fold × 10 CV over all 416 direct+chart pairs; balanced accuracy; CV permutation test.

| Feature set | CV balanced acc | ±1 SD | CV perm p |
|---|---|---|---|
| `claim_only` | 0.464 ± 0.054 | [0.410, 0.518] | 0.905 |
| `image_only` | **0.514 ± 0.047** | [0.467, 0.561] | **0.294** |
| `claim + image` | 0.488 ± 0.044 | [0.444, 0.532] | 0.617 |

- The single-split image edge (0.565, p = 0.02) **collapses to 0.514, p = 0.29** under resampling → the official split was simply favorable.
- The ±0.047 SD is the headline: with ~140 training rows and hundreds–thousands of features, any single accuracy from ~0.45 to ~0.56 is within one SD of a coin flip.

## Conclusion

- The reported **~0.55 is not statistically distinguishable from chance**: it is <1.2 SD from 0.50, the zero-skill null reaches 0.55+ on this n, and the image edge does not survive cross-validation.
- Text contributes nothing leakage-free (confirmed three ways); any larger-sample text signal is a synthetic-data shortcut, not verification.
- The leakage-free features cannot faithfully support the task as built: the admissible claim is undecidable by design, evidence text is empty (captions unused in `paper_path`), and CLIP-base global vectors cannot read dense chart values.
- To attempt the task faithfully would require pulling real captions/context from `paper_path`, a chart/document-native image encoder instead of CLIP-base, and far more than 140 training examples.

Post-hoc audits (duplication/split integrity, grouped CV, family-wise
permutation, caption recoverability, power analysis) and the project's
finalized limitations: `docs/POSTHOC_AUDITS_AND_LIMITATIONS.md`.

## Reproducibility notes

- Encoders: `sentence-transformers/all-MiniLM-L6-v2` (text, 384-d), `openai/clip-vit-base-patch32` (image, 512-d), both L2-normalized — identical to the project pipeline (`sciver_vector_db/embeddings.py`).
- Pairs and leakage filtering via `sciver_vector_db/parsing.py` (`build_pair_records`, `single_visual_only`).
- Analyses were run on the embeddings the pipeline produces; classifier and split match `notebooks/02_direct_chart_logistic_regression.ipynb`.

## Reproduction note (2026-06-09)

The full pipeline was rederived from the raw SciVer snapshot in a fresh CPU-only
environment (rebuild Qdrant → export parquet → re-execute notebook 02 → rerun
`scripts/faithfulness_probe.py`). Machine-readable results:
`docs/faithfulness_probe_results.json`.

- **Exact matches:** all dataset counts (1,500 pairs; 416 direct+chart; empty
  evidence text 1,500/1,500 — also confirmed directly against the raw JSON,
  whose records contain no caption/OCR/context keys at all); the TF-IDF probe
  to the third decimal including its null distribution (0.507/p = 0.379 and
  0.537/p = 0.030); and every individual-block ablation accuracy (claim 0.493,
  image 0.565, claim+image 0.576). Repeated-CV results match within a point
  (image 0.515 ± 0.041, p = 0.45) and reach the same conclusions.
- **One deviation:** the full notebook stack reproduced at **0.547 (perm
  p = 0.08)** instead of 0.565 (p = 0.023). The re-executed notebook 02 also
  yields exactly 0.547, so the rerun is internally consistent. Most of the
  direct+chart image embeddings were byte-identical cache hits from the original
  session; a handful of images were re-embedded under transformers' PIL
  image-processor fallback (torchvision absent in the new environment), and that
  numerically trivial preprocessing difference moved the 2,432-feature result by
  ~2 points across the nominal-significance boundary.
- **Interpretation:** the deviation is inside the report's own error bars
  (bootstrap CI [0.489, 0.609]) and *strengthens* the conclusion — the headline
  number is preprocessing-noise-dominated, exactly as argued above.
- Small permutation p-value shifts on unchanged accuracies (e.g. image
  0.023 → 0.030) are RNG-stream differences only; accuracies themselves are
  RNG-independent.

## Two-sided null recording (2026-06-10)

The permutation tests originally recorded only the upper tail of each null
distribution (95th percentile and maximum), because the one-sided hypothesis
test only needs that tail. Under-performance is as informative as
over-performance, so `scripts/faithfulness_probe.py` and
`scripts/posthoc_audits.py` now record the full distribution
(min / 5th / mean / 95th / max) and a two-sided permutation p-value: the
fraction of null scores at least as far from 0.500 as the observed score, in
either direction. The scripts were rerun in the rederived environment with
unchanged seeds and draw order; all previously published values reproduced
exactly. Machine-readable results: `docs/faithfulness_probe_results.json`,
`docs/posthoc_audit_results.json`.

| Probe (rederived run) | Accuracy | Null min / 5th / mean / 95th / max | Perm p (one- / two-sided) |
|---|---|---|---|
| TF-IDF claim text, direct+chart | 0.507 | 0.373 / 0.446 / 0.495 / 0.547 / 0.565 | 0.379 / 0.834 |
| TF-IDF claim text, all pairs (n=996) | 0.537 | 0.451 / 0.473 / 0.500 / 0.530 / 0.549 | 0.030 / 0.043 |
| `claim_vec` only | 0.493 | 0.420 / 0.453 / 0.500 / 0.547 / 0.565 | 0.648 / 0.847 |
| `evidence_text_vec` only | 0.504 | 0.504 / 0.504 / 0.504 / 0.504 / 0.504 | 1.000 / 1.000 |
| `pair_text_vec` only | 0.489 | 0.409 / 0.453 / 0.501 / 0.551 / 0.598 | 0.648 / 0.811 |
| `image_vec` only | 0.565 | 0.406 / 0.449 / 0.500 / 0.551 / 0.580 | 0.030 / 0.060 |
| `claim + image` | 0.576 | 0.424 / 0.456 / 0.501 / 0.551 / 0.572 | 0.003 / 0.007 |
| full notebook stack | 0.547 | 0.388 / 0.446 / 0.499 / 0.551 / 0.572 | 0.080 / 0.166 |

(`evidence_text_vec` is degenerate: the features are all-zero, the classifier
makes constant predictions, and every permutation scores the majority rate
0.504.)

Findings:

- Every null distribution is centered on 0.500 (means 0.495–0.504) and
  symmetric: the 5th percentiles (0.446–0.473) mirror the 95th percentiles
  (0.530–0.551). Any earlier appearance of an upward skew in the nulls was an
  artifact of recording only maxima.
- No probe scores significantly **below** chance; the lower tail is
  unremarkable everywhere.
- Under the two-sided test, `image_vec` is no longer nominally significant
  (p = 0.060 vs one-sided 0.030). `claim + image` stays significant
  (p = 0.007) on the single split but still collapses under cross-validation
  (Analysis 3), which remains the decisive evidence.
- Family-wise two-sided results (best-of-six and worst-of-six nulls):
  `docs/POSTHOC_AUDITS_AND_LIMITATIONS.md`, Audit 3.
