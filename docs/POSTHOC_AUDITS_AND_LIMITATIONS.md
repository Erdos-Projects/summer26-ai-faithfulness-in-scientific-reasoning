# Post-hoc Audits and Finalized Limitations

Follow-up to `FAITHFULNESS_PROBE_FINDINGS.md`. That report established that the
lightweight baseline's ~0.55 accuracy is not robust signal; this document
closes the verification gaps identified in review and consolidates the
project's limitations. All numbers were produced by
`scripts/posthoc_audits.py` on the rederived pipeline (2026-06-09);
machine-readable results: `docs/posthoc_audit_results.json`.

## Audit 1 — Duplication and split integrity

| Population | Pairs | Unique images | Multi-pair images | Both-label images | Images spanning val/test | Papers in both splits |
|---|---|---|---|---|---|---|
| All chart+table pairs | 1,500 | 1,399 | 101 | 48 | **0** | **0** |
| Direct+chart (probe subset) | 416 | 416 | 0 | 0 | **0** | **0** |

- **The official val/test split is leak-free**: no image and no paper appears
  on both sides. The single-split image edge (0.565) cannot be explained by
  cross-split memorization — "lucky split" survives its strongest rival
  explanation.
- **The probe subset is internally clean too**: every direct+chart pair has its
  own image, so the findings report's repeated CV was not inflated by images
  crossing folds.
- The duplication that does exist (101 multi-pair images, 48 carrying both
  labels) lives entirely in the analytical/table subsets. Any future modeling
  on all 1,500 pairs must use grouped CV and respect the deterministic
  image-only accuracy ceiling of **0.968** it implies.

## Audit 2 — Grouped vs plain cross-validation

StratifiedGroupKFold by `paperid` (no paper spans a fold boundary) vs plain
StratifiedKFold, 5 folds x 10 seeds, balanced accuracy, direct+chart:

| Feature set | Grouped by paper | Plain |
|---|---|---|
| claim_vec | 0.476 ± 0.056 | 0.479 ± 0.052 |
| image_vec | 0.519 ± 0.047 | 0.519 ± 0.049 |
| claim+image | 0.480 ± 0.050 | 0.490 ± 0.044 |

Identical to within noise. The CV-collapse result in the findings report is
not an artifact of paper-level leakage in either direction.

## Audit 3 — Family-wise (max-statistic) permutation test

Six feature blocks were evaluated on the same official split, so the best one
benefits from selection. Each of 300 permutations refits **all six blocks** on
the same shuffled training labels and keeps the best test accuracy:

- Observed best: claim+image, **0.576**
- Null best-of-six: mean 0.526, 95th percentile 0.562, max 0.616
- **Family-wise p = 0.027**

So the strongest single-split result remains nominally significant even after
multiple-comparisons correction. The decisive evidence against it is not
selection bias but **resampling**: under CV — which trains on ~333 rows versus
the official split's 140 — claim+image drops to 0.48–0.49. A real effect
should improve with more training data; this one vanishes. Combined with the
preprocessing fragility documented in the findings report's reproduction note
(full stack 0.565 → 0.547 from a PIL-vs-torchvision image-resize difference),
the single-split edge is best read as a favorable train/test draw, possibly
helped by population differences between val papers and test papers (the two
pools share no papers, so they are different paper samples by construction).

## Audit 4 — Caption/context recoverability from `paper_path`

The findings report identified that evidence text is empty for all 1,500
pairs because captions live in the `paper_path` JSONs the parser never reads.
Checking those JSONs directly:

- Paper JSON found for **1,500/1,500** pairs.
- Nonempty caption recoverable for **1,245 pairs (83%)**, mean 221 characters
  (figures via `image_paths[n].caption`, tables via `tables[n].capture`; the
  remaining 17% are mostly filename-pattern variants the audit's regexes do
  not cover, so 83% is a lower bound).
- Related-sentence context (`image_result`/`table_result` →
  `related_sentences_ids` into section text) available for **1,177 pairs (78%)**.

The evidence-pipeline fix is therefore feasible, not speculative: a parser
extension can populate `evidence_text_vec` with real captions and local
context for at least ~4 in 5 pairs.

## Audit 5 — Statistical power

Minimum detectable accuracy vs 0.5 at 80% power (binomial, normal
approximation):

| n (test size) | One-sided α=0.05 | Two-sided α=0.05 | Role in this project |
|---|---|---|---|
| 83 | 0.636 | 0.654 | one CV fold |
| 140 | 0.605 | 0.618 | official val (train) |
| 276 | **0.575** | **0.584** | official test |
| 416 | 0.561 | 0.569 | direct+chart pool |
| 996 | 0.539 | 0.544 | all-pairs test |

The headline experiment could only reliably detect effects of ~7.5+ points;
the observed 0.565–0.576 sits at the detection threshold, and single CV folds
cannot detect anything under ~14 points. Plausible small effects (2–5 points)
were undetectable by design. Conversely, at n = 996 even tiny perturbation
artifacts become "significant" — which is exactly what the TF-IDF shortcut
(0.537, p = 0.03) shows.

## Finalized limitations

1. **Underpowered design.** With 140 training rows, thousands of features, and
   a 276-example test set, only effects ≥ ~7.5 points were detectable. Every
   reported accuracy difference smaller than that is uninterpretable.
2. **Single-split evidence is weak and fragile.** The best split result
   survives family-wise correction (p = 0.027) but collapses under grouped and
   plain CV despite more training data, and its point estimate moves ~2 points
   under numerically irrelevant preprocessing changes.
3. **Degenerate evidence representation.** Evidence text was empty for
   1,500/1,500 pairs; the claim-only features that remain cannot decide
   entailment by construction. Captions/context are recoverable for ≥83%/78%
   of pairs but were never wired in.
4. **Encoder mismatch with the task.** CLIP-base produces global style/layout
   vectors and cannot read axis values, units, or trends off charts; no OCR
   was applied.
5. **All-pairs analyses carry a duplication caveat.** 48 images appear with
   both labels (image-only ceiling 0.968), and 101 images span multiple pairs
   — future all-pairs modeling needs grouped CV by paper and image-aware
   splits. The direct+chart subset and the official split itself are clean.
6. **Larger-n text "signal" is artifact-prone.** At n ≈ 1,000 the design
   detects ~4-point effects, and the only text effect found at that scale
   (TF-IDF 0.537) is a perturbation fingerprint, not verification.
7. **Scope.** These findings indict this feature pipeline (frozen MiniLM +
   CLIP-base + linear head on 416 examples), not the SciVer task: multimodal
   LLMs score well above chance on it. Conclusions should not be generalized
   to other encoders, training scales, or evidence representations.

## Implications for future directions

- **If continuing the lightweight track:** wire real captions/context from
  `paper_path` into `evidence_text_vec` (Audit 4 shows this is feasible),
  replace CLIP-base with a chart-native encoder or add OCR, and size the
  experiment so the minimum detectable effect is below the effect you care
  about (Audit 5's table gives the targets). Use grouped, image-aware CV for
  anything beyond direct+chart.
- **If pivoting:** the dataset, parser, embeddings, and the statistical
  machinery used here (permutation nulls, grouped CV, max-statistic
  correction, power analysis) transfer directly to difficulty modeling /
  confidence-metric work over multiple models, where the per-item signal comes
  from model behavior instead of frozen features.
